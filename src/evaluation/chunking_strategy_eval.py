"""Evaluate chunking strategies over raw NQ-retrieval rows.

This module is intentionally analysis-oriented: it uses ``long_answers`` as labels to compare
candidate chunking policies, but production chunking must not depend on those labels.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Literal

from rank_bm25 import BM25Okapi

CHUNK_ID_NAMESPACE = uuid.UUID("9c92e2fa-5f06-4db4-9cd9-2a69695be660")
BOILERPLATE_EXACT = {"vte", "v t e", "v\nt\ne", "external links", "references", "see also", "notes"}
HEADER_TYPES = {"list_definition", "section", "header"}
LIST_OR_TABLE_TYPES = {"list", "table"}
INTRO_MARKERS = ("including", "include", "includes", "such as", "the following", "for example")

ChunkingStrategy = Literal[
    "raw_candidate",
    "whole_row",
    "dpr_window",
    "parent_dedup",
    "min_context",
    "parent_child",
]


@dataclass(slots=True)
class Candidate:
    """One source candidate from a raw NQ row."""

    index: int
    text: str
    passage_type: str | None
    tokens: int


@dataclass(slots=True)
class RawRow:
    """Raw row with normalized candidate records."""

    row_ordinal: int
    title: str | None
    question: str | None
    document_url: str | None
    long_answer_indices: list[int]
    candidates: list[Candidate]


@dataclass(slots=True)
class EvalChunk:
    """One candidate chunk emitted by a strategy."""

    chunk_id: str
    strategy: str
    row_ordinal: int
    start_candidate_idx: int
    end_candidate_idx: int
    text: str
    passage_types: list[str | None]
    title: str | None = None
    question: str | None = None
    parent_candidate_idx: int | None = None
    chunk_kind: str = "candidate"
    context_text: str | None = None

    @property
    def token_count(self) -> int:
        return token_count(self.text)

    @property
    def context_token_count(self) -> int:
        return token_count(self.context_text or self.text)


@dataclass(slots=True)
class DatasetProfile:
    """Shape statistics for raw rows and long-answer labels."""

    sample_rows: int
    total_candidates: int
    avg_candidates_per_row: float
    median_candidates_per_row: float
    candidate_token_quantiles: dict[str, int]
    short_candidate_counts: dict[str, int]
    short_candidate_ratios: dict[str, float]
    passage_type_counts: dict[str, int]
    gold_long_answer_count: int
    gold_token_avg: float
    gold_token_median: float
    gold_token_quantiles: dict[str, int]
    gold_type_counts: dict[str, int]
    nearby_nested_parent_ratio: float


@dataclass(slots=True)
class StrategyProfile:
    """Structural metrics for a chunking strategy."""

    strategy: str
    chunks: int
    avg_tokens: float
    median_tokens: float
    tiny_chunks_le_10: int
    tiny_chunks_le_20: int
    huge_chunks_gt_300: int
    relevant_chunks: int
    relevant_token_median: float
    duplicate_like_ratio: float


@dataclass(slots=True)
class RetrievalMetrics:
    """BM25 proxy retrieval metrics for one chunking strategy."""

    query_count: int
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    precision_at_k: float
    context_budget_recall: dict[str, float]
    relevant_token_density_at_k: float


@dataclass(slots=True)
class StrategyResult:
    """Combined structural and retrieval result."""

    profile: StrategyProfile
    retrieval: RetrievalMetrics


@dataclass(slots=True)
class ExampleChunk:
    """Human-readable example for reports."""

    chunk_id: str
    title: str | None
    question: str | None
    span: str
    passage_types: list[str | None]
    tokens: int
    text: str


@dataclass(slots=True)
class ChunkingEvaluationReport:
    """Full report payload for JSON/Markdown output."""

    dataset_profile: DatasetProfile
    strategy_results: dict[str, StrategyResult]
    example_chunks: list[ExampleChunk] = field(default_factory=list)


def load_raw_rows(path: Path, *, max_rows: int | None = None) -> list[RawRow]:
    """Load raw NQ rows from JSONL for analysis."""

    rows: list[RawRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_ordinal, line in enumerate(handle):
            if max_rows is not None and len(rows) >= max_rows:
                break
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError("raw dataset line must be a JSON object")
            rows.append(_raw_row_from_mapping(payload, row_ordinal))
    return rows


def profile_dataset(rows: list[RawRow]) -> DatasetProfile:
    """Compute dataset shape metrics used to motivate strategy selection."""

    candidate_counts = [len(row.candidates) for row in rows]
    candidates = [candidate for row in rows for candidate in row.candidates]
    token_counts = sorted(candidate.tokens for candidate in candidates)
    short_counts = {
        "<=3": sum(1 for value in token_counts if value <= 3),
        "<=5": sum(1 for value in token_counts if value <= 5),
        "<=10": sum(1 for value in token_counts if value <= 10),
        "<=20": sum(1 for value in token_counts if value <= 20),
    }
    passage_type_counts = Counter(candidate.passage_type or "<missing>" for candidate in candidates)

    gold_candidates = [
        row.candidates[gold_idx]
        for row in rows
        for gold_idx in row.long_answer_indices
        if 0 <= gold_idx < len(row.candidates)
    ]
    gold_tokens = sorted(candidate.tokens for candidate in gold_candidates)
    gold_type_counts = Counter(
        candidate.passage_type or "<missing>" for candidate in gold_candidates
    )
    nested_ratio = _nearby_nested_parent_ratio(rows)

    return DatasetProfile(
        sample_rows=len(rows),
        total_candidates=len(candidates),
        avg_candidates_per_row=_round(mean(candidate_counts)) if candidate_counts else 0.0,
        median_candidates_per_row=float(median(candidate_counts)) if candidate_counts else 0.0,
        candidate_token_quantiles=_quantiles(token_counts),
        short_candidate_counts=short_counts,
        short_candidate_ratios={
            key: _round(value / len(candidates)) if candidates else 0.0
            for key, value in short_counts.items()
        },
        passage_type_counts=dict(passage_type_counts.most_common()),
        gold_long_answer_count=len(gold_candidates),
        gold_token_avg=_round(mean(gold_tokens)) if gold_tokens else 0.0,
        gold_token_median=float(median(gold_tokens)) if gold_tokens else 0.0,
        gold_token_quantiles=_quantiles(gold_tokens),
        gold_type_counts=dict(gold_type_counts.most_common()),
        nearby_nested_parent_ratio=nested_ratio,
    )


def build_chunks(rows: list[RawRow], strategy: ChunkingStrategy) -> list[EvalChunk]:
    """Build chunks for a named strategy."""

    builders = {
        "raw_candidate": _build_raw_candidate_chunks,
        "whole_row": _build_whole_row_chunks,
        "dpr_window": _build_dpr_window_chunks,
        "parent_dedup": _build_parent_dedup_chunks,
        "min_context": _build_min_context_chunks,
        "parent_child": _build_parent_child_chunks,
    }
    return builders[strategy](rows)


def profile_strategy(rows: list[RawRow], chunks: list[EvalChunk], strategy: str) -> StrategyProfile:
    """Compute structural metrics for a chunk set."""

    token_counts = [chunk.token_count for chunk in chunks]
    relevant = [chunk for chunk in chunks if is_relevant_chunk(chunk, rows[chunk.row_ordinal])]
    relevant_tokens = [chunk.token_count for chunk in relevant]
    return StrategyProfile(
        strategy=strategy,
        chunks=len(chunks),
        avg_tokens=_round(mean(token_counts)) if token_counts else 0.0,
        median_tokens=float(median(token_counts)) if token_counts else 0.0,
        tiny_chunks_le_10=sum(1 for value in token_counts if value <= 10),
        tiny_chunks_le_20=sum(1 for value in token_counts if value <= 20),
        huge_chunks_gt_300=sum(1 for value in token_counts if value > 300),
        relevant_chunks=len(relevant),
        relevant_token_median=float(median(relevant_tokens)) if relevant_tokens else 0.0,
        duplicate_like_ratio=_duplicate_like_ratio(chunks),
    )


def evaluate_bm25_proxy(
    rows: list[RawRow],
    chunks: list[EvalChunk],
    *,
    top_k: int = 10,
    max_queries: int | None = 100,
    context_budgets: tuple[int, ...] = (512, 1024, 2048),
) -> RetrievalMetrics:
    """Evaluate a chunk strategy with a BM25 proxy over generated chunks."""

    cases = _eval_cases(rows, max_queries=max_queries)
    if not cases or not chunks:
        return RetrievalMetrics(0, 0.0, 0.0, 0.0, 0.0, {}, 0.0)

    corpus_tokens = [_tokenize_for_bm25(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    precisions: list[float] = []
    densities: list[float] = []
    budget_hits = {budget: [] for budget in context_budgets}

    for row in cases:
        scores = bm25.get_scores(_tokenize_for_bm25(row.question or ""))
        top_indices = _top_indices(scores, top_k)
        top_chunks = [chunks[index] for index in top_indices]
        relevant_flags = [is_relevant_chunk(chunk, rows[chunk.row_ordinal]) for chunk in top_chunks]
        recalls.append(1.0 if any(relevant_flags) else 0.0)
        precisions.append(sum(1 for flag in relevant_flags if flag) / float(top_k))
        mrrs.append(_mrr_from_flags(relevant_flags))
        ndcgs.append(_ndcg_from_flags(relevant_flags, ideal_hits=1))
        densities.append(_relevant_token_density(top_chunks, rows))
        for budget in context_budgets:
            budget_hits[budget].append(_budget_contains_relevant(top_chunks, rows, budget))

    return RetrievalMetrics(
        query_count=len(cases),
        recall_at_k=_avg(recalls),
        mrr_at_k=_avg(mrrs),
        ndcg_at_k=_avg(ndcgs),
        precision_at_k=_avg(precisions),
        context_budget_recall={
            str(budget): _avg(hits) for budget, hits in budget_hits.items()
        },
        relevant_token_density_at_k=_avg(densities),
    )


def run_chunking_evaluation(
    raw_path: Path,
    *,
    max_rows: int | None = 1000,
    max_queries: int | None = 100,
    top_k: int = 10,
) -> ChunkingEvaluationReport:
    """Run the full chunking strategy analysis."""

    rows = load_raw_rows(raw_path, max_rows=max_rows)
    dataset_profile = profile_dataset(rows)
    strategies: tuple[ChunkingStrategy, ...] = (
        "raw_candidate",
        "whole_row",
        "dpr_window",
        "parent_dedup",
        "min_context",
        "parent_child",
    )
    results: dict[str, StrategyResult] = {}
    for strategy in strategies:
        chunks = build_chunks(rows, strategy)
        results[strategy] = StrategyResult(
            profile=profile_strategy(rows, chunks, strategy),
            retrieval=evaluate_bm25_proxy(
                rows,
                chunks,
                top_k=top_k,
                max_queries=max_queries,
            ),
        )

    examples = build_example_chunks(rows, strategy="parent_dedup", limit=10)
    return ChunkingEvaluationReport(dataset_profile, results, examples)


def build_example_chunks(
    rows: list[RawRow], *, strategy: ChunkingStrategy = "parent_dedup", limit: int = 10
) -> list[ExampleChunk]:
    """Return representative answer-bearing chunks from a strategy."""

    chunks = build_chunks(rows, strategy)
    out: list[ExampleChunk] = []
    for chunk in chunks:
        row = rows[chunk.row_ordinal]
        if not is_relevant_chunk(chunk, row):
            continue
        out.append(
            ExampleChunk(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                question=chunk.question,
                span=f"{chunk.start_candidate_idx}-{chunk.end_candidate_idx}",
                passage_types=chunk.passage_types,
                tokens=chunk.token_count,
                text=display_text(chunk.text, max_len=420),
            )
        )
        if len(out) >= limit:
            break
    return out


def write_report_json(path: Path, report: ChunkingEvaluationReport) -> None:
    """Write machine-readable report JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_to_jsonable(report), indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def write_report_markdown(path: Path, report: ChunkingEvaluationReport) -> None:
    """Write a human-readable Markdown report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Chunking Strategy Evaluation",
        "",
        "## Executive Summary",
        "",
        "This report compares candidate, row, fixed-window, parent/deduplicated, "
        "minimum-context, and parent-child chunking strategies over the raw NQ artifact. "
        "`long_answers` are used only as evaluation labels, never as production chunking "
        "inputs.",
        "",
        _recommendation_summary(report),
        "",
        "## Foundational Concepts",
        "",
        "- A **retrieval unit** is what gets embedded, sparse-indexed, scored, and returned "
        "from Qdrant.",
        "- A **context unit** is the larger evidence payload optionally sent to the reranker "
        "or generator after a hit.",
        "- Small retrieval units improve precision only when they are self-contained; tiny "
        "HTML child candidates can be useless alone.",
        "- Large retrieval units preserve context but dilute dense/sparse signals and can "
        "exceed generation budgets.",
        "- This dataset needs both ideas: focused retrieval plus parent provenance for "
        "selective expansion.",
        "",
        "## Literature And Practice Basis",
        "",
        "- **Natural Questions** labels long answers as bounded HTML candidate regions, not "
        "whole pages. That makes `long_answers` candidate indices a useful evaluation "
        "signal for chunk granularity.",
        "- **DPR / Wiki DPR** used Wikipedia split into disjoint 100-word passages, giving a "
        "strong open-domain QA baseline scale rather than article-level retrieval.",
        "- **Dense X Retrieval** argues retrieval granularity materially changes dense "
        "retrieval quality; useful units should be minimal, distinct, and self-contained.",
        "- **Lost in the Middle** shows long contexts are not automatically usable: answer "
        "evidence buried in a large retrieved block can still be missed.",
        "- **Microsoft RAG chunking guidance** warns against both under-contextualized tiny "
        "chunks and oversized chunks with irrelevant information.",
        "- **NVIDIA chunking experiments** found no universal winner and recommend comparing "
        "page/section/token strategies on the actual corpus and query type.",
        "",
        "## Dataset Profile",
        "",
        _markdown_dict(asdict(report.dataset_profile)),
        "",
        "## Strategy Results",
        "",
        "The BM25 proxy is not the final dense/hybrid answer, but it is useful for quickly "
        "testing lexical retrieval behavior under a fixed setup. The context-budget column "
        "is critical: it asks whether an answer-bearing chunk appears before the retrieved "
        "context exceeds 1024 tokens.",
        "",
        _strategy_table(report),
        "",
        "## Result Interpretation",
        "",
        _result_interpretation(report),
        "",
        "## Qualitative Review Slices",
        "",
        "- Inspect tiny answer-bearing chunks, because a small number of short fact sentences "
        "are genuinely useful and should not be blindly discarded.",
        "- Inspect nested table/list rows where a large parent candidate contains many small "
        "child candidates; these are prime cases for duplicate suppression.",
        "- Inspect whole-row wins to determine whether they succeed because of useful broad "
        "context or because they trivially contain the answer while exceeding context budget.",
        "- Inspect row-level failures under token budgets; these show why retrieval recall "
        "alone is insufficient.",
        "- Inspect list/table chunks near gold answers and confirm that headers remain "
        "attached when split.",
        "- Inspect cases where `parent_child` fails budget recall because parent expansion is "
        "too broad; use this to design selective expansion rather than always expanding to "
        "the largest parent.",
        "",
        "## Example Answer-Bearing Chunks",
        "",
    ]
    for idx, example in enumerate(report.example_chunks, start=1):
        lines.extend(
            [
                f"### {idx}. `{example.chunk_id}`",
                "",
                f"- `title`: {example.title}",
                f"- `question`: {example.question}",
                f"- `span`: {example.span}",
                f"- `passage_types`: {example.passage_types}",
                f"- `tokens`: {example.tokens}",
                "",
                "```text",
                example.text,
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Recommendation",
            "",
            _final_recommendation(report),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def is_relevant_chunk(chunk: EvalChunk, row: RawRow) -> bool:
    """Return True if a chunk overlaps or contains a gold long-answer candidate."""

    for gold_idx in row.long_answer_indices:
        if chunk.start_candidate_idx <= gold_idx <= chunk.end_candidate_idx:
            return True
        if 0 <= gold_idx < len(row.candidates):
            gold_text = normalize_text(row.candidates[gold_idx].text).casefold()
            chunk_text = normalize_text(chunk.text).casefold()
            if gold_text and (gold_text in chunk_text or chunk_text in gold_text):
                return True
    return False


def normalize_text(text: object) -> str:
    """Collapse whitespace and normalize non-breaking spaces."""

    return re.sub(r"\s+", " ", str(text).replace("\u00a0", " ")).strip()


def display_text(text: str, *, max_len: int = 220) -> str:
    """Compact text for reports."""

    normalized = normalize_text(text)
    return normalized if len(normalized) <= max_len else normalized[:max_len] + "..."


def token_count(text: object) -> int:
    """Approximate word/token count."""

    return len(re.findall(r"\w+", str(text)))


def _raw_row_from_mapping(payload: Mapping[str, object], row_ordinal: int) -> RawRow:
    raw_candidates = payload.get("candidates")
    raw_types = payload.get("passage_types")
    candidates: list[Candidate] = []
    if isinstance(raw_candidates, list):
        type_list = raw_types if isinstance(raw_types, list) else []
        for idx, raw_candidate in enumerate(raw_candidates):
            text = str(raw_candidate).strip()
            if not text:
                continue
            passage_type = str(type_list[idx]).strip() if idx < len(type_list) else None
            candidates.append(Candidate(idx, text, passage_type or None, token_count(text)))
    return RawRow(
        row_ordinal=row_ordinal,
        title=_optional_str(payload.get("title")),
        question=_optional_str(payload.get("question")),
        document_url=_optional_str(payload.get("document_url")),
        long_answer_indices=_long_answer_indices(payload.get("long_answers")),
        candidates=candidates,
    )


def _build_raw_candidate_chunks(rows: list[RawRow]) -> list[EvalChunk]:
    return [
        _chunk(
            "raw_candidate",
            row,
            cand.index,
            cand.index,
            cand.text,
            [cand.passage_type],
            "candidate",
        )
        for row in rows
        for cand in row.candidates
    ]


def _build_whole_row_chunks(rows: list[RawRow]) -> list[EvalChunk]:
    chunks: list[EvalChunk] = []
    for row in rows:
        if not row.candidates:
            continue
        chunks.append(
            _chunk(
                "whole_row",
                row,
                row.candidates[0].index,
                row.candidates[-1].index,
                "\n".join(candidate.text for candidate in row.candidates),
                ["row"],
                "row",
            )
        )
    return chunks


def _build_dpr_window_chunks(rows: list[RawRow], *, target_tokens: int = 100) -> list[EvalChunk]:
    chunks: list[EvalChunk] = []
    for row in rows:
        current: list[Candidate] = []
        current_tokens = 0
        for candidate in row.candidates:
            if current and current_tokens + candidate.tokens > target_tokens:
                chunks.append(_chunk_from_candidates("dpr_window", row, current, "fixed_window"))
                current = []
                current_tokens = 0
            current.append(candidate)
            current_tokens += candidate.tokens
        if current:
            chunks.append(_chunk_from_candidates("dpr_window", row, current, "fixed_window"))
    return chunks


def _build_parent_dedup_chunks(rows: list[RawRow]) -> list[EvalChunk]:
    chunks: list[EvalChunk] = []
    for row in rows:
        retained_texts: list[tuple[int, str]] = []
        for candidate in row.candidates:
            normalized = normalize_text(candidate.text).casefold()
            if _is_boilerplate(candidate.text):
                continue
            if candidate.tokens <= 20 and any(normalized in parent for _, parent in retained_texts):
                continue
            chunks.append(
                _chunk(
                    "parent_dedup",
                    row,
                    candidate.index,
                    candidate.index,
                    candidate.text,
                    [candidate.passage_type],
                    "parent_or_standalone",
                )
            )
            if candidate.tokens >= 40:
                retained_texts.append((candidate.index, normalized))
    return chunks


def _build_min_context_chunks(
    rows: list[RawRow], *, min_tokens: int = 60, max_tokens: int = 220
) -> list[EvalChunk]:
    chunks: list[EvalChunk] = []
    for row in rows:
        i = 0
        while i < len(row.candidates):
            candidate = row.candidates[i]
            if _is_boilerplate(candidate.text):
                i += 1
                continue
            span = [candidate]
            total = candidate.tokens
            while total < min_tokens and i + len(span) < len(row.candidates):
                next_candidate = row.candidates[i + len(span)]
                if _is_boilerplate(next_candidate.text):
                    span.append(next_candidate)
                    continue
                next_total = total + next_candidate.tokens
                if next_total > max_tokens or not _compatible(span[-1], next_candidate):
                    break
                span.append(next_candidate)
                total = next_total
            chunks.append(_chunk_from_candidates("min_context", row, span, "minimum_context_span"))
            i += len(span)
    return chunks


def _build_parent_child_chunks(rows: list[RawRow]) -> list[EvalChunk]:
    chunks = _build_min_context_chunks(rows, min_tokens=40, max_tokens=180)
    parent_by_row = {
        row.row_ordinal: max(row.candidates, key=lambda candidate: candidate.tokens, default=None)
        for row in rows
    }
    for chunk in chunks:
        parent = parent_by_row.get(chunk.row_ordinal)
        if parent is None:
            continue
        chunk.context_text = parent.text
        chunk.parent_candidate_idx = parent.index
        chunk.chunk_kind = "child_with_parent_context"
    return chunks


def _chunk_from_candidates(
    strategy: str, row: RawRow, candidates: list[Candidate], chunk_kind: str
) -> EvalChunk:
    return _chunk(
        strategy,
        row,
        candidates[0].index,
        candidates[-1].index,
        "\n".join(
            candidate.text for candidate in candidates if not _is_boilerplate(candidate.text)
        ),
        [candidate.passage_type for candidate in candidates],
        chunk_kind,
    )


def _chunk(
    strategy: str,
    row: RawRow,
    start: int,
    end: int,
    text: str,
    passage_types: list[str | None],
    chunk_kind: str,
) -> EvalChunk:
    key = f"{strategy}\x1f{row.row_ordinal}\x1f{start}\x1f{end}\x1f{normalize_text(text)[:120]}"
    return EvalChunk(
        chunk_id=str(uuid.uuid5(CHUNK_ID_NAMESPACE, key)),
        strategy=strategy,
        row_ordinal=row.row_ordinal,
        start_candidate_idx=start,
        end_candidate_idx=end,
        text=text,
        passage_types=passage_types,
        title=row.title,
        question=row.question,
        chunk_kind=chunk_kind,
    )


def _compatible(left: Candidate, right: Candidate) -> bool:
    if left.passage_type == right.passage_type:
        return True
    if {left.passage_type, right.passage_type}.issubset(LIST_OR_TABLE_TYPES):
        return True
    if _looks_like_intro(left.text) or left.passage_type in HEADER_TYPES:
        return True
    return right.tokens <= 12 and left.tokens <= 30


def _is_boilerplate(text: str) -> bool:
    normalized = normalize_text(text).casefold()
    return normalized in BOILERPLATE_EXACT or len(normalized) <= 2


def _looks_like_intro(text: str) -> bool:
    normalized = normalize_text(text).casefold()
    return normalized.endswith(":") or (
        len(normalized.split()) <= 40 and any(marker in normalized for marker in INTRO_MARKERS)
    )


def _eval_cases(rows: list[RawRow], *, max_queries: int | None) -> list[RawRow]:
    cases = [row for row in rows if row.question and row.long_answer_indices]
    return cases if max_queries is None else cases[:max_queries]


def _tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _top_indices(scores: Iterable[float], k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:k]


def _mrr_from_flags(flags: list[bool]) -> float:
    for rank, flag in enumerate(flags, start=1):
        if flag:
            return 1.0 / rank
    return 0.0


def _ndcg_from_flags(flags: list[bool], *, ideal_hits: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, flag in enumerate(flags, start=1) if flag)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(flags), ideal_hits) + 1))
    return dcg / idcg if idcg else 0.0


def _budget_contains_relevant(
    top_chunks: list[EvalChunk], rows: list[RawRow], budget: int
) -> float:
    used = 0
    for chunk in top_chunks:
        used += chunk.context_token_count
        if used > budget:
            return 0.0
        if is_relevant_chunk(chunk, rows[chunk.row_ordinal]):
            return 1.0
    return 0.0


def _relevant_token_density(top_chunks: list[EvalChunk], rows: list[RawRow]) -> float:
    total = sum(chunk.context_token_count for chunk in top_chunks)
    if total == 0:
        return 0.0
    relevant = sum(
        chunk.context_token_count
        for chunk in top_chunks
        if is_relevant_chunk(chunk, rows[chunk.row_ordinal])
    )
    return relevant / total


def _duplicate_like_ratio(chunks: list[EvalChunk]) -> float:
    by_row: dict[int, list[str]] = {}
    duplicate_like = 0
    for chunk in chunks:
        normalized = normalize_text(chunk.text).casefold()
        row_texts = by_row.setdefault(chunk.row_ordinal, [])
        is_duplicate_like = any(
            normalized in other or other in normalized for other in row_texts
        )
        if len(normalized) > 5 and is_duplicate_like:
            duplicate_like += 1
        row_texts.append(normalized)
    return _round(duplicate_like / len(chunks)) if chunks else 0.0


def _nearby_nested_parent_ratio(rows: list[RawRow]) -> float:
    large = 0
    nested = 0
    for row in rows:
        for position, candidate in enumerate(row.candidates):
            if candidate.tokens <= 80:
                continue
            large += 1
            parent_text = normalize_text(candidate.text).casefold()
            for child in row.candidates[position + 1 : position + 25]:
                child_text = normalize_text(child.text).casefold()
                if child.tokens <= 20 and len(child_text) > 5 and child_text in parent_text:
                    nested += 1
                    break
    return _round(nested / large) if large else 0.0


def _quantiles(sorted_values: list[int]) -> dict[str, int]:
    if not sorted_values:
        return {}
    return {
        "p10": sorted_values[int(len(sorted_values) * 0.10)],
        "p25": sorted_values[int(len(sorted_values) * 0.25)],
        "p50": sorted_values[int(len(sorted_values) * 0.50)],
        "p75": sorted_values[int(len(sorted_values) * 0.75)],
        "p90": sorted_values[int(len(sorted_values) * 0.90)],
        "p95": sorted_values[int(len(sorted_values) * 0.95)],
    }


def _long_answer_indices(raw: object) -> list[int]:
    if not isinstance(raw, list):
        return []
    return [int(value) for value in raw if isinstance(value, int) and value >= 0]


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _avg(values: list[float]) -> float:
    return _round(sum(values) / len(values)) if values else 0.0


def _round(value: float) -> float:
    return round(float(value), 4)


def _to_jsonable(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _markdown_dict(data: dict[str, object]) -> str:
    return "\n".join(f"- `{key}`: {value}" for key, value in data.items())


def _strategy_table(report: ChunkingEvaluationReport) -> str:
    lines = [
        "| Strategy | Chunks | Median Tokens | Tiny <=10 | Duplicate Ratio | "
        "Recall@10 | MRR@10 | Budget Recall 1024 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, result in report.strategy_results.items():
        budget_1024 = result.retrieval.context_budget_recall.get("1024", 0.0)
        lines.append(
            f"| `{strategy}` | {result.profile.chunks} | {result.profile.median_tokens} | "
            f"{result.profile.tiny_chunks_le_10} | {result.profile.duplicate_like_ratio} | "
            f"{result.retrieval.recall_at_k} | {result.retrieval.mrr_at_k} | {budget_1024} |"
        )
    return "\n".join(lines)


def _recommendation_summary(report: ChunkingEvaluationReport) -> str:
    whole_row = report.strategy_results["whole_row"]
    min_context = report.strategy_results["min_context"]
    return (
        "The source data confirms the concern that raw candidates are often too small: "
        f"{report.dataset_profile.short_candidate_ratios['<=10']:.1%} of sampled candidates "
        "are 10 tokens or fewer. However, whole-row retrieval is also not sufficient: it "
        f"achieves BM25 proxy Recall@10 of {whole_row.retrieval.recall_at_k}, but its "
        "1024-token budget recall is only "
        f"{whole_row.retrieval.context_budget_recall.get('1024', 0.0)} because rows are "
        f"large (median {whole_row.profile.median_tokens} tokens). The best practical "
        "direction is a hierarchical structure-aware chunker. In this proxy run, "
        f"`min_context` keeps a healthier median size ({min_context.profile.median_tokens} "
        "tokens), nearly matches raw-candidate recall, and dramatically reduces tiny chunks."
    )


def _result_interpretation(report: ChunkingEvaluationReport) -> str:
    raw = report.strategy_results["raw_candidate"]
    row = report.strategy_results["whole_row"]
    dpr = report.strategy_results["dpr_window"]
    min_context = report.strategy_results["min_context"]
    parent_child = report.strategy_results["parent_child"]
    return "\n".join(
        [
            f"- `raw_candidate` has Recall@10 `{raw.retrieval.recall_at_k}`, but "
            f"`{raw.profile.tiny_chunks_le_10}` chunks are 10 tokens or fewer. This supports "
            "the concern that many raw candidates are weak standalone retrieval units.",
            f"- `whole_row` has the best proxy Recall@10 `{row.retrieval.recall_at_k}` and "
            f"MRR@10 `{row.retrieval.mrr_at_k}`, but its median chunk is "
            f"`{row.profile.median_tokens}` tokens and 1024-token budget recall is only "
            f"`{row.retrieval.context_budget_recall.get('1024', 0.0)}`. It wins by "
            "containing everything, not by being a good context unit.",
            f"- `dpr_window` is a useful academic baseline: median `{dpr.profile.median_tokens}` "
            f"tokens and Recall@10 `{dpr.retrieval.recall_at_k}`. It is competitive, but it "
            "does not know about table/list/header boundaries.",
            f"- `min_context` is the strongest simple production-shaped candidate: median "
            f"`{min_context.profile.median_tokens}` tokens, only "
            f"`{min_context.profile.tiny_chunks_le_10}` tiny chunks, and budget recall "
            f"`{min_context.retrieval.context_budget_recall.get('1024', 0.0)}`.",
            f"- `parent_child` shows that parent expansion must be selective. Its retrieval "
            f"unit is reasonable, but expanding to broad parents drops 1024-token budget "
            f"recall to `{parent_child.retrieval.context_budget_recall.get('1024', 0.0)}`.",
        ]
    )


def _final_recommendation(report: ChunkingEvaluationReport) -> str:
    min_context = report.strategy_results["min_context"]
    dpr = report.strategy_results["dpr_window"]
    return (
        "Do not ship raw candidate indexing or whole-row indexing as the final strategy. "
        "Use `min_context` as the next implementation baseline, then fold in the best parts "
        "of `parent_dedup`: suppress nested duplicate children, merge tiny adjacent siblings "
        "into coherent list/table/prose units, keep standalone paragraph/list/table parents, "
        "and split oversized candidates by sentence or list boundaries while avoiding tiny "
        "tail chunks. Keep `group_id`, candidate span, and parent metadata so retrieval can "
        "stay focused while reranking/generation can selectively expand context. Before "
        "locking this in, run the same comparison with dense and hybrid Qdrant indexes; the "
        f"lexical proxy currently shows `min_context` (Recall@10 "
        f"{min_context.retrieval.recall_at_k}) and `dpr_window` (Recall@10 "
        f"{dpr.retrieval.recall_at_k}) as the strongest budget-aware contenders."
    )
