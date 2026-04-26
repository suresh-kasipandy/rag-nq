"""Milestone 6: offline retrieval evaluation utilities."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.config.settings import Settings
from src.ingestion.chunk_store import IndexRecord, iter_index_records_jsonl
from src.ingestion.models import IndexChunk, Passage, SparseIndexManifest
from src.observability.logging_setup import get_stage_logger
from src.observability.progress import ProgressTicker
from src.retrieval.qdrant_retrievers import Mode, QdrantModeRetriever

LOGGER = get_stage_logger(__name__)
SUPPORTED_EVAL_MODES: tuple[Mode, Mode, Mode] = ("dense", "sparse", "hybrid")
RelevanceContract = Literal[
    "answer_overlap",
    "point_id",
    "group_id",
    "source_row_ordinal",
    "candidate_span_overlap",
]


class EvalCase(BaseModel):
    """One retrieval evaluation query with chunk-aware relevance labels."""

    query: str = Field(min_length=1)
    relevant_passage_ids: list[str] = Field(default_factory=list)
    relevant_group_ids: list[str] = Field(default_factory=list)
    relevant_source_row_ordinals: list[int] = Field(default_factory=list)
    relevant_candidate_spans: list[CandidateSpan] = Field(default_factory=list)
    answer_texts: list[str] = Field(default_factory=list)


class CandidateSpan(BaseModel):
    """Source row candidate span used for chunk provenance relevance."""

    source_row_ordinal: int = Field(ge=0)
    start_candidate_idx: int = Field(ge=0)
    end_candidate_idx: int = Field(ge=0)


class SparseEvalIdentifiers(BaseModel):
    """Sparse index identifiers needed to reproduce sparse/hybrid eval runs."""

    sparse_pass1_path: str
    sparse_manifest_path: str
    sparse_manifest_schema_version: str
    sparse_manifest_created_at_utc: str
    sparse_manifest_vocabulary_size: int
    sparse_manifest_document_count: int
    sparse_manifest_silver_path_resolved: str


class EvalRunConfig(BaseModel):
    """Runtime configuration snapshot for one retrieval-eval run."""

    top_k: int = Field(ge=1)
    k_values: list[int] = Field(min_length=1)
    modes: list[Mode] = Field(min_length=1)
    relevance_contract: RelevanceContract
    max_queries: int | None = Field(default=None, ge=1)
    query_count: int = Field(ge=0)
    corpus_path: str
    passages_path: str | None = None
    qdrant_url: str
    qdrant_collection: str
    qdrant_vector_name: str
    qdrant_sparse_vector_name: str
    sparse_identifiers: SparseEvalIdentifiers | None = None


class ModeMetricSummary(BaseModel):
    """Per-mode metric summary (averaged over all eval queries)."""

    query_count: int = Field(ge=0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr_at_k: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: float = Field(ge=0.0, le=1.0)
    metrics_by_k: dict[str, MetricAtK] = Field(default_factory=dict)


class MetricAtK(BaseModel):
    """Per-mode metric summary for one cutoff k."""

    query_count: int = Field(ge=0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr_at_k: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: float = Field(ge=0.0, le=1.0)


class RetrievalEvalReport(BaseModel):
    """Final retrieval-evaluation artifact schema."""

    run_config: EvalRunConfig
    modes: dict[Mode, ModeMetricSummary]


def compute_recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], *, k: int) -> float:
    """Compute Recall@k for one query."""

    _ensure_positive_k(k)
    if not relevant_ids:
        return 0.0
    top_ids = set(retrieved_ids[:k])
    return len(top_ids.intersection(relevant_ids)) / float(len(relevant_ids))


def compute_mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], *, k: int) -> float:
    """Compute MRR@k for one query."""

    _ensure_positive_k(k)
    if not relevant_ids:
        return 0.0
    for rank, passage_id in enumerate(retrieved_ids[:k], start=1):
        if passage_id in relevant_ids:
            return 1.0 / float(rank)
    return 0.0


def compute_ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], *, k: int) -> float:
    """Compute binary NDCG@k for one query."""

    _ensure_positive_k(k)
    if not relevant_ids:
        return 0.0
    dcg = 0.0
    for rank, passage_id in enumerate(retrieved_ids[:k], start=1):
        if passage_id in relevant_ids:
            dcg += 1.0 / math.log2(float(rank + 1))
    ideal_hits = min(k, len(relevant_ids))
    if ideal_hits <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(float(rank + 1)) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def run_retrieval_evaluation(
    settings: Settings,
    *,
    top_k: int | None = None,
    k_values: list[int] | None = None,
    modes: list[Mode] | None = None,
    relevance_contract: RelevanceContract = "answer_overlap",
    max_queries: int | None,
    output_path: Path,
    corpus_path: Path | None = None,
    csv_output_path: Path | None = None,
) -> RetrievalEvalReport:
    """Run offline retrieval evaluation and persist machine-readable artifacts."""

    resolved_k_values = _resolve_k_values(top_k=top_k, k_values=k_values)
    resolved_modes = _resolve_modes(modes)
    resolved_corpus_path = corpus_path or settings.index_chunks_path
    if corpus_path is not None and not corpus_path.is_file():
        raise FileNotFoundError(f"Evaluation corpus does not exist: {corpus_path}")
    if (
        corpus_path is None
        and not resolved_corpus_path.is_file()
        and settings.passages_path.is_file()
    ):
        resolved_corpus_path = settings.passages_path
    eval_cases = build_eval_cases_from_index_artifact(
        resolved_corpus_path,
        max_queries=max_queries,
        progress_every_records=settings.progress_log_every_records,
        progress_every_seconds=settings.progress_log_every_seconds,
    )
    LOGGER.info(
        "retrieval eval start modes=%s k_values=%s relevance_contract=%s query_count=%s",
        ",".join(resolved_modes),
        ",".join(str(k) for k in resolved_k_values),
        relevance_contract,
        len(eval_cases),
        extra={"stage": "retrieval_eval"},
    )
    mode_summaries = {
        mode: evaluate_mode(
            settings,
            mode=mode,
            cases=eval_cases,
            k_values=resolved_k_values,
            relevance_contract=relevance_contract,
        )
        for mode in resolved_modes
    }

    report = RetrievalEvalReport(
        run_config=EvalRunConfig(
            top_k=max(resolved_k_values),
            k_values=resolved_k_values,
            modes=resolved_modes,
            relevance_contract=relevance_contract,
            max_queries=max_queries,
            query_count=len(eval_cases),
            corpus_path=str(resolved_corpus_path.resolve()),
            passages_path=(
                str(settings.passages_path.resolve()) if settings.passages_path.is_file() else None
            ),
            qdrant_url=settings.qdrant_url,
            qdrant_collection=settings.qdrant_collection,
            qdrant_vector_name=settings.qdrant_vector_name,
            qdrant_sparse_vector_name=settings.qdrant_sparse_vector_name,
            sparse_identifiers=load_sparse_eval_identifiers(settings),
        ),
        modes=mode_summaries,
    )
    write_eval_report(output_path, report)
    if csv_output_path is not None:
        write_eval_csv(csv_output_path, report)
    return report


def build_eval_cases_from_index_artifact(
    index_path: Path,
    *,
    max_queries: int | None,
    progress_every_records: int | None = None,
    progress_every_seconds: float = 60.0,
) -> list[EvalCase]:
    """Build deterministic eval cases from chunked index artifacts."""

    if not index_path.is_file():
        return build_eval_cases_from_passages_jsonl(
            index_path,
            max_queries=max_queries,
            progress_every_records=progress_every_records,
            progress_every_seconds=progress_every_seconds,
        )
    ticker = _build_case_progress_ticker(
        progress_every_records=progress_every_records,
        progress_every_seconds=progress_every_seconds,
    )
    if ticker is not None:
        ticker.start(corpus=index_path, max_queries=max_queries)
    relevant_by_question: dict[str, _RelevantLabels] = {}
    records_scanned = 0
    for records_scanned, record in enumerate(iter_index_records_jsonl(index_path), start=1):
        question = (record.question or "").strip()
        if not question or not _is_answer_relevant_record(record):
            if ticker is not None:
                ticker.tick(records_scanned, relevant_questions=len(relevant_by_question))
            continue
        labels = relevant_by_question.setdefault(question, _RelevantLabels())
        labels.point_ids.add(_record_point_id(record))
        for answer in record.long_answers:
            if answer.strip():
                labels.answer_texts.add(answer.strip())
        group_id = getattr(record, "group_id", None)
        if isinstance(group_id, str) and group_id:
            labels.group_ids.add(group_id)
        source_row_ordinal = getattr(record, "source_row_ordinal", None)
        if isinstance(source_row_ordinal, int):
            labels.source_row_ordinals.add(source_row_ordinal)
            start = getattr(record, "start_candidate_idx", None)
            end = getattr(record, "end_candidate_idx", None)
            if isinstance(start, int) and isinstance(end, int):
                labels.candidate_spans.add((source_row_ordinal, start, end))
        if ticker is not None:
            ticker.tick(records_scanned, relevant_questions=len(relevant_by_question))

    cases = _eval_cases_from_relevance_map(relevant_by_question, max_queries=max_queries)
    if ticker is not None:
        ticker.finish(
            records_scanned,
            relevant_questions=len(relevant_by_question),
            query_count=len(cases),
        )
    return cases


def build_eval_cases_from_passages_jsonl(
    passages_path: Path,
    *,
    max_queries: int | None,
    progress_every_records: int | None = None,
    progress_every_seconds: float = 60.0,
) -> list[EvalCase]:
    """Build deterministic eval cases from silver passages + long-answer overlap."""

    ticker = _build_case_progress_ticker(
        progress_every_records=progress_every_records,
        progress_every_seconds=progress_every_seconds,
    )
    if ticker is not None:
        ticker.start(corpus=passages_path, max_queries=max_queries)
    relevant_by_question: dict[str, _RelevantLabels] = {}

    records_scanned = 0
    with passages_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            records_scanned += 1
            passage = Passage.model_validate_json(raw_line)
            question = (passage.question or "").strip()
            if not question:
                if ticker is not None:
                    ticker.tick(records_scanned, relevant_questions=len(relevant_by_question))
                continue
            if not _is_relevant_passage(passage):
                if ticker is not None:
                    ticker.tick(records_scanned, relevant_questions=len(relevant_by_question))
                continue
            relevant = relevant_by_question.setdefault(question, _RelevantLabels())
            relevant.point_ids.add(passage.passage_id)
            for answer in passage.long_answers:
                if answer.strip():
                    relevant.answer_texts.add(answer.strip())
            if ticker is not None:
                ticker.tick(records_scanned, relevant_questions=len(relevant_by_question))

    cases = _eval_cases_from_relevance_map(relevant_by_question, max_queries=max_queries)
    if ticker is not None:
        ticker.finish(
            records_scanned,
            relevant_questions=len(relevant_by_question),
            query_count=len(cases),
        )
    return cases


def _build_case_progress_ticker(
    *, progress_every_records: int | None, progress_every_seconds: float
) -> ProgressTicker | None:
    if progress_every_records is None:
        return None
    return ProgressTicker(
        logger=LOGGER,
        stage="retrieval_eval_cases",
        label="records",
        every_items=progress_every_records,
        every_seconds=progress_every_seconds,
    )


def evaluate_mode(
    settings: Settings,
    *,
    mode: Mode,
    cases: list[EvalCase],
    k_values: list[int],
    relevance_contract: RelevanceContract,
) -> ModeMetricSummary:
    """Evaluate one retrieval mode over provided eval cases."""

    resolved_k_values = _resolve_k_values(k_values=k_values)
    top_k = max(resolved_k_values)
    if not cases:
        empty = {
            str(k): MetricAtK(query_count=0, recall_at_k=0.0, mrr_at_k=0.0, ndcg_at_k=0.0)
            for k in resolved_k_values
        }
        return ModeMetricSummary(
            query_count=0,
            recall_at_k=0.0,
            mrr_at_k=0.0,
            ndcg_at_k=0.0,
            metrics_by_k=empty,
        )

    retriever = QdrantModeRetriever(settings=settings, mode=mode)
    ticker = ProgressTicker(
        logger=LOGGER,
        stage="retrieval_eval_mode",
        label="queries",
        total=len(cases),
        every_items=settings.progress_log_every_records,
        every_seconds=settings.progress_log_every_seconds,
    )
    ticker.start(
        mode=mode,
        top_k=top_k,
        k_values=",".join(str(k) for k in resolved_k_values),
        relevance_contract=relevance_contract,
    )
    recalls_by_k: dict[int, list[float]] = {k: [] for k in resolved_k_values}
    mrrs_by_k: dict[int, list[float]] = {k: [] for k in resolved_k_values}
    ndcgs_by_k: dict[int, list[float]] = {k: [] for k in resolved_k_values}

    for query_index, case in enumerate(cases, start=1):
        hits = retriever.retrieve(case.query, top_k=top_k)
        retrieved_labels, relevant_labels = _metric_labels_for_hits(
            hits, case=case, relevance_contract=relevance_contract
        )
        for k in resolved_k_values:
            recalls_by_k[k].append(compute_recall_at_k(retrieved_labels, relevant_labels, k=k))
            mrrs_by_k[k].append(compute_mrr_at_k(retrieved_labels, relevant_labels, k=k))
            ndcgs_by_k[k].append(compute_ndcg_at_k(retrieved_labels, relevant_labels, k=k))
        ticker.tick(query_index, mode=mode)

    metrics_by_k = {
        str(k): MetricAtK(
            query_count=len(cases),
            recall_at_k=sum(recalls_by_k[k]) / float(len(recalls_by_k[k])),
            mrr_at_k=sum(mrrs_by_k[k]) / float(len(mrrs_by_k[k])),
            ndcg_at_k=sum(ndcgs_by_k[k]) / float(len(ndcgs_by_k[k])),
        )
        for k in resolved_k_values
    }
    primary = metrics_by_k[str(max(resolved_k_values))]
    ticker.finish(
        len(cases),
        mode=mode,
        recall_at_k=f"{primary.recall_at_k:.4f}",
        mrr_at_k=f"{primary.mrr_at_k:.4f}",
        ndcg_at_k=f"{primary.ndcg_at_k:.4f}",
    )
    return ModeMetricSummary(
        query_count=len(cases),
        recall_at_k=primary.recall_at_k,
        mrr_at_k=primary.mrr_at_k,
        ndcg_at_k=primary.ndcg_at_k,
        metrics_by_k=metrics_by_k,
    )


def load_sparse_eval_identifiers(settings: Settings) -> SparseEvalIdentifiers | None:
    """Load sparse manifest metadata used for reproducing sparse/hybrid eval runs."""

    if not settings.sparse_manifest_path.is_file():
        return None
    with settings.sparse_manifest_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    manifest = SparseIndexManifest.model_validate(payload)
    return SparseEvalIdentifiers(
        sparse_pass1_path=str(settings.sparse_pass1_path.resolve()),
        sparse_manifest_path=str(settings.sparse_manifest_path.resolve()),
        sparse_manifest_schema_version=manifest.schema_version,
        sparse_manifest_created_at_utc=manifest.created_at_utc,
        sparse_manifest_vocabulary_size=manifest.vocabulary_size,
        sparse_manifest_document_count=manifest.document_count,
        sparse_manifest_silver_path_resolved=manifest.silver_path_resolved,
    )


def write_eval_report(path: Path, report: RetrievalEvalReport) -> None:
    """Persist report JSON in a deterministic, machine-readable format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.model_dump(mode="json"), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_eval_csv(path: Path, report: RetrievalEvalReport) -> None:
    """Persist a flat CSV comparison artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "k",
                "query_count",
                "recall_at_k",
                "mrr_at_k",
                "ndcg_at_k",
                "relevance_contract",
                "qdrant_collection",
            ],
        )
        writer.writeheader()
        for mode, summary in report.modes.items():
            for k, metrics in summary.metrics_by_k.items():
                writer.writerow(
                    {
                        "mode": mode,
                        "k": k,
                        "query_count": metrics.query_count,
                        "recall_at_k": metrics.recall_at_k,
                        "mrr_at_k": metrics.mrr_at_k,
                        "ndcg_at_k": metrics.ndcg_at_k,
                        "relevance_contract": report.run_config.relevance_contract,
                        "qdrant_collection": report.run_config.qdrant_collection,
                    }
                )


def _is_relevant_passage(passage: Passage) -> bool:
    """Heuristic relevance label: any long answer appears in passage text."""

    text = passage.text.casefold()
    for answer in passage.long_answers:
        normalized = answer.strip().casefold()
        if not normalized:
            continue
        if normalized in text:
            return True
    return False


def _is_answer_relevant_record(record: IndexRecord) -> bool:
    text = _record_text_for_answer_overlap(record).casefold()
    for answer in record.long_answers:
        normalized = answer.strip().casefold()
        if normalized and normalized in text:
            return True
    return False


def _record_text_for_answer_overlap(record: IndexRecord) -> str:
    context_text = getattr(record, "context_text", None)
    if isinstance(context_text, str) and context_text.strip():
        return f"{record.text}\n{context_text}"
    return record.text


def _record_point_id(record: IndexRecord) -> str:
    if isinstance(record, IndexChunk):
        return record.chunk_id
    return record.passage_id


def _eval_cases_from_relevance_map(
    relevant_by_question: dict[str, _RelevantLabels], *, max_queries: int | None
) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for question in sorted(relevant_by_question):
        labels = relevant_by_question[question]
        cases.append(
            EvalCase(
                query=question,
                relevant_passage_ids=sorted(labels.point_ids),
                relevant_group_ids=sorted(labels.group_ids),
                relevant_source_row_ordinals=sorted(labels.source_row_ordinals),
                relevant_candidate_spans=[
                    CandidateSpan(
                        source_row_ordinal=row,
                        start_candidate_idx=start,
                        end_candidate_idx=end,
                    )
                    for row, start, end in sorted(labels.candidate_spans)
                ],
                answer_texts=sorted(labels.answer_texts),
            )
        )
        if max_queries is not None and len(cases) >= max_queries:
            break
    return cases


def _metric_labels_for_hits(
    hits: list[Any], *, case: EvalCase, relevance_contract: RelevanceContract
) -> tuple[list[str], set[str]]:
    relevant_labels = _relevant_labels_for_case(case, relevance_contract=relevance_contract)
    seen_relevant_labels: set[str] = set()
    retrieved_labels: list[str] = []
    for rank, hit in enumerate(hits, start=1):
        label = _label_for_hit(hit, case=case, relevance_contract=relevance_contract)
        if label is None or label in seen_relevant_labels:
            retrieved_labels.append(f"__non_relevant_or_duplicate__:{rank}")
            continue
        seen_relevant_labels.add(label)
        retrieved_labels.append(label)
    return retrieved_labels, relevant_labels


def _relevant_labels_for_case(
    case: EvalCase, *, relevance_contract: RelevanceContract
) -> set[str]:
    if relevance_contract in {"answer_overlap", "point_id"}:
        if relevance_contract == "answer_overlap":
            return {_answer_label(answer) for answer in case.answer_texts}
        return set(case.relevant_passage_ids)
    if relevance_contract == "group_id":
        return set(case.relevant_group_ids)
    if relevance_contract == "source_row_ordinal":
        return {str(value) for value in case.relevant_source_row_ordinals}
    if relevance_contract == "candidate_span_overlap":
        return {_span_label(span) for span in case.relevant_candidate_spans}
    raise ValueError(f"Unsupported relevance contract {relevance_contract!r}.")


def _label_for_hit(
    hit: Any, *, case: EvalCase, relevance_contract: RelevanceContract
) -> str | None:
    if relevance_contract in {"answer_overlap", "point_id"}:
        if relevance_contract == "answer_overlap":
            text = _hit_text_for_answer_overlap(hit).casefold()
            for answer in case.answer_texts:
                if _answer_label(answer) in text:
                    return _answer_label(answer)
            return None
        return hit.point_id if hit.point_id in case.relevant_passage_ids else None
    if relevance_contract == "group_id":
        return hit.group_id if hit.group_id in case.relevant_group_ids else None
    if relevance_contract == "source_row_ordinal":
        if hit.source_row_ordinal in case.relevant_source_row_ordinals:
            return str(hit.source_row_ordinal)
        return None
    if relevance_contract == "candidate_span_overlap":
        if hit.source_row_ordinal is None:
            return None
        if hit.start_candidate_idx is None or hit.end_candidate_idx is None:
            return None
        for span in case.relevant_candidate_spans:
            if _spans_overlap(
                hit.source_row_ordinal,
                hit.start_candidate_idx,
                hit.end_candidate_idx,
                span,
            ):
                return _span_label(span)
        return None
    raise ValueError(f"Unsupported relevance contract {relevance_contract!r}.")


def _span_label(span: CandidateSpan) -> str:
    return f"{span.source_row_ordinal}:{span.start_candidate_idx}:{span.end_candidate_idx}"


def _hit_text_for_answer_overlap(hit: Any) -> str:
    text = hit.text
    context_text = hit.context_text
    if isinstance(context_text, str) and context_text.strip():
        return f"{text}\n{context_text}"
    return text


def _answer_label(answer: str) -> str:
    return answer.strip().casefold()


def _spans_overlap(
    source_row_ordinal: int, start_candidate_idx: int, end_candidate_idx: int, span: CandidateSpan
) -> bool:
    if source_row_ordinal != span.source_row_ordinal:
        return False
    return (
        start_candidate_idx <= span.end_candidate_idx
        and end_candidate_idx >= span.start_candidate_idx
    )


class _RelevantLabels:
    def __init__(self) -> None:
        self.point_ids: set[str] = set()
        self.group_ids: set[str] = set()
        self.source_row_ordinals: set[int] = set()
        self.candidate_spans: set[tuple[int, int, int]] = set()
        self.answer_texts: set[str] = set()


def _resolve_k_values(
    *, top_k: int | None = None, k_values: list[int] | None = None
) -> list[int]:
    resolved = k_values if k_values is not None else [top_k if top_k is not None else 10]
    if not resolved:
        raise ValueError("At least one k value is required.")
    for k in resolved:
        _ensure_positive_k(k)
    return sorted(set(resolved))


def _resolve_modes(modes: list[Mode] | None) -> list[Mode]:
    resolved = modes or list(SUPPORTED_EVAL_MODES)
    if not resolved:
        raise ValueError("At least one retrieval mode is required.")
    invalid = [mode for mode in resolved if mode not in SUPPORTED_EVAL_MODES]
    if invalid:
        raise ValueError(f"Unsupported retrieval modes: {invalid}.")
    return list(dict.fromkeys(resolved))


def _ensure_positive_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}.")
