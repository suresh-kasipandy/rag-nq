"""Milestone 3.5: lightweight retrieval-only evaluation utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.config.settings import Settings
from src.ingestion.models import Passage, SparseIndexManifest
from src.ingestion.passage_store import PassageStore
from src.retrieval.qdrant_retrievers import Mode, QdrantModeRetriever

SUPPORTED_EVAL_MODES: tuple[Mode, Mode, Mode] = ("dense", "sparse", "hybrid")


class EvalCase(BaseModel):
    """One retrieval evaluation query with relevant passage IDs."""

    query: str = Field(min_length=1)
    relevant_passage_ids: list[str] = Field(default_factory=list)


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
    max_queries: int | None = Field(default=None, ge=1)
    query_count: int = Field(ge=0)
    passages_path: str
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
    top_k: int,
    max_queries: int | None,
    output_path: Path,
) -> RetrievalEvalReport:
    """Run retrieval-only baseline evaluation for dense/sparse/hybrid modes."""

    _ensure_positive_k(top_k)
    eval_cases = build_eval_cases_from_passages_jsonl(
        settings.passages_path,
        max_queries=max_queries,
    )
    mode_summaries = {
        mode: evaluate_mode(settings, mode=mode, cases=eval_cases, top_k=top_k)
        for mode in SUPPORTED_EVAL_MODES
    }

    report = RetrievalEvalReport(
        run_config=EvalRunConfig(
            top_k=top_k,
            max_queries=max_queries,
            query_count=len(eval_cases),
            passages_path=str(settings.passages_path.resolve()),
            qdrant_url=settings.qdrant_url,
            qdrant_collection=settings.qdrant_collection,
            qdrant_vector_name=settings.qdrant_vector_name,
            qdrant_sparse_vector_name=settings.qdrant_sparse_vector_name,
            sparse_identifiers=load_sparse_eval_identifiers(settings),
        ),
        modes=mode_summaries,
    )
    write_eval_report(output_path, report)
    return report


def build_eval_cases_from_passages_jsonl(
    passages_path: Path, *, max_queries: int | None
) -> list[EvalCase]:
    """Build deterministic eval cases from silver passages + long-answer overlap."""

    passages = PassageStore.read_jsonl(passages_path)
    relevant_by_question: dict[str, set[str]] = {}

    for passage in passages:
        question = (passage.question or "").strip()
        if not question:
            continue
        if not _is_relevant_passage(passage):
            continue
        relevant = relevant_by_question.setdefault(question, set())
        relevant.add(passage.passage_id)

    questions_sorted = sorted(relevant_by_question.keys())
    cases: list[EvalCase] = []
    for question in questions_sorted:
        cases.append(
            EvalCase(
                query=question,
                relevant_passage_ids=sorted(relevant_by_question[question]),
            )
        )
        if max_queries is not None and len(cases) >= max_queries:
            break
    return cases


def evaluate_mode(
    settings: Settings,
    *,
    mode: Mode,
    cases: list[EvalCase],
    top_k: int,
) -> ModeMetricSummary:
    """Evaluate one retrieval mode over provided eval cases."""

    if not cases:
        return ModeMetricSummary(query_count=0, recall_at_k=0.0, mrr_at_k=0.0, ndcg_at_k=0.0)

    retriever = QdrantModeRetriever(settings=settings, mode=mode)
    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []

    for case in cases:
        hits = retriever.retrieve(case.query, top_k=top_k)
        retrieved_ids = [hit.passage_id for hit in hits]
        relevant = set(case.relevant_passage_ids)
        recalls.append(compute_recall_at_k(retrieved_ids, relevant, k=top_k))
        mrrs.append(compute_mrr_at_k(retrieved_ids, relevant, k=top_k))
        ndcgs.append(compute_ndcg_at_k(retrieved_ids, relevant, k=top_k))

    return ModeMetricSummary(
        query_count=len(cases),
        recall_at_k=sum(recalls) / float(len(recalls)),
        mrr_at_k=sum(mrrs) / float(len(mrrs)),
        ndcg_at_k=sum(ndcgs) / float(len(ndcgs)),
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


def _ensure_positive_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}.")
