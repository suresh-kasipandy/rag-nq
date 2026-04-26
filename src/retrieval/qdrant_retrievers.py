"""Qdrant-backed dense, sparse, hybrid, and reranked retrieval."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, Protocol

from src.config.settings import Settings
from src.contracts.retrieval import Retriever
from src.models.query_schemas import (
    DedupeMetrics,
    PassageHit,
    RetrievalMetrics,
    RetrievalStageTimings,
)
from src.retrieval.rerank import (
    CrossEncoderLike,
    build_default_cross_encoder,
    dedupe_hits,
    rerank_hits,
)
from src.retrieval.sparse_qdrant import (
    SparsePass1Data,
    encode_query_sparse_vector,
    load_sparse_pass1_artifact,
)

LOGGER = logging.getLogger(__name__)


class QueryEmbeddingModel(Protocol):
    """Embedding protocol required by dense query-time retrieval."""

    def encode(
        self, texts: list[str], *, batch_size: int, normalize_embeddings: bool
    ) -> list[list[float]]: ...


class QdrantQueryClientLike(Protocol):
    """Qdrant query operations used by Milestone 3 retrievers."""

    def query_points(
        self,
        collection_name: str,
        query: Any,
        *,
        using: str,
        limit: int,
        with_payload: bool,
    ) -> Any: ...


Mode = Literal["dense", "sparse", "hybrid"]


@dataclass(slots=True)
class DenseQdrantRetriever:
    """Dense retriever backed by Qdrant named dense vector search."""

    settings: Settings
    client: QdrantQueryClientLike | None = None
    model: QueryEmbeddingModel | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = _build_default_qdrant_client(
                self.settings.qdrant_url,
                timeout=self.settings.qdrant_retrieval_timeout_seconds,
            )
        if self.model is None:
            self.model = _build_default_embedding_model(self.settings.embedding_model_name)

    def retrieve(self, query: str, top_k: int) -> list[PassageHit]:
        vector = self.model.encode(
            [query], batch_size=1, normalize_embeddings=True
        )[0]
        response = self.client.query_points(
            self.settings.qdrant_collection,
            query=list(vector),
            using=self.settings.qdrant_vector_name,
            limit=top_k,
            with_payload=True,
        )
        return _to_hits(
            _extract_points(response),
            score_field="dense_score",
            rank_field="dense_rank",
        )


@dataclass(slots=True)
class SparseQdrantRetriever:
    """Sparse retriever backed by Qdrant sparse vector search."""

    settings: Settings
    client: QdrantQueryClientLike | None = None
    pass1_data: SparsePass1Data | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = _build_default_qdrant_client(
                self.settings.qdrant_url,
                timeout=self.settings.qdrant_retrieval_timeout_seconds,
            )
        if self.pass1_data is None:
            self.pass1_data = load_sparse_pass1_artifact(
                path=self.settings.sparse_pass1_path,
                silver_path=self.settings.index_chunks_path,
                max_index_rows=self.settings.max_index_rows,
                max_passages=self.settings.max_passages,
                sparse_analyzer=self.settings.sparse_analyzer,
            )

    def retrieve(self, query: str, top_k: int) -> list[PassageHit]:
        indices, values = encode_query_sparse_vector(
            query,
            term_to_id=self.pass1_data.term_to_id,
            analyzer=self.pass1_data.sparse_analyzer,
        )
        if not indices:
            return []
        sparse_query = _build_sparse_query(indices=indices, values=values)
        response = self.client.query_points(
            self.settings.qdrant_collection,
            query=sparse_query,
            using=self.settings.qdrant_sparse_vector_name,
            limit=top_k,
            with_payload=True,
        )
        return _to_hits(
            _extract_points(response),
            score_field="sparse_score",
            rank_field="sparse_rank",
        )


@dataclass(slots=True)
class HybridQdrantRetriever:
    """Hybrid retriever using weighted reciprocal-rank fusion over dense+sparse ranks."""

    settings: Settings
    dense: DenseQdrantRetriever | None = None
    sparse: SparseQdrantRetriever | None = None
    reranker: CrossEncoderLike | None = None
    last_dedupe_metrics: DedupeMetrics | None = None
    last_stage_timings: RetrievalStageTimings | None = None

    def __post_init__(self) -> None:
        if self.dense is None:
            self.dense = DenseQdrantRetriever(settings=self.settings)
        if self.sparse is None:
            self.sparse = SparseQdrantRetriever(settings=self.settings)
        if self.settings.rerank_enabled and self.reranker is None:
            self.reranker = build_default_cross_encoder(self.settings.rerank_model_name)

    def retrieve(self, query: str, top_k: int) -> list[PassageHit]:
        started_at = time.monotonic()
        candidate_k = max(top_k, self.settings.retrieve_k)
        dense_hits = self.dense.retrieve(query, candidate_k)
        sparse_hits = self.sparse.retrieve(query, candidate_k)
        retrieved_at = time.monotonic()
        combined: dict[str, PassageHit] = {}
        fusion_score: dict[str, float] = {}
        rrf_k = float(self.settings.hybrid_rrf_k)

        for hit in dense_hits:
            combined[hit.point_id] = hit.model_copy(deep=True)
            if hit.dense_rank is not None:
                fusion_score[hit.point_id] = fusion_score.get(hit.point_id, 0.0) + (
                    self.settings.hybrid_dense_weight / (rrf_k + float(hit.dense_rank))
                )

        for hit in sparse_hits:
            existing = combined.get(hit.point_id)
            if existing is None:
                combined[hit.point_id] = hit.model_copy(deep=True)
            else:
                existing.sparse_score = hit.sparse_score
                existing.sparse_rank = hit.sparse_rank
            if hit.sparse_rank is not None:
                fusion_score[hit.point_id] = fusion_score.get(hit.point_id, 0.0) + (
                    self.settings.hybrid_sparse_weight / (rrf_k + float(hit.sparse_rank))
                )

        ranked = sorted(
            combined.values(),
            key=lambda hit: (-fusion_score.get(hit.point_id, 0.0), hit.point_id),
        )
        for rank, hit in enumerate(ranked, start=1):
            hit.fusion_rank = rank

        fused_at = time.monotonic()
        if self.settings.rerank_enabled and self.reranker is not None:
            ranked = rerank_hits(
                query=query,
                hits=ranked,
                model=self.reranker,
                context_token_budget=self.settings.rerank_context_token_budget,
            )
        reranked_at = time.monotonic()

        final_k = min(top_k, self.settings.rerank_k or top_k)
        if self.settings.retrieval_dedupe_enabled:
            deduped = dedupe_hits(ranked, top_k=final_k)
            final_hits = deduped.hits
            self.last_dedupe_metrics = deduped.metrics
        else:
            final_hits = ranked[:final_k]
            self.last_dedupe_metrics = None
        finished_at = time.monotonic()
        self.last_stage_timings = RetrievalStageTimings(
            retrieve_seconds=retrieved_at - started_at,
            fusion_seconds=fused_at - retrieved_at,
            rerank_seconds=reranked_at - fused_at,
            dedupe_seconds=finished_at - reranked_at,
            total_seconds=finished_at - started_at,
        )
        LOGGER.info(
            "hybrid retrieval complete raw=%s final=%s dedupe_drops=%s timings=%s",
            len(ranked),
            len(final_hits),
            self.last_dedupe_metrics.dedupe_drop_count if self.last_dedupe_metrics else 0,
            self.last_stage_timings.model_dump(),
        )
        return final_hits


@dataclass(slots=True)
class QdrantModeRetriever:
    """Single retriever entrypoint routing to dense, sparse, or hybrid mode."""

    settings: Settings
    mode: Mode
    dense: Retriever | None = None
    sparse: Retriever | None = None
    hybrid: Retriever | None = None
    last_retrieval_metrics: RetrievalMetrics | None = None

    def retrieve(self, query: str, top_k: int) -> list[PassageHit]:
        if self.mode == "dense":
            if self.dense is None:
                self.dense = DenseQdrantRetriever(settings=self.settings)
            return self._retrieve_single_mode(self.dense, query=query, top_k=top_k)
        if self.mode == "sparse":
            if self.sparse is None:
                self.sparse = SparseQdrantRetriever(settings=self.settings)
            return self._retrieve_single_mode(self.sparse, query=query, top_k=top_k)
        if self.mode == "hybrid":
            if self.hybrid is None:
                dense = self.dense or DenseQdrantRetriever(settings=self.settings)
                sparse = self.sparse or SparseQdrantRetriever(settings=self.settings)
                self.hybrid = HybridQdrantRetriever(
                    settings=self.settings,
                    dense=dense,
                    sparse=sparse,
                )
            hits = self.hybrid.retrieve(query, top_k)
            dedupe_metrics = getattr(self.hybrid, "last_dedupe_metrics", None)
            timings = getattr(self.hybrid, "last_stage_timings", None)
            self.last_retrieval_metrics = (
                RetrievalMetrics(dedupe=dedupe_metrics, timings=timings)
                if dedupe_metrics is not None or timings is not None
                else None
            )
            return hits
        raise ValueError(f"Unsupported retrieval mode {self.mode!r}.")

    def _retrieve_single_mode(
        self, retriever: Retriever, *, query: str, top_k: int
    ) -> list[PassageHit]:
        started_at = time.monotonic()
        candidate_k = max(top_k, self.settings.retrieve_k)
        hits = retriever.retrieve(query, candidate_k)
        retrieved_at = time.monotonic()
        if not self.settings.retrieval_dedupe_enabled:
            finished_at = time.monotonic()
            self.last_retrieval_metrics = RetrievalMetrics(
                timings=RetrievalStageTimings(
                    retrieve_seconds=retrieved_at - started_at,
                    total_seconds=finished_at - started_at,
                )
            )
            return hits[:top_k]
        deduped = dedupe_hits(hits, top_k=top_k)
        finished_at = time.monotonic()
        self.last_retrieval_metrics = RetrievalMetrics(
            dedupe=deduped.metrics,
            timings=RetrievalStageTimings(
                retrieve_seconds=retrieved_at - started_at,
                dedupe_seconds=finished_at - retrieved_at,
                total_seconds=finished_at - started_at,
            ),
        )
        return deduped.hits


def _build_default_qdrant_client(url: str, *, timeout: float) -> Any:
    try:
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "qdrant-client is required for retrieval. Install project dependencies first."
        ) from exc

    try:
        return QdrantClient(url=url, timeout=timeout, check_compatibility=False)
    except TypeError:
        return QdrantClient(url=url, timeout=timeout)


def _build_default_embedding_model(model_name: str) -> QueryEmbeddingModel:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentence-transformers is required for dense retrieval. Install dependencies first."
        ) from exc
    return SentenceTransformer(model_name)


def _extract_points(response: Any) -> list[Any]:
    points = getattr(response, "points", None)
    if points is None:
        if isinstance(response, list):
            return response
        return []
    return list(points)


def _to_hits(points: list[Any], *, score_field: str, rank_field: str) -> list[PassageHit]:
    hits: list[PassageHit] = []
    for rank, point in enumerate(points, start=1):
        payload = getattr(point, "payload", None) or {}
        text = payload.get("text")
        if not isinstance(text, str) or not text:
            continue
        context_text = payload.get("context_text")
        title = payload.get("title")
        document_url = payload.get("document_url")
        group_id = payload.get("group_id")
        chunk_kind = payload.get("chunk_kind")
        source_row_ordinal = payload.get("source_row_ordinal")
        start_candidate_idx = payload.get("start_candidate_idx")
        end_candidate_idx = payload.get("end_candidate_idx")
        passage_types = payload.get("passage_types")
        score = float(getattr(point, "score", 0.0))
        point_id = str(getattr(point, "id", ""))
        data: dict[str, Any] = {
            "point_id": point_id,
            "text": text,
            "context_text": context_text if isinstance(context_text, str) else None,
            "title": title if isinstance(title, str) else None,
            "document_url": document_url if isinstance(document_url, str) else None,
            "group_id": group_id if isinstance(group_id, str) else None,
            "chunk_kind": chunk_kind if isinstance(chunk_kind, str) else None,
            "source_row_ordinal": (
                source_row_ordinal if isinstance(source_row_ordinal, int) else None
            ),
            "start_candidate_idx": (
                start_candidate_idx if isinstance(start_candidate_idx, int) else None
            ),
            "end_candidate_idx": end_candidate_idx if isinstance(end_candidate_idx, int) else None,
            "passage_types": (
                [value for value in passage_types if isinstance(value, str)]
                if isinstance(passage_types, list)
                else []
            ),
            score_field: score,
            rank_field: rank,
        }
        hits.append(PassageHit(**data))
    return hits


def _build_sparse_query(*, indices: list[int], values: list[float]) -> Any:
    try:
        from qdrant_client.http import models as qm
    except ModuleNotFoundError:
        return SimpleNamespace(indices=indices, values=values)
    return qm.SparseVector(indices=indices, values=values)
