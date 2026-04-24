"""Milestone 3 retrievers: dense, sparse, and hybrid retrieval backed by Qdrant."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, Protocol

from src.config.settings import Settings
from src.models.query_schemas import PassageHit
from src.retrieval.sparse_qdrant import (
    SparsePass1Data,
    encode_query_sparse_vector,
    load_sparse_pass1_artifact,
)


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


class RetrieverLike(Protocol):
    """Minimal retriever contract used for mode routing."""

    def retrieve(self, query: str, top_k: int) -> list[PassageHit]: ...


@dataclass(slots=True)
class DenseQdrantRetriever:
    """Dense retriever backed by Qdrant named dense vector search."""

    settings: Settings
    client: QdrantQueryClientLike | None = None
    model: QueryEmbeddingModel | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = _build_default_qdrant_client(self.settings.qdrant_url)
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
        return _to_hits(_extract_points(response), score_field="dense_score", rank_field="dense_rank")


@dataclass(slots=True)
class SparseQdrantRetriever:
    """Sparse retriever backed by Qdrant sparse vector search."""

    settings: Settings
    client: QdrantQueryClientLike | None = None
    pass1_data: SparsePass1Data | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = _build_default_qdrant_client(self.settings.qdrant_url)
        if self.pass1_data is None:
            self.pass1_data = load_sparse_pass1_artifact(
                path=self.settings.sparse_pass1_path,
                silver_path=self.settings.passages_path,
                max_passages=self.settings.max_passages,
            )

    def retrieve(self, query: str, top_k: int) -> list[PassageHit]:
        indices, values = encode_query_sparse_vector(query, term_to_id=self.pass1_data.term_to_id)
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

    def __post_init__(self) -> None:
        if self.dense is None:
            self.dense = DenseQdrantRetriever(settings=self.settings)
        if self.sparse is None:
            self.sparse = SparseQdrantRetriever(settings=self.settings)

    def retrieve(self, query: str, top_k: int) -> list[PassageHit]:
        dense_hits = self.dense.retrieve(query, top_k)
        sparse_hits = self.sparse.retrieve(query, top_k)
        combined: dict[str, PassageHit] = {}
        fusion_score: dict[str, float] = {}
        rrf_k = float(self.settings.hybrid_rrf_k)

        for hit in dense_hits:
            combined[hit.passage_id] = hit.model_copy(deep=True)
            if hit.dense_rank is not None:
                fusion_score[hit.passage_id] = fusion_score.get(hit.passage_id, 0.0) + (
                    self.settings.hybrid_dense_weight / (rrf_k + float(hit.dense_rank))
                )

        for hit in sparse_hits:
            existing = combined.get(hit.passage_id)
            if existing is None:
                combined[hit.passage_id] = hit.model_copy(deep=True)
            else:
                existing.sparse_score = hit.sparse_score
                existing.sparse_rank = hit.sparse_rank
            if hit.sparse_rank is not None:
                fusion_score[hit.passage_id] = fusion_score.get(hit.passage_id, 0.0) + (
                    self.settings.hybrid_sparse_weight / (rrf_k + float(hit.sparse_rank))
                )

        ranked = sorted(
            combined.values(),
            key=lambda hit: (-fusion_score.get(hit.passage_id, 0.0), hit.passage_id),
        )[:top_k]
        for rank, hit in enumerate(ranked, start=1):
            hit.fusion_rank = rank
        return ranked


@dataclass(slots=True)
class QdrantModeRetriever:
    """Single retriever entrypoint routing to dense, sparse, or hybrid mode."""

    settings: Settings
    mode: Mode
    dense: RetrieverLike | None = None
    sparse: RetrieverLike | None = None
    hybrid: RetrieverLike | None = None

    def retrieve(self, query: str, top_k: int) -> list[PassageHit]:
        if self.mode == "dense":
            if self.dense is None:
                self.dense = DenseQdrantRetriever(settings=self.settings)
            return self.dense.retrieve(query, top_k)
        if self.mode == "sparse":
            if self.sparse is None:
                self.sparse = SparseQdrantRetriever(settings=self.settings)
            return self.sparse.retrieve(query, top_k)
        if self.mode == "hybrid":
            if self.hybrid is None:
                dense = self.dense or DenseQdrantRetriever(settings=self.settings)
                sparse = self.sparse or SparseQdrantRetriever(settings=self.settings)
                self.hybrid = HybridQdrantRetriever(
                    settings=self.settings,
                    dense=dense,
                    sparse=sparse,
                )
            return self.hybrid.retrieve(query, top_k)
        raise ValueError(f"Unsupported retrieval mode {self.mode!r}.")


def _build_default_qdrant_client(url: str) -> Any:
    try:
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "qdrant-client is required for retrieval. Install project dependencies first."
        ) from exc

    try:
        return QdrantClient(url=url, check_compatibility=False)
    except TypeError:
        return QdrantClient(url=url)


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
        source = payload.get("source")
        score = float(getattr(point, "score", 0.0))
        data: dict[str, Any] = {
            "passage_id": str(getattr(point, "id", "")),
            "text": text,
            "source": source if isinstance(source, str) else None,
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
