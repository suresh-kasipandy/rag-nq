"""Dense index build helpers backed by Qdrant."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.config.settings import Settings
from src.ingestion.models import Passage


class EmbeddingModel(Protocol):
    """Protocol for sentence embedding model."""

    def encode(
        self, texts: Sequence[str], *, batch_size: int, normalize_embeddings: bool
    ) -> Sequence[Sequence[float]]:
        """Encode input texts to dense vectors."""

    def get_sentence_embedding_dimension(self) -> int: ...


class QdrantLikeClient(Protocol):
    """Protocol for Qdrant operations used in Milestone 1."""

    def collection_exists(self, collection_name: str) -> bool: ...

    def create_collection(self, collection_name: str, vectors_config: Any) -> None: ...

    def upsert(self, collection_name: str, points: list[dict[str, Any]]) -> None: ...

    def count(self, collection_name: str, exact: bool = True) -> Any: ...


@dataclass(slots=True)
class DenseBuildResult:
    """Dense build metadata."""

    vector_count: int
    vector_size: int


def _qdrant_payload(passage: Passage) -> dict[str, object]:
    """Serialize passage fields stored alongside dense vectors."""

    payload: dict[str, object] = {"text": passage.text}
    if passage.source is not None:
        payload["source"] = passage.source
    if passage.title is not None:
        payload["title"] = passage.title
    if passage.question is not None:
        payload["question"] = passage.question
    if passage.passage_type is not None:
        payload["passage_type"] = passage.passage_type
    if passage.document_url is not None:
        payload["document_url"] = passage.document_url
    if passage.long_answers:
        payload["long_answers"] = passage.long_answers
    return payload


class DenseIndexer:
    """Dense index builder that writes embeddings into Qdrant."""

    def __init__(
        self,
        settings: Settings,
        client: QdrantLikeClient | None = None,
        model: EmbeddingModel | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or _build_default_qdrant_client(settings.qdrant_url)
        self._model = model or _build_default_embedding_model(settings.embedding_model_name)

    def build(self, passages: list[Passage]) -> DenseBuildResult:
        """Create or reuse collection and upsert all passage vectors."""

        self._ensure_collection(vector_size=self._embedding_dimension())
        texts = [passage.text for passage in passages]
        vectors = self._model.encode(
            texts,
            batch_size=self._settings.embedding_batch_size,
            normalize_embeddings=True,
        )
        points = [
            {
                "id": passage.passage_id,
                "vector": {self._settings.qdrant_vector_name: list(vector)},
                "payload": _qdrant_payload(passage),
            }
            for passage, vector in zip(passages, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._settings.qdrant_collection, points=points)

        vector_size = len(points[0]["vector"][self._settings.qdrant_vector_name]) if points else 0
        return DenseBuildResult(vector_count=len(points), vector_size=vector_size)

    def count(self) -> int:
        """Return number of vectors in collection."""

        result = self._client.count(collection_name=self._settings.qdrant_collection, exact=True)
        return int(result.count)

    def _ensure_collection(self, vector_size: int) -> None:
        if self._client.collection_exists(self._settings.qdrant_collection):
            return
        self._client.create_collection(
            collection_name=self._settings.qdrant_collection,
            vectors_config=_build_vector_params(
                size=vector_size,
                distance_name=self._settings.qdrant_distance,
            ),
        )

    def _embedding_dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())


def _build_default_qdrant_client(url: str) -> Any:
    try:
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "qdrant-client is required for dense indexing. Install project dependencies first."
        ) from exc

    return QdrantClient(url=url)


def _build_default_embedding_model(model_name: str) -> EmbeddingModel:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentence-transformers is required for dense indexing. "
            "Install project dependencies first."
        ) from exc

    return SentenceTransformer(model_name)


def _build_vector_params(size: int, distance_name: str) -> Any:
    from qdrant_client.http import models as qm

    return qm.VectorParams(size=size, distance=getattr(qm.Distance, distance_name))
