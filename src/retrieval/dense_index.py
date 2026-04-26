"""Dense index build helpers backed by Qdrant."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from src.config.settings import Settings
from src.ingestion.models import IndexChunk, Passage
from src.observability.progress import ProgressTicker, count_non_empty_jsonl

LOGGER = logging.getLogger(__name__)

_UPSERT_RETRY_MAX_ATTEMPTS = 3
_UPSERT_RETRY_BASE_SLEEP_SECONDS = 1.0
_DENSE_CHECKPOINT_SCHEMA_VERSION = "1"


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

    def upsert(self, collection_name: str, points: list[Any]) -> None: ...

    def count(self, collection_name: str, exact: bool = True) -> Any: ...


@dataclass(slots=True)
class DenseBuildResult:
    """Dense build metadata."""

    vector_count: int
    vector_size: int


def _qdrant_payload(record: IndexChunk | Passage) -> dict[str, object]:
    """Serialize fields stored alongside dense vectors."""

    payload: dict[str, object] = {"text": record.text}
    for name in (
        "context_text",
        "source",
        "title",
        "question",
        "document_url",
        "group_id",
        "chunk_kind",
    ):
        value = getattr(record, name, None)
        if value is not None:
            payload[name] = value
    if isinstance(record, IndexChunk):
        payload.update(
            {
                "chunk_id": record.chunk_id,
                "source_row_ordinal": record.source_row_ordinal,
                "start_candidate_idx": record.start_candidate_idx,
                "end_candidate_idx": record.end_candidate_idx,
                "passage_types": record.passage_types,
                "token_count": record.token_count,
                "context_token_count": record.context_token_count,
            }
        )
        if record.parent_candidate_idx is not None:
            payload["parent_candidate_idx"] = record.parent_candidate_idx
    elif record.passage_type is not None:
        payload["passage_type"] = record.passage_type
    if record.long_answers:
        payload["long_answers"] = record.long_answers
    return payload


def _record_id(record: IndexChunk | Passage) -> str:
    return record.chunk_id if isinstance(record, IndexChunk) else record.passage_id


def _record_from_json(line: str) -> IndexChunk | Passage:
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL line must decode to an object.")
    if "chunk_id" in payload:
        return IndexChunk.model_validate(payload)
    return Passage.model_validate(payload)


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

    def build_from_jsonl_streaming(
        self,
        jsonl_path: Path,
        *,
        lines_per_batch: int,
        max_passages: int | None = None,
    ) -> DenseBuildResult:
        """Encode and upsert passages by reading silver JSONL in bounded line batches.

        When ``max_passages`` is set, only the first N non-empty silver lines are indexed.
        """

        self._ensure_collection(vector_size=self._embedding_dimension())
        checkpoint_path = self._settings.dense_checkpoint_path
        resume_count = _load_dense_checkpoint(
            path=checkpoint_path,
            silver_path=jsonl_path,
            collection_name=self._settings.qdrant_collection,
            vector_name=self._settings.qdrant_vector_name,
            max_passages=max_passages,
        )
        if resume_count:
            LOGGER.info("resuming dense from checkpoint at %s passages", resume_count)
        total_records = count_non_empty_jsonl(jsonl_path, max_records=max_passages)
        remaining_records = max(total_records - resume_count, 0)
        total_batches = math.ceil(remaining_records / lines_per_batch) if remaining_records else 0
        ticker = ProgressTicker(
            logger=LOGGER,
            stage="dense_index",
            label="batches",
            total=total_batches,
            every_items=self._settings.progress_log_every_batches,
            every_seconds=self._settings.progress_log_every_seconds,
        )
        ticker.start(
            input=jsonl_path,
            collection=self._settings.qdrant_collection,
            vector_name=self._settings.qdrant_vector_name,
            records_total=total_records,
            read_batch_lines=lines_per_batch,
            embedding_batch_size=self._settings.embedding_batch_size,
            checkpoint=checkpoint_path,
        )
        vector_size = 0
        total_vectors = resume_count
        batches_completed = 0
        batch: list[IndexChunk | Passage] = []
        consumed_non_empty_lines = 0

        def flush() -> None:
            nonlocal batches_completed, vector_size, total_vectors, batch
            if not batch:
                return
            texts = [p.text for p in batch]
            vectors = self._model.encode(
                texts,
                batch_size=self._settings.embedding_batch_size,
                normalize_embeddings=True,
            )
            points = [
                {
                    "id": _record_id(passage),
                    "vector": {self._settings.qdrant_vector_name: list(vector)},
                    "payload": _qdrant_payload(passage),
                }
                for passage, vector in zip(batch, vectors, strict=True)
            ]
            _upsert_with_retry(
                client=self._client,
                collection_name=self._settings.qdrant_collection,
                points=points,
            )
            total_vectors += len(points)
            _write_dense_checkpoint(
                path=checkpoint_path,
                silver_path=jsonl_path,
                collection_name=self._settings.qdrant_collection,
                vector_name=self._settings.qdrant_vector_name,
                max_passages=max_passages,
                indexed_count=total_vectors,
            )
            if points:
                vec_key = self._settings.qdrant_vector_name
                vector_size = len(points[0]["vector"][vec_key])
            batches_completed += 1
            ticker.tick(
                batches_completed,
                records_indexed=total_vectors,
                records_total=total_records,
                vector_size=vector_size,
                checkpoint=checkpoint_path,
            )
            batch = []

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if max_passages is not None and total_vectors + len(batch) >= max_passages:
                    break
                line = raw.strip()
                if not line:
                    continue
                if consumed_non_empty_lines < resume_count:
                    consumed_non_empty_lines += 1
                    continue
                batch.append(_record_from_json(line))
                if len(batch) >= lines_per_batch:
                    flush()
        flush()
        _remove_dense_checkpoint(checkpoint_path)
        ticker.finish(
            batches_completed,
            records_indexed=total_vectors,
            records_total=total_records,
            vector_size=vector_size,
        )

        return DenseBuildResult(vector_count=total_vectors, vector_size=vector_size)

    def count(self) -> int:
        """Return number of vectors in collection."""

        result = self._client.count(collection_name=self._settings.qdrant_collection, exact=True)
        return int(result.count)

    def _ensure_collection(self, vector_size: int) -> None:
        if self._client.collection_exists(self._settings.qdrant_collection):
            return
        from qdrant_client.http import models as qm

        self._client.create_collection(
            collection_name=self._settings.qdrant_collection,
            vectors_config={
                self._settings.qdrant_vector_name: _build_vector_params(
                    size=vector_size,
                    distance_name=self._settings.qdrant_distance,
                )
            },
            sparse_vectors_config={
                self._settings.qdrant_sparse_vector_name: qm.SparseVectorParams(),
            },
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

    try:
        return QdrantClient(url=url, check_compatibility=False)
    except TypeError:
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


def _upsert_with_retry(
    *, client: QdrantLikeClient, collection_name: str, points: list[Any]
) -> None:
    """Best-effort retry wrapper for transient upsert failures (timeouts, temporary overload)."""

    delay = _UPSERT_RETRY_BASE_SLEEP_SECONDS
    for attempt in range(1, _UPSERT_RETRY_MAX_ATTEMPTS + 1):
        try:
            client.upsert(collection_name=collection_name, points=points)
            return
        except Exception:
            if attempt >= _UPSERT_RETRY_MAX_ATTEMPTS:
                raise
            LOGGER.warning(
                "qdrant upsert failed (attempt %s/%s); retrying in %.1fs",
                attempt,
                _UPSERT_RETRY_MAX_ATTEMPTS,
                delay,
            )
            time.sleep(delay)
            delay *= 2.0


def _remove_dense_checkpoint(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _load_dense_checkpoint(
    *,
    path: Path,
    silver_path: Path,
    collection_name: str,
    vector_name: str,
    max_passages: int | None,
) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return 0
    expected = {
        "schema_version": _DENSE_CHECKPOINT_SCHEMA_VERSION,
        "silver_path": str(silver_path.resolve()),
        "collection_name": collection_name,
        "vector_name": vector_name,
        "max_passages": max_passages,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            return 0
    indexed_count = payload.get("indexed_count")
    if not isinstance(indexed_count, int) or indexed_count < 0:
        return 0
    return indexed_count


def _write_dense_checkpoint(
    *,
    path: Path,
    silver_path: Path,
    collection_name: str,
    vector_name: str,
    max_passages: int | None,
    indexed_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": _DENSE_CHECKPOINT_SCHEMA_VERSION,
        "silver_path": str(silver_path.resolve()),
        "collection_name": collection_name,
        "vector_name": vector_name,
        "max_passages": max_passages,
        "indexed_count": indexed_count,
        "updated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _build_vector_params(size: int, distance_name: str) -> Any:
    from qdrant_client.http import models as qm

    # qdrant-client 1.13+: ``Distance.COSINE``; settings keep title case (``Cosine``).
    attr = distance_name.upper()
    return qm.VectorParams(size=size, distance=getattr(qm.Distance, attr))
