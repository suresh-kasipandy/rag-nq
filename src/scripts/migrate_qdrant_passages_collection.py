"""One-time CLI: copy dense points from a legacy collection into a new dense+sparse-slot collection.

Scrolls the source with stored vectors and payloads (no re-embedding), upserts only the named
dense vector into the target, then you can run ``index_sparse`` to attach BM25 sparse vectors.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.config.settings import Settings
from src.observability.logging_setup import get_stage_logger, setup_logging
from src.retrieval.dense_index import _upsert_with_retry

LOGGER = get_stage_logger(__name__)


class QdrantMigrateClient(Protocol):
    """Operations used by ``migrate_passages_collection`` (``QdrantClient`` or test doubles)."""

    def collection_exists(self, collection_name: str) -> bool: ...

    def get_collection(self, collection_name: str) -> Any: ...

    def create_collection(
        self,
        collection_name: str,
        *,
        vectors_config: Any,
        sparse_vectors_config: Any | None = None,
    ) -> Any: ...

    def delete_collection(self, collection_name: str) -> Any: ...

    def scroll(
        self,
        collection_name: str,
        *,
        limit: int,
        offset: Any,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[list[Any], Any]: ...

    def upsert(self, collection_name: str, points: list[Any]) -> Any: ...

    def count(self, collection_name: str, *, exact: bool = True) -> Any: ...


def _dense_vector_params(source_info: Any, vector_name: str) -> qm.VectorParams:
    """Return ``VectorParams`` for the named dense vector from ``get_collection`` result."""

    params = source_info.config.params
    vectors = params.vectors
    if not isinstance(vectors, dict):
        msg = f"Expected named vectors dict on source collection; got {type(vectors).__name__}"
        raise ValueError(msg)
    if vector_name not in vectors:
        names = ", ".join(sorted(vectors))
        raise ValueError(f"Source collection has no vector named {vector_name!r}. Present: {names}")
    return vectors[vector_name]


def _record_to_point_struct(record: Any, vector_name: str) -> qm.PointStruct:
    """Build a ``PointStruct`` so upserts work with HTTP and in-memory Qdrant clients."""

    raw_vec = record.vector
    if not isinstance(raw_vec, dict):
        msg = f"Point {record.id!r}: expected named vector dict; got {type(raw_vec).__name__}"
        raise ValueError(msg)
    if vector_name not in raw_vec:
        keys = ", ".join(sorted(raw_vec))
        raise ValueError(f"Point {record.id!r}: missing vector {vector_name!r}. Keys: {keys}")
    dense = raw_vec[vector_name]
    return qm.PointStruct(
        id=record.id,
        vector={vector_name: list(dense)},
        payload=record.payload,
    )


def migrate_passages_collection(
    client: QdrantMigrateClient,
    *,
    source_collection: str,
    target_collection: str,
    vector_name: str,
    sparse_vector_name: str,
    batch_size: int,
    recreate_target: bool,
    dry_run: bool,
) -> None:
    """Run migration; raises ``ValueError`` on invalid configuration or count mismatch."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    source_info = client.get_collection(source_collection)
    _dense_vector_params(source_info, vector_name)
    LOGGER.info(
        "preflight ok source=%s vector=%s",
        source_collection,
        vector_name,
        extra={"stage": "qdrant_migrate"},
    )

    if dry_run:
        records, _ = client.scroll(
            collection_name=source_collection,
            limit=batch_size,
            offset=None,
            with_payload=True,
            with_vectors=True,
        )
        LOGGER.info(
            "dry-run: first scroll page size=%s (no writes)",
            len(records),
            extra={"stage": "qdrant_migrate"},
        )
        return

    if client.collection_exists(target_collection):
        if not recreate_target:
            raise ValueError(
                f"Target collection {target_collection!r} already exists. "
                "Pass --recreate-target to delete it first (destructive)."
            )
        LOGGER.warning(
            "deleting existing target collection=%s (--recreate-target)",
            target_collection,
            extra={"stage": "qdrant_migrate"},
        )
        client.delete_collection(target_collection)

    dense_params = _dense_vector_params(source_info, vector_name)
    client.create_collection(
        target_collection,
        vectors_config={vector_name: dense_params},
        sparse_vectors_config={sparse_vector_name: qm.SparseVectorParams()},
    )
    LOGGER.info(
        "created target=%s with dense=%s sparse_slot=%s",
        target_collection,
        vector_name,
        sparse_vector_name,
        extra={"stage": "qdrant_migrate"},
    )

    offset: Any = None
    total = 0
    t0 = time.perf_counter()
    while True:
        records, offset = client.scroll(
            collection_name=source_collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            break
        points = [_record_to_point_struct(rec, vector_name) for rec in records]
        _upsert_with_retry(client=client, collection_name=target_collection, points=points)
        total += len(points)
        LOGGER.info(
            "upserted batch size=%s total=%s",
            len(points),
            total,
            extra={"stage": "qdrant_migrate"},
        )
        if offset is None:
            break

    elapsed = time.perf_counter() - t0
    src_count = int(client.count(source_collection, exact=True).count)
    tgt_count = int(client.count(target_collection, exact=True).count)
    LOGGER.info(
        "done copied_points=%s source_count=%s target_count=%s elapsed_s=%.2f",
        total,
        src_count,
        tgt_count,
        elapsed,
        extra={"stage": "qdrant_migrate"},
    )
    if src_count != tgt_count or total != tgt_count:
        raise ValueError(
            f"Count mismatch after migration: upserted={total} source_count={src_count} "
            f"target_count={tgt_count}"
        )


def _build_qdrant_client(url: str) -> QdrantClient:
    try:
        return QdrantClient(url=url, check_compatibility=False)
    except TypeError:
        return QdrantClient(url=url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-collection",
        default="nq_passages_dense",
        help="Legacy collection name (default: %(default)s)",
    )
    parser.add_argument(
        "--target-collection",
        default="nq_passages",
        help="New collection name (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Scroll/upsert batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--recreate-target",
        action="store_true",
        help="If the target collection exists, delete it before creating (destructive).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preflight source collection and report first scroll page size only; no writes.",
    )
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    settings = Settings.from_env()
    client = _build_qdrant_client(settings.qdrant_url)

    try:
        migrate_passages_collection(
            client,
            source_collection=args.source_collection,
            target_collection=args.target_collection,
            vector_name=settings.qdrant_vector_name,
            sparse_vector_name=settings.qdrant_sparse_vector_name,
            batch_size=args.batch_size,
            recreate_target=args.recreate_target,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
