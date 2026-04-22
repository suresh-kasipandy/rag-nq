"""CLI: stream silver JSONL into Qdrant sparse vectors (Milestone 2.2)."""

from __future__ import annotations

import logging
import sys

from src.config.settings import Settings
from src.observability.logging_setup import get_stage_logger, setup_logging
from src.retrieval.sparse_qdrant import SparseQdrantIndexer

LOGGER = get_stage_logger(__name__)


def main() -> None:
    setup_logging(level=logging.INFO)
    settings = Settings.from_env()

    if not settings.passages_path.is_file() or not settings.ingest_manifest_path.is_file():
        print(
            "Silver artifact missing. Run: python -m src.scripts.ingest_passages",
            file=sys.stderr,
        )
        raise SystemExit(1)

    indexer = SparseQdrantIndexer(settings=settings)
    result = indexer.build_from_jsonl(settings.passages_path, max_passages=settings.max_passages)
    LOGGER.info(
        "sparse_qdrant_index complete: documents=%s vocab=%s points_updated=%s manifest=%s",
        result.document_count,
        result.vocabulary_size,
        result.points_updated,
        settings.sparse_manifest_path,
        extra={"stage": "sparse_qdrant_index"},
    )


if __name__ == "__main__":
    main()
