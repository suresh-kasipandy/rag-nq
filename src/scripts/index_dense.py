"""CLI: chunked dense embed + Qdrant upsert from index chunks JSONL."""

from __future__ import annotations

import argparse
import logging
import sys

from src.config.settings import Settings
from src.ingestion.chunk_raw import run_chunk_ingest
from src.ingestion.models import INDEX_CHUNK_SCHEMA_VERSION
from src.observability.logging_setup import get_stage_logger, setup_logging
from src.retrieval.dense_index import DenseIndexer

LOGGER = get_stage_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-hf",
        action="store_true",
        help="Refresh raw dataset from Hugging Face before chunking and dense indexing.",
    )
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    settings = Settings.from_env()

    if args.from_hf:
        run_chunk_ingest(settings, force=True)

    if not settings.index_chunks_path.is_file() or not settings.chunk_manifest_path.is_file():
        print(
            "Chunk artifact missing. Run: python -m src.scripts.ingest_chunks\n"
            "Or pass --from-hf to refresh raw data and build chunks first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    source = "hf" if args.from_hf else "artifact"
    LOGGER.info(
        "dense_index source=%s",
        source,
        extra={"stage": "dense_index", "dense_source": source},
    )

    indexer = DenseIndexer(settings=settings)
    result = indexer.build_from_jsonl_streaming(
        settings.index_chunks_path,
        lines_per_batch=settings.dense_read_batch_lines,
        max_index_rows=settings.max_index_rows,
    )
    LOGGER.info(
        "dense_index complete: vectors=%s dim=%s schema=%s",
        result.vector_count,
        result.vector_size,
        INDEX_CHUNK_SCHEMA_VERSION,
        extra={"stage": "dense_index"},
    )


if __name__ == "__main__":
    main()
