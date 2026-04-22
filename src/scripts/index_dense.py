"""CLI: chunked dense embed + Qdrant upsert from silver JSONL (artifact-first)."""

from __future__ import annotations

import argparse
import logging
import sys

from src.config.settings import Settings
from src.ingestion.ingest_silver import run_ingest
from src.ingestion.models import SILVER_SCHEMA_VERSION
from src.observability.logging_setup import get_stage_logger, setup_logging
from src.retrieval.dense_index import DenseIndexer

LOGGER = get_stage_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-hf",
        action="store_true",
        help="Run streaming ingest from Hugging Face into silver before dense indexing.",
    )
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    settings = Settings.from_env()

    if args.from_hf:
        run_ingest(settings, force=True)

    if not settings.passages_path.is_file() or not settings.ingest_manifest_path.is_file():
        print(
            "Silver artifact missing. Run: python -m src.scripts.ingest_passages\n"
            "Or pass --from-hf to ingest from Hugging Face first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    source = "hf" if args.from_hf else "artifact"
    LOGGER.info("dense_index source=%s", source, extra={"stage": "dense_index", "dense_source": source})

    indexer = DenseIndexer(settings=settings)
    result = indexer.build_from_jsonl_streaming(
        settings.passages_path,
        lines_per_batch=settings.dense_read_batch_lines,
        max_passages=settings.max_passages,
    )
    LOGGER.info(
        "dense_index complete: vectors=%s dim=%s schema=%s",
        result.vector_count,
        result.vector_size,
        SILVER_SCHEMA_VERSION,
        extra={"stage": "dense_index"},
    )


if __name__ == "__main__":
    main()
