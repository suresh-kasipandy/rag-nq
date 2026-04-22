"""CLI: stream Hugging Face NQ-retrieval into silver ``passages.jsonl`` + ingest manifest."""

from __future__ import annotations

import logging

from src.config.settings import Settings
from src.ingestion.ingest_silver import run_ingest
from src.observability.logging_setup import get_stage_logger, setup_logging

LOGGER = get_stage_logger(__name__)


def main() -> None:
    setup_logging(level=logging.INFO)
    settings = Settings.from_env()
    manifest, skipped = run_ingest(settings, force=Settings.force_ingest_from_env())
    LOGGER.info(
        "ingest complete: lines=%s skipped=%s",
        manifest.line_count,
        skipped,
        extra={"stage": "done"},
    )


if __name__ == "__main__":
    main()
