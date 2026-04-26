"""CLI: build chunked index corpus from raw NQ rows."""

from __future__ import annotations

import argparse
import logging

from src.config.settings import Settings
from src.ingestion.chunk_raw import run_chunk_ingest
from src.observability.logging_setup import get_stage_logger, setup_logging

LOGGER = get_stage_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild chunk artifacts even if manifests match.",
    )
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    settings = Settings.from_env()
    manifest, skipped = run_chunk_ingest(
        settings,
        force=args.force or Settings.force_ingest_from_env(),
    )
    LOGGER.info(
        "chunk ingest complete: chunks=%s skipped=%s",
        manifest.line_count,
        skipped,
        extra={"stage": "done"},
    )


if __name__ == "__main__":
    main()
