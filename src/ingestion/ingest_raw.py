"""Download Hugging Face rows into local raw dataset artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from src.config.settings import Settings
from src.ingestion.models import RAW_DATASET_SCHEMA_VERSION, RawDatasetManifest
from src.observability.logging_setup import get_stage_logger, setup_logging
from src.observability.progress import ProgressTicker

LOGGER = get_stage_logger(__name__)


def count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a UTF-8 JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_raw_manifest(path: Path) -> RawDatasetManifest | None:
    """Return parsed raw dataset manifest, or ``None`` if missing or invalid."""

    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return RawDatasetManifest.model_validate_json(handle.read())
    except (json.JSONDecodeError, ValueError, OSError):
        return None


def raw_inputs_match_manifest(settings: Settings, manifest: RawDatasetManifest) -> bool:
    """Return True when raw manifest corresponds to current dataset settings."""

    return (
        manifest.schema_version == RAW_DATASET_SCHEMA_VERSION
        and manifest.dataset_name == settings.dataset_name
        and manifest.dataset_split == settings.dataset_split
        and manifest.max_passages == settings.max_passages
    )


def should_skip_raw_ingest(settings: Settings, *, force: bool = False) -> bool:
    """Return True when existing raw snapshot matches manifest and force is off."""

    if force or Settings.force_raw_ingest_from_env():
        return False
    raw_path = settings.raw_dataset_path
    manifest_path = settings.raw_manifest_path
    if not raw_path.is_file() or not manifest_path.is_file():
        return False
    manifest = load_raw_manifest(manifest_path)
    if manifest is None:
        return False
    if not raw_inputs_match_manifest(settings, manifest):
        return False
    try:
        lines = count_jsonl_lines(raw_path)
    except OSError:
        return False
    return lines == manifest.row_count


def _replace_atomic(tmp: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)


def run_raw_ingest(settings: Settings, *, force: bool = False) -> tuple[RawDatasetManifest, bool]:
    """Stream HF rows into ``raw_dataset.jsonl`` and write raw manifest atomically."""

    if should_skip_raw_ingest(settings, force=force):
        manifest = load_raw_manifest(settings.raw_manifest_path)
        assert manifest is not None  # guarded by should_skip_raw_ingest
        LOGGER.info(
            "skip raw ingest: raw manifest matches (%s rows)",
            manifest.row_count,
            extra={"stage": "ingest_raw"},
        )
        return manifest, True

    from src.ingestion.nq_loader import _iter_hf_rows  # local import avoids eager HF dependency

    tmp_raw = settings.raw_dataset_path.with_suffix(settings.raw_dataset_path.suffix + ".tmp")
    tmp_manifest = settings.raw_manifest_path.with_suffix(
        settings.raw_manifest_path.suffix + ".tmp"
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    ticker = ProgressTicker(
        logger=LOGGER,
        stage="ingest_raw",
        label="rows",
        total=settings.max_passages,
        every_items=settings.progress_log_every_records,
        every_seconds=settings.progress_log_every_seconds,
    )
    ticker.start(
        dataset=settings.dataset_name,
        split=settings.dataset_split,
        output=settings.raw_dataset_path,
        streaming=settings.dataset_streaming,
        max_passages=settings.max_passages,
    )
    row_count = 0
    with tmp_raw.open("w", encoding="utf-8") as handle:
        for row in _iter_hf_rows(settings):
            handle.write(json.dumps(row))
            handle.write("\n")
            row_count += 1
            ticker.tick(row_count, output=settings.raw_dataset_path)
            if settings.max_passages is not None and row_count >= settings.max_passages:
                break
        handle.flush()
        os.fsync(handle.fileno())

    _replace_atomic(tmp_raw, settings.raw_dataset_path)

    manifest = RawDatasetManifest(
        schema_version=RAW_DATASET_SCHEMA_VERSION,
        dataset_name=settings.dataset_name,
        dataset_split=settings.dataset_split,
        max_passages=settings.max_passages,
        row_count=row_count,
        created_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
    )
    with tmp_manifest.open("w", encoding="utf-8") as handle:
        handle.write(manifest.model_dump_json(indent=2, exclude_none=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _replace_atomic(tmp_manifest, settings.raw_manifest_path)

    ticker.finish(row_count, output=settings.raw_dataset_path, manifest=settings.raw_manifest_path)
    return manifest, False


def main() -> None:
    """CLI entrypoint for raw dataset ingest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download raw dataset even when raw manifest matches.",
    )
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    settings = Settings.from_env()
    manifest, skipped = run_raw_ingest(
        settings,
        force=args.force or Settings.force_raw_ingest_from_env(),
    )
    LOGGER.info(
        "raw ingest complete: rows=%s skipped=%s",
        manifest.row_count,
        skipped,
        extra={"stage": "done"},
    )


if __name__ == "__main__":
    main()
