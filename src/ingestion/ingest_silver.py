"""Stream Hugging Face rows into silver ``passages.jsonl`` + ingest manifest (Milestone 2)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.config.settings import Settings
from src.ingestion.models import SILVER_SCHEMA_VERSION, IngestManifest
from src.ingestion.nq_loader import iter_nq_passages
from src.observability.logging_setup import get_stage_logger

LOGGER = get_stage_logger(__name__)


def count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a UTF-8 JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_ingest_manifest(path: Path) -> IngestManifest | None:
    """Return parsed ingest manifest, or ``None`` if missing or invalid."""

    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return IngestManifest.model_validate_json(handle.read())
    except (json.JSONDecodeError, ValueError, OSError):
        return None


def _ingest_inputs_match_manifest(settings: Settings, manifest: IngestManifest) -> bool:
    return (
        manifest.schema_version == SILVER_SCHEMA_VERSION
        and manifest.dataset_name == settings.dataset_name
        and manifest.dataset_split == settings.dataset_split
        and manifest.max_passages == settings.max_passages
    )


def should_skip_ingest(settings: Settings, *, force: bool = False) -> bool:
    """Return True when existing silver matches manifest and ``RAG_FORCE_INGEST`` is off."""

    if force or Settings.force_ingest_from_env():
        return False
    passages = settings.passages_path
    manifest_path = settings.ingest_manifest_path
    if not passages.is_file() or not manifest_path.is_file():
        return False
    manifest = load_ingest_manifest(manifest_path)
    if manifest is None:
        return False
    if not _ingest_inputs_match_manifest(settings, manifest):
        return False
    try:
        lines = count_jsonl_lines(passages)
    except OSError:
        return False
    return lines == manifest.line_count


def _replace_atomic(tmp: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)


def run_ingest(settings: Settings, *, force: bool = False) -> tuple[IngestManifest, bool]:
    """Stream NQ rows into ``passages.jsonl`` and write ``ingest_manifest.json`` atomically.

    Returns ``(manifest, skipped)`` where ``skipped`` is True when existing silver was reused.
    """

    if should_skip_ingest(settings, force=force):
        manifest = load_ingest_manifest(settings.ingest_manifest_path)
        assert manifest is not None  # guarded by should_skip_ingest
        LOGGER.info(
            "skip ingest: silver matches ingest_manifest (%s lines)",
            manifest.line_count,
            extra={"stage": "ingest"},
        )
        return manifest, True

    tmp_jsonl = settings.passages_path.with_suffix(settings.passages_path.suffix + ".tmp")
    tmp_manifest = settings.ingest_manifest_path.with_suffix(
        settings.ingest_manifest_path.suffix + ".tmp"
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    line_count = 0
    with tmp_jsonl.open("w", encoding="utf-8") as handle:
        for passage in iter_nq_passages(settings):
            handle.write(passage.model_dump_json())
            handle.write("\n")
            line_count += 1
        handle.flush()
        os.fsync(handle.fileno())

    _replace_atomic(tmp_jsonl, settings.passages_path)

    manifest = IngestManifest(
        schema_version=SILVER_SCHEMA_VERSION,
        dataset_name=settings.dataset_name,
        dataset_split=settings.dataset_split,
        max_passages=settings.max_passages,
        line_count=line_count,
        created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    with tmp_manifest.open("w", encoding="utf-8") as handle:
        handle.write(manifest.model_dump_json(indent=2, exclude_none=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _replace_atomic(tmp_manifest, settings.ingest_manifest_path)

    LOGGER.info("wrote silver (%s lines)", line_count, extra={"stage": "persist"})
    return manifest, False
