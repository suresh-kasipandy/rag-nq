from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.settings import Settings
from src.ingestion.ingest_silver import (
    count_jsonl_lines,
    load_ingest_manifest,
    run_ingest,
    should_skip_ingest,
)
from src.ingestion.models import SILVER_SCHEMA_VERSION, IngestManifest, Passage
from src.ingestion import ingest_silver


def test_count_jsonl_lines_skips_blanks(tmp_path: Path) -> None:
    path = tmp_path / "f.jsonl"
    path.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    assert count_jsonl_lines(path) == 2


def test_should_skip_ingest_when_manifest_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    out.mkdir()
    settings = Settings(
        output_dir=out,
        dataset_name="ds",
        dataset_split="train",
        max_passages=2,
    )
    silver = settings.passages_path
    silver.write_text(
        Passage(passage_id="a", text="x").model_dump_json()
        + "\n"
        + Passage(passage_id="b", text="y").model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    manifest = IngestManifest(
        schema_version=SILVER_SCHEMA_VERSION,
        dataset_name="ds",
        dataset_split="train",
        max_passages=2,
        line_count=2,
        created_at_utc="2020-01-01T00:00:00+00:00",
    )
    settings.ingest_manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")

    monkeypatch.setattr(ingest_silver, "iter_nq_passages", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("HF should not run")))
    assert should_skip_ingest(settings, force=False) is True


def test_run_ingest_streams_without_full_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    out.mkdir()
    settings = Settings(output_dir=out, dataset_name="ds", dataset_split="train", max_passages=3)

    def fake_iter(_settings: Settings):
        for i in range(3):
            yield Passage(passage_id=f"id{i}", text=f"t{i}")

    monkeypatch.setattr(ingest_silver, "iter_nq_passages", fake_iter)

    manifest, skipped = run_ingest(settings, force=True)
    assert skipped is False
    assert manifest.line_count == 3
    assert count_jsonl_lines(settings.passages_path) == 3
    loaded = load_ingest_manifest(settings.ingest_manifest_path)
    assert loaded is not None
    assert loaded.line_count == 3
    assert json.loads(settings.passages_path.read_text(encoding="utf-8").splitlines()[0])["passage_id"] == "id0"
