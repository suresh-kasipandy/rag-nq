from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.settings import Settings
from src.ingestion.ingest_raw import (
    load_raw_manifest,
    run_raw_ingest,
    should_skip_raw_ingest,
)
from src.ingestion.ingest_silver import (
    count_jsonl_lines,
    load_ingest_manifest,
    run_ingest,
    should_skip_ingest,
)
from src.ingestion.models import (
    RAW_DATASET_SCHEMA_VERSION,
    SILVER_SCHEMA_VERSION,
    IngestManifest,
    Passage,
    RawDatasetManifest,
)
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
    settings.raw_dataset_path.write_text('{"id":"r1"}\n{"id":"r2"}\n', encoding="utf-8")
    raw_manifest = RawDatasetManifest(
        schema_version=RAW_DATASET_SCHEMA_VERSION,
        dataset_name="ds",
        dataset_split="train",
        max_passages=2,
        row_count=2,
        created_at_utc="2020-01-01T00:00:00+00:00",
    )
    settings.raw_manifest_path.write_text(raw_manifest.model_dump_json(), encoding="utf-8")
    manifest = IngestManifest(
        schema_version=SILVER_SCHEMA_VERSION,
        dataset_name="ds",
        dataset_split="train",
        max_passages=2,
        line_count=2,
        raw_schema_version=RAW_DATASET_SCHEMA_VERSION,
        raw_row_count=2,
        created_at_utc="2020-01-01T00:00:00+00:00",
    )
    settings.ingest_manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")

    monkeypatch.setattr(
        ingest_silver,
        "run_raw_ingest",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("HF should not run")),
    )
    assert should_skip_ingest(settings, force=False) is True


def test_should_skip_raw_ingest_when_manifest_matches(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    settings = Settings(output_dir=out, dataset_name="ds", dataset_split="train", max_passages=2)
    settings.raw_dataset_path.write_text('{"id":"1"}\n{"id":"2"}\n', encoding="utf-8")
    settings.raw_manifest_path.write_text(
        RawDatasetManifest(
            schema_version=RAW_DATASET_SCHEMA_VERSION,
            dataset_name="ds",
            dataset_split="train",
            max_passages=2,
            row_count=2,
            created_at_utc="2020-01-01T00:00:00+00:00",
        ).model_dump_json(),
        encoding="utf-8",
    )
    assert should_skip_raw_ingest(settings, force=False) is True


def test_run_raw_ingest_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    out.mkdir()
    settings = Settings(output_dir=out, dataset_name="ds", dataset_split="train", max_passages=None)

    def fake_rows(_settings: Settings):
        yield {"id": "x", "candidates": ["A"]}
        yield {"id": "y", "candidates": ["B"]}

    monkeypatch.setattr("src.ingestion.nq_loader._iter_hf_rows", fake_rows)
    manifest, skipped = run_raw_ingest(settings, force=True)
    assert skipped is False
    assert manifest.row_count == 2
    assert count_jsonl_lines(settings.raw_dataset_path) == 2
    loaded = load_raw_manifest(settings.raw_manifest_path)
    assert loaded is not None
    assert loaded.row_count == 2


def test_run_ingest_streams_from_raw_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    out.mkdir()
    settings = Settings(output_dir=out, dataset_name="ds", dataset_split="train", max_passages=3)

    def fake_passages(_settings: Settings):
        for i in range(3):
            yield Passage(passage_id=f"id{i}", text=f"t{i}")

    def fake_run_raw(_settings: Settings, *, force: bool = False, from_hf: bool = False):
        del force, from_hf
        return (
            RawDatasetManifest(
                schema_version=RAW_DATASET_SCHEMA_VERSION,
                dataset_name="ds",
                dataset_split="train",
                max_passages=3,
                row_count=2,
                created_at_utc="2020-01-01T00:00:00+00:00",
            ),
            False,
        )

    monkeypatch.setattr(ingest_silver, "run_raw_ingest", fake_run_raw)
    monkeypatch.setattr(ingest_silver, "iter_nq_passages_from_raw_artifact", fake_passages)

    manifest, skipped = run_ingest(settings, force=True)
    assert skipped is False
    assert manifest.line_count == 3
    assert count_jsonl_lines(settings.passages_path) == 3
    loaded = load_ingest_manifest(settings.ingest_manifest_path)
    assert loaded is not None
    assert loaded.line_count == 3
    assert json.loads(settings.passages_path.read_text(encoding="utf-8").splitlines()[0])["passage_id"] == "id0"
