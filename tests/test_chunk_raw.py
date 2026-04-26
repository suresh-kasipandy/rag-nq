from __future__ import annotations

import json

from src.config.settings import Settings
from src.ingestion.chunk_raw import (
    annotate_row_candidates,
    iter_index_chunks_from_raw_artifact,
    run_chunk_ingest,
)
from src.ingestion.models import (
    CHUNK_MANIFEST_SCHEMA_VERSION,
    INDEX_CHUNK_SCHEMA_VERSION,
    RAW_DATASET_SCHEMA_VERSION,
    RawDatasetManifest,
)


def _write_raw_fixture(settings: Settings, rows: list[dict[str, object]]) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    with settings.raw_dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
    manifest = RawDatasetManifest(
        schema_version=RAW_DATASET_SCHEMA_VERSION,
        dataset_name=settings.dataset_name,
        dataset_split=settings.dataset_split,
        max_passages=settings.max_passages,
        row_count=len(rows),
        created_at_utc="2026-01-01T00:00:00+00:00",
    )
    settings.raw_manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")


def test_candidate_annotation_tags_boilerplate_and_duplicate_child() -> None:
    settings = Settings(chunk_min_tokens_hard=20)
    row = {
        "candidates": [
            "References",
            "Mercury is the smallest planet in the Solar System and orbits closest to the Sun.",
            "Mercury",
        ],
        "passage_types": ["paragraph", "paragraph", "paragraph"],
    }

    annotations = annotate_row_candidates(row, row_ordinal=0, settings=settings)

    assert [a.role for a in annotations] == [
        "boilerplate",
        "short_fact_candidate",
        "duplicate_child",
    ]
    assert annotations[2].parent_candidate_idx == 1


def test_iter_index_chunks_merges_small_compatible_candidates(tmp_path) -> None:
    settings = Settings(
        output_dir=tmp_path,
        chunk_min_tokens_soft=12,
        chunk_min_tokens_hard=5,
        chunk_max_tokens=80,
    )
    _write_raw_fixture(
        settings,
        [
            {
                "title": "Planet",
                "question": "Which planet is smallest?",
                "candidates": [
                    "Mercury is the smallest planet in the Solar System.",
                    "It has no natural moons and orbits closest to the Sun.",
                ],
                "passage_types": ["paragraph", "paragraph"],
                "long_answers": ["Mercury is the smallest planet in the Solar System."],
            }
        ],
    )

    chunks = list(iter_index_chunks_from_raw_artifact(settings))

    assert len(chunks) == 1
    assert chunks[0].chunk_id
    assert chunks[0].text.startswith("Mercury is the smallest")
    assert chunks[0].context_text == chunks[0].text
    assert chunks[0].chunk_kind == "minimum_context_span"
    assert chunks[0].passage_types == ["paragraph", "paragraph"]


def test_context_text_expands_structurally_dependent_chunk(tmp_path) -> None:
    settings = Settings(
        output_dir=tmp_path,
        chunk_min_tokens_soft=30,
        chunk_min_tokens_hard=5,
        chunk_max_tokens=80,
    )
    _write_raw_fixture(
        settings,
        [
            {
                "title": "Planets",
                "candidates": [
                    (
                        "Notable planets include Mercury Venus Earth Mars Jupiter Saturn "
                        "Uranus Neptune and several dwarf planets described by astronomy "
                        "texts for general readers and students in introductory reference "
                        "material about the Solar System."
                    ),
                    "Mercury is the smallest planet.",
                    "Venus is the second planet from the Sun.",
                ],
                "passage_types": ["list", "list", "list"],
            }
        ],
    )

    chunks = list(iter_index_chunks_from_raw_artifact(settings))

    assert len(chunks) == 2
    assert chunks[1].text == (
        "Mercury is the smallest planet. Venus is the second planet from the Sun."
    )
    assert chunks[1].context_text.startswith("Notable planets include Mercury")
    assert chunks[1].context_text.endswith(chunks[1].text)


def test_run_chunk_ingest_writes_manifest_and_skips_when_fresh(tmp_path) -> None:
    settings = Settings(
        output_dir=tmp_path,
        chunk_min_tokens_soft=12,
        chunk_min_tokens_hard=5,
        chunk_max_tokens=80,
    )
    _write_raw_fixture(
        settings,
        [
            {
                "title": "Planet",
                "candidates": [
                    "Mercury is the smallest planet in the Solar System.",
                    "It has no natural moons and orbits closest to the Sun.",
                ],
                "passage_types": ["paragraph", "paragraph"],
            }
        ],
    )

    manifest, skipped = run_chunk_ingest(settings)
    second_manifest, second_skipped = run_chunk_ingest(settings)

    assert not skipped
    assert second_skipped
    assert second_manifest.line_count == manifest.line_count == 1
    assert manifest.schema_version == CHUNK_MANIFEST_SCHEMA_VERSION
    assert manifest.chunk_schema_version == INDEX_CHUNK_SCHEMA_VERSION
    assert settings.index_chunks_path.is_file()
    assert settings.chunk_manifest_path.is_file()
