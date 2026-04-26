from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.chunking_strategy_eval import (
    build_chunks,
    is_relevant_chunk,
    load_raw_rows,
    profile_dataset,
    profile_strategy,
    run_chunking_evaluation,
    write_report_markdown,
)


def _write_raw(path: Path) -> None:
    rows = [
        {
            "title": "Example Page",
            "question": "Who is listed?",
            "candidates": [
                "Example Page\nCreated by Alice\nDirected by Bob",
                "Example Page",
                "Created by Alice",
                "Directed by Bob",
                "Famous people include:",
                "Ada Lovelace\nGrace Hopper",
            ],
            "passage_types": ["table", "table", "table", "table", "text", "list"],
            "long_answers": [5],
            "document_url": "https://example.com",
        },
        {
            "title": "Film",
            "question": "Where was it filmed?",
            "candidates": [
                "External links",
                "Filming happened in Vancouver and finished in October 2009.",
            ],
            "passage_types": ["table", "text"],
            "long_answers": [1],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_dataset_profile_captures_short_candidates_and_gold_shape(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    _write_raw(raw_path)

    rows = load_raw_rows(raw_path)
    profile = profile_dataset(rows)

    assert profile.sample_rows == 2
    assert profile.total_candidates == 8
    assert profile.short_candidate_counts["<=10"] >= 6
    assert profile.gold_long_answer_count == 2
    assert profile.gold_token_median >= 6


def test_parent_dedup_suppresses_nested_child_candidates(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    _write_raw(raw_path)
    rows = load_raw_rows(raw_path)

    raw_chunks = build_chunks(rows, "raw_candidate")
    parent_chunks = build_chunks(rows, "parent_dedup")

    assert len(parent_chunks) < len(raw_chunks)
    assert not any(chunk.text == "External links" for chunk in parent_chunks)
    assert any(is_relevant_chunk(chunk, rows[chunk.row_ordinal]) for chunk in parent_chunks)


def test_strategy_profile_and_report_writer(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    report_path = tmp_path / "report.md"
    _write_raw(raw_path)
    rows = load_raw_rows(raw_path)
    chunks = build_chunks(rows, "min_context")

    profile = profile_strategy(rows, chunks, "min_context")
    assert profile.chunks > 0
    assert profile.relevant_chunks > 0

    report = run_chunking_evaluation(raw_path, max_rows=2, max_queries=2, top_k=2)
    write_report_markdown(report_path, report)

    body = report_path.read_text(encoding="utf-8")
    assert "Chunking Strategy Evaluation" in body
    assert "`parent_dedup`" in body
