from __future__ import annotations

import uuid

import pytest

from src.config.settings import Settings
from src.ingestion.nq_loader import load_nq_passages

_PASSAGE_ID_NAMESPACE = uuid.UUID("018f0884-e1c7-7e4f-a9c2-0f6b2a8d4e10")


def _expected_passage_id(settings: Settings, row_ordinal: int, candidate_index: int) -> str:
    key = (
        f"{settings.dataset_name}\x1f{settings.dataset_split}\x1f{row_ordinal}\x1f{candidate_index}"
    )
    return str(uuid.uuid5(_PASSAGE_ID_NAMESPACE, key))


def test_expand_candidates_cardinality_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "question": "  What is X?  ",
            "title": " Article ",
            "candidates": ["  first  ", "second"],
            "passage_types": ["intro", "body"],
            "long_answers": [7, 8],
            "document_url": " https://example.com/a ",
        }
    ]

    monkeypatch.setattr("src.ingestion.nq_loader._load_dataset", lambda *_a, **_k: rows)
    settings = Settings()
    passages = load_nq_passages(settings)

    assert len(passages) == 2
    assert passages[0].text == "first"
    assert passages[0].question == "What is X?"
    assert passages[0].title == "Article"
    assert passages[0].source == "Article"
    assert passages[0].passage_type == "intro"
    assert passages[0].document_url == "https://example.com/a"
    assert passages[0].long_answers == ["7", "8"]

    assert passages[1].text == "second"
    assert passages[1].passage_type == "body"


def test_expand_candidates_skips_blank_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"title": "T", "question": "Q", "candidates": ["  ", "\t", "ok"]}]
    monkeypatch.setattr("src.ingestion.nq_loader._load_dataset", lambda *_a, **_k: rows)

    passages = load_nq_passages(Settings())
    assert len(passages) == 1
    assert passages[0].text == "ok"


def test_passage_types_shorter_than_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"title": "T", "question": "Q", "candidates": ["a", "b"], "passage_types": ["only"]}]
    monkeypatch.setattr("src.ingestion.nq_loader._load_dataset", lambda *_a, **_k: rows)

    passages = load_nq_passages(Settings())
    assert passages[0].passage_type == "only"
    assert passages[1].passage_type is None


def test_document_url_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"title": "T", "question": "Q", "candidates": ["x"]}]
    monkeypatch.setattr("src.ingestion.nq_loader._load_dataset", lambda *_a, **_k: rows)

    passages = load_nq_passages(Settings())
    assert passages[0].document_url is None


def test_synthetic_passage_ids_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"title": "A", "question": "Q1", "candidates": ["p0"]},
        {"title": "B", "question": "Q2", "candidates": ["p1", "p2"]},
    ]
    monkeypatch.setattr("src.ingestion.nq_loader._load_dataset", lambda *_a, **_k: rows)
    settings = Settings()

    first = load_nq_passages(settings)
    second = load_nq_passages(settings)

    assert [p.passage_id for p in first] == [p.passage_id for p in second]
    assert first[0].passage_id == _expected_passage_id(settings, 0, 0)
    assert first[1].passage_id == _expected_passage_id(settings, 1, 0)
    assert first[2].passage_id == _expected_passage_id(settings, 1, 1)


def test_max_passages_caps_expanded_passages(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"title": "A", "question": "Q", "candidates": ["1", "2"]},
        {"title": "B", "question": "Q", "candidates": ["3", "4"]},
    ]
    monkeypatch.setattr("src.ingestion.nq_loader._load_dataset", lambda *_a, **_k: rows)

    passages = load_nq_passages(Settings(max_passages=3))
    assert [p.text for p in passages] == ["1", "2", "3"]
