from __future__ import annotations

import pytest

from src.config.settings import Settings
from src.ingestion.nq_loader import load_nq_passages


def test_load_nq_passages_preserves_passage_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"id": "p1", "text": "  First passage.  ", "title": "Doc 1"},
        {"id": "p2", "text": "Second passage.", "title": "Doc 2"},
    ]

    def fake_load_dataset(name: str, split: str):
        assert name == "sentence-transformers/NQ-retrieval"
        assert split == "train"
        return rows

    monkeypatch.setattr("src.ingestion.nq_loader._load_dataset", fake_load_dataset)
    settings = Settings(max_passages=None)

    passages = load_nq_passages(settings)

    assert [p.passage_id for p in passages] == ["p1", "p2"]
    assert [p.text for p in passages] == ["First passage.", "Second passage."]


def test_load_nq_passages_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"id": "p1", "text": "A", "title": "T1"},
        {"id": "p2", "text": "B", "title": "T2"},
    ]

    monkeypatch.setattr("src.ingestion.nq_loader._load_dataset", lambda *_args, **_kwargs: rows)
    settings = Settings()

    first = load_nq_passages(settings)
    second = load_nq_passages(settings)
    assert [p.model_dump() for p in first] == [p.model_dump() for p in second]


def test_load_nq_passages_raises_when_id_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"text": "passage without id", "title": "Doc"}]
    monkeypatch.setattr("src.ingestion.nq_loader._load_dataset", lambda *_args, **_kwargs: rows)

    with pytest.raises(ValueError, match="Missing passage id field"):
        load_nq_passages(Settings())


def test_load_nq_passages_rejects_non_mapping_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.ingestion.nq_loader._load_dataset", lambda *_a, **_k: ["not-a-dict"])

    with pytest.raises(TypeError, match="Dataset row must be a mapping"):
        load_nq_passages(Settings())
