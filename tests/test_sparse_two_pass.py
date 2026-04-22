from __future__ import annotations

import sys
import types
from pathlib import Path

from src.ingestion.models import Passage
from src.ingestion.passage_store import PassageStore
from src.retrieval.sparse_index import (
    SparseIndexer,
    compute_sparse_corpus_stats_from_passages,
    compute_sparse_corpus_stats_pass1,
)


class PickleableFakeBM25Okapi:
    """Module-level stand-in for ``rank_bm25.BM25Okapi`` (pickle-safe in tests)."""

    __slots__ = ("corpus",)

    def __init__(self, corpus: list[list[str]]) -> None:
        self.corpus = corpus


def test_sparse_pass1_stats_match_reference(tmp_path: Path) -> None:
    passages = [
        Passage(passage_id="1", text="aa bb", source=None),
        Passage(passage_id="2", text="bb cc", source=None),
    ]
    path = tmp_path / "p.jsonl"
    PassageStore.write_jsonl(passages, path)

    from_file = compute_sparse_corpus_stats_pass1(path)
    from_objs = compute_sparse_corpus_stats_from_passages(passages)

    assert from_file.document_count == from_objs.document_count == 2
    assert from_file.total_tokens == from_objs.total_tokens
    assert from_file.term_document_frequency == from_objs.term_document_frequency


def test_sparse_two_pass_build_roundtrip(tmp_path: Path) -> None:
    sys.modules["rank_bm25"] = types.SimpleNamespace(BM25Okapi=PickleableFakeBM25Okapi)
    passages = [
        Passage(passage_id="p1", text="heart treatment center", source="doc1"),
        Passage(passage_id="p2", text="oncology outpatient schedule", source="doc2"),
    ]
    path = tmp_path / "silver.jsonl"
    PassageStore.write_jsonl(passages, path)

    try:
        index, result, stats = SparseIndexer.build_from_jsonl_two_pass(path)
        assert result.document_count == 2
        assert stats.document_count == 2

        out = tmp_path / "bm25.pkl"
        index.save(out)
        loaded = SparseIndexer.load(out)
        assert loaded.document_count == 2
    finally:
        del sys.modules["rank_bm25"]
