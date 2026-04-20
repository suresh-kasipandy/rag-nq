from __future__ import annotations

import sys
import types

from src.ingestion.models import Passage
from src.retrieval.sparse_index import SparseIndexer


class FakeBM25:
    def __init__(self, corpus):
        self.corpus = corpus


def test_sparse_index_save_load_roundtrip(tmp_path) -> None:
    sys.modules["rank_bm25"] = types.SimpleNamespace(BM25Okapi=FakeBM25)

    passages = [
        Passage(passage_id="p1", text="heart treatment center", source="doc1"),
        Passage(passage_id="p2", text="oncology outpatient schedule", source="doc2"),
    ]
    try:
        index, result = SparseIndexer.build(passages)
        assert result.document_count == 2

        path = tmp_path / "bm25.pkl"
        index.save(path)
        loaded = SparseIndexer.load(path)

        assert loaded.document_count == 2
    finally:
        del sys.modules["rank_bm25"]
