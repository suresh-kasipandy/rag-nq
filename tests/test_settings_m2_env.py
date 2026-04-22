from __future__ import annotations

import pytest

from src.config.settings import Settings


def test_from_env_max_passages_and_embedding_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MAX_PASSAGES", "100")
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "32")
    monkeypatch.setenv("RAG_DENSE_READ_BATCH_LINES", "128")
    monkeypatch.setenv("RAG_SPARSE_UPSERT_BATCH_SIZE", "64")
    monkeypatch.setenv("RAG_QDRANT_SPARSE_VECTOR_NAME", "sparse_x")
    monkeypatch.setenv("RAG_BM25_K1", "1.2")
    monkeypatch.setenv("RAG_BM25_B", "0.8")
    monkeypatch.setenv("RAG_BM25_EPSILON", "0.3")
    s = Settings.from_env()
    assert s.max_passages == 100
    assert s.embedding_batch_size == 32
    assert s.dense_read_batch_lines == 128
    assert s.sparse_upsert_batch_size == 64
    assert s.qdrant_sparse_vector_name == "sparse_x"
    assert s.bm25_k1 == pytest.approx(1.2)
    assert s.bm25_b == pytest.approx(0.8)
    assert s.bm25_epsilon == pytest.approx(0.3)
