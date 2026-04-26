from __future__ import annotations

import pytest

from src.config.settings import Settings


def test_from_env_max_passages_and_embedding_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MAX_PASSAGES", "100")
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "32")
    monkeypatch.setenv("RAG_DENSE_READ_BATCH_LINES", "128")
    monkeypatch.setenv("RAG_SPARSE_UPSERT_BATCH_SIZE", "64")
    monkeypatch.setenv("RAG_SPARSE_WORKERS", "3")
    monkeypatch.setenv("RAG_SPARSE_WRITE_CONCURRENCY", "2")
    monkeypatch.setenv("RAG_SPARSE_CHECKPOINT_FILE", "checkpoint_sparse.json")
    monkeypatch.setenv("RAG_QDRANT_SPARSE_VECTOR_NAME", "sparse_x")
    monkeypatch.setenv("RAG_QDRANT_RETRIEVAL_TIMEOUT_SECONDS", "22.5")
    monkeypatch.setenv("RAG_BM25_K1", "1.2")
    monkeypatch.setenv("RAG_BM25_B", "0.8")
    monkeypatch.setenv("RAG_BM25_EPSILON", "0.3")
    monkeypatch.setenv("RAG_PROGRESS_LOG_EVERY_RECORDS", "5000")
    monkeypatch.setenv("RAG_PROGRESS_LOG_EVERY_BATCHES", "250")
    monkeypatch.setenv("RAG_PROGRESS_LOG_EVERY_SECONDS", "30.5")
    s = Settings.from_env()
    assert s.max_passages == 100
    assert s.embedding_batch_size == 32
    assert s.dense_read_batch_lines == 128
    assert s.sparse_upsert_batch_size == 64
    assert s.sparse_workers == 3
    assert s.sparse_write_concurrency == 2
    assert s.sparse_checkpoint_file == "checkpoint_sparse.json"
    assert s.qdrant_sparse_vector_name == "sparse_x"
    assert s.qdrant_retrieval_timeout_seconds == pytest.approx(22.5)
    assert s.bm25_k1 == pytest.approx(1.2)
    assert s.bm25_b == pytest.approx(0.8)
    assert s.bm25_epsilon == pytest.approx(0.3)
    assert s.progress_log_every_records == 5000
    assert s.progress_log_every_batches == 250
    assert s.progress_log_every_seconds == pytest.approx(30.5)
