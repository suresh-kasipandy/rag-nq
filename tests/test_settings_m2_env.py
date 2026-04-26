from __future__ import annotations

import pytest

from src.config.settings import Settings


def test_from_env_max_passages_and_embedding_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MAX_PASSAGES", "100")
    monkeypatch.setenv("RAG_MAX_RAW_ROWS", "101")
    monkeypatch.setenv("RAG_MAX_CHUNK_ROWS", "102")
    monkeypatch.setenv("RAG_MAX_INDEX_ROWS", "103")
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
    monkeypatch.setenv("RAG_RETRIEVE_K", "75")
    monkeypatch.setenv("RAG_RERANK_K", "12")
    monkeypatch.setenv("RAG_RERANK_ENABLED", "true")
    monkeypatch.setenv("RAG_RERANK_MODEL_NAME", "cross-encoder/test")
    monkeypatch.setenv("RAG_RERANK_CONTEXT_TOKEN_BUDGET", "222")
    monkeypatch.setenv("RAG_RETRIEVAL_DEDUPE_ENABLED", "false")
    s = Settings.from_env()
    assert s.max_passages == 100
    assert s.max_raw_rows == 101
    assert s.max_chunk_rows == 102
    assert s.max_index_rows == 103
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
    assert s.retrieve_k == 75
    assert s.rerank_k == 12
    assert s.rerank_enabled is True
    assert s.rerank_model_name == "cross-encoder/test"
    assert s.rerank_context_token_budget == 222
    assert s.retrieval_dedupe_enabled is False


def test_from_env_max_passages_aliases_raw_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MAX_PASSAGES", "100")

    s = Settings.from_env()

    assert s.max_passages == 100
    assert s.max_raw_rows == 100
