from __future__ import annotations

from pathlib import Path

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
    monkeypatch.setenv("RAG_GENERATION_PROVIDER", "openai")
    monkeypatch.setenv("RAG_GENERATION_MODEL_NAME", "generator/test")
    monkeypatch.setenv("RAG_GENERATION_TEMPERATURE", "0.2")
    monkeypatch.setenv("RAG_GENERATION_MAX_TOKENS", "123")
    monkeypatch.setenv("RAG_GENERATION_TIMEOUT_SECONDS", "9.5")
    monkeypatch.setenv("RAG_GENERATION_CONTEXT_TOKEN_BUDGET", "777")
    monkeypatch.setenv("RAG_GENERATION_MIN_CITATIONS", "2")
    monkeypatch.setenv("RAG_GENERATION_API_URL", "http://generator.local")
    monkeypatch.setenv("RAG_GENERATION_API_KEY_ENV", "GEN_KEY")
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
    assert s.generation_provider == "openai"
    assert s.generation_model_name == "generator/test"
    assert s.generation_temperature == pytest.approx(0.2)
    assert s.generation_max_tokens == 123
    assert s.generation_timeout_seconds == pytest.approx(9.5)
    assert s.generation_context_token_budget == 777
    assert s.generation_min_citations == 2
    assert s.generation_api_url == "http://generator.local"
    assert s.generation_api_key_env == "GEN_KEY"


def test_from_env_max_passages_aliases_raw_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MAX_PASSAGES", "100")

    s = Settings.from_env()

    assert s.max_passages == 100
    assert s.max_raw_rows == 100


def test_from_env_loads_dotenv_without_overriding_shell_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "RAG_QDRANT_COLLECTION=from_dotenv",
                "RAG_MAX_INDEX_ROWS=123",
                "RAG_RERANK_ENABLED=1",
                "RAG_GENERATION_PROVIDER=heuristic",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_QDRANT_COLLECTION", "from_shell")

    s = Settings.from_env()

    assert s.qdrant_collection == "from_shell"
    assert s.max_index_rows == 123
    assert s.rerank_enabled is True
    assert s.generation_provider == "heuristic"
