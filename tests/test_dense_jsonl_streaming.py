from __future__ import annotations

import logging

import pytest

from src.config.settings import Settings
from src.ingestion.chunk_store import count_index_records_jsonl
from src.ingestion.models import IndexChunkGroup, IndexChunkItem, IndexText, Passage
from src.retrieval import dense_index
from src.retrieval.dense_index import DenseIndexer


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.last_texts: list[str] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, *, batch_size: int, normalize_embeddings: bool):
        assert batch_size > 0
        self.last_texts = list(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.exists = False
        self.upsert_batches: list[int] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(
        self, collection_name: str, vectors_config, sparse_vectors_config=None, **kwargs
    ) -> None:
        self.exists = True

    def upsert(self, collection_name: str, points: list[dict]) -> None:
        self.upsert_batches.append(len(points))

    def count(self, collection_name: str, exact: bool = True):
        return type("CountResult", (), {"count": sum(self.upsert_batches)})()


class FlakyFakeQdrantClient(FakeQdrantClient):
    def __init__(self, fail_attempts: int) -> None:
        super().__init__()
        self._fail_attempts = fail_attempts
        self._calls = 0

    def upsert(self, collection_name: str, points: list[dict]) -> None:
        self._calls += 1
        if self._calls <= self._fail_attempts:
            raise RuntimeError("transient timeout")
        super().upsert(collection_name, points)


class FailOnSecondUpsertQdrantClient(FakeQdrantClient):
    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    def upsert(self, collection_name: str, points: list[dict]) -> None:
        self._calls += 1
        if self._calls >= 2:
            raise RuntimeError("persistent failure")
        super().upsert(collection_name, points)


class HttpNoiseQdrantClient(FakeQdrantClient):
    def upsert(self, collection_name: str, points: list[dict]) -> None:
        logging.getLogger("httpx").info("httpx-info-dense")
        logging.getLogger("httpx").warning("httpx-warning-dense")
        super().upsert(collection_name, points)


def test_dense_jsonl_streaming_batches(tmp_path) -> None:
    path = tmp_path / "silver.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for i in range(5):
            p = Passage(passage_id=f"p{i}", text=f"doc{i}")
            handle.write(p.model_dump_json())
            handle.write("\n")

    settings = Settings(
        qdrant_collection="c",
        dense_read_batch_lines=2,
        embedding_batch_size=8,
    )
    fake_client = FakeQdrantClient()
    fake_model = FakeEmbeddingModel()
    indexer = DenseIndexer(settings=settings, client=fake_client, model=fake_model)
    original_build_vector_params = dense_index._build_vector_params

    def _stub_vector_params(size: int, distance_name: str) -> dict[str, str | int]:
        return {"size": size, "distance": distance_name}

    dense_index._build_vector_params = _stub_vector_params
    try:
        result = indexer.build_from_jsonl_streaming(path, lines_per_batch=2)
        assert result.vector_count == 5
        assert fake_client.upsert_batches == [2, 2, 1]
    finally:
        dense_index._build_vector_params = original_build_vector_params


def test_dense_jsonl_streaming_accepts_index_chunks(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "index_chunks.jsonl"
    group = IndexChunkGroup(
        group_id="g1",
        source_row_ordinal=0,
        title="title",
        texts=[
            IndexText(text_idx=0, raw_idx=0, text="chunk text"),
            IndexText(text_idx=1, raw_idx=None, text="chunk text with context"),
        ],
        chunks=[
            IndexChunkItem(
                chunk_id="11111111-1111-1111-1111-111111111111",
                text_idxs=[0],
                context_idxs=[1],
                start_candidate_idx=0,
                end_candidate_idx=0,
            )
        ],
    )
    path.write_text(group.model_dump_json() + "\n", encoding="utf-8")

    settings = Settings(
        qdrant_collection="c_chunks",
        embedding_batch_size=8,
        progress_log_every_batches=1,
    )
    fake_client = FakeQdrantClient()
    fake_model = FakeEmbeddingModel()
    indexer = DenseIndexer(settings=settings, client=fake_client, model=fake_model)
    original_build_vector_params = dense_index._build_vector_params

    def _stub_vector_params(size: int, distance_name: str) -> dict[str, str | int]:
        return {"size": size, "distance": distance_name}

    dense_index._build_vector_params = _stub_vector_params
    caplog.set_level(logging.INFO)
    try:
        result = indexer.build_from_jsonl_streaming(path, lines_per_batch=2)
        assert result.vector_count == 1
        assert fake_client.upsert_batches == [1]
        messages = [record.getMessage() for record in caplog.records]
        assert any("start batches=0/1" in message for message in messages)
        assert any("progress batches=1/1" in message for message in messages)
        assert any("complete batches=1/1" in message for message in messages)
    finally:
        dense_index._build_vector_params = original_build_vector_params


def test_dense_jsonl_streaming_appends_title_to_embedding_input(tmp_path) -> None:
    path = tmp_path / "index_chunks_title.jsonl"
    group = IndexChunkGroup(
        group_id="g_title",
        source_row_ordinal=0,
        title="Xbox Article",
        texts=[IndexText(text_idx=0, raw_idx=0, text="chunk text")],
        chunks=[
            IndexChunkItem(
                chunk_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                text_idxs=[0],
                context_idxs=[0],
                start_candidate_idx=0,
                end_candidate_idx=0,
            )
        ],
    )
    path.write_text(group.model_dump_json() + "\n", encoding="utf-8")

    settings = Settings(
        qdrant_collection="c_title_embed",
        dense_read_batch_lines=1,
        embedding_batch_size=8,
    )
    fake_client = FakeQdrantClient()
    fake_model = FakeEmbeddingModel()
    indexer = DenseIndexer(settings=settings, client=fake_client, model=fake_model)
    original_build_vector_params = dense_index._build_vector_params

    def _stub_vector_params(size: int, distance_name: str) -> dict[str, str | int]:
        return {"size": size, "distance": distance_name}

    dense_index._build_vector_params = _stub_vector_params
    try:
        result = indexer.build_from_jsonl_streaming(path, lines_per_batch=1)
        assert result.vector_count == 1
        assert fake_model.last_texts == ["chunk text\nTitle: Xbox Article"]
        assert fake_client.upsert_batches == [1]
    finally:
        dense_index._build_vector_params = original_build_vector_params


def test_dense_jsonl_streaming_quiets_httpx_info_keeps_warning(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "silver_http_noise.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        p = Passage(passage_id="p0", text="doc0")
        handle.write(p.model_dump_json())
        handle.write("\n")

    settings = Settings(
        qdrant_collection="c_http_noise",
        dense_read_batch_lines=1,
        embedding_batch_size=8,
    )
    fake_client = HttpNoiseQdrantClient()
    fake_model = FakeEmbeddingModel()
    indexer = DenseIndexer(settings=settings, client=fake_client, model=fake_model)
    original_build_vector_params = dense_index._build_vector_params

    def _stub_vector_params(size: int, distance_name: str) -> dict[str, str | int]:
        return {"size": size, "distance": distance_name}

    dense_index._build_vector_params = _stub_vector_params
    caplog.set_level(logging.INFO)
    try:
        result = indexer.build_from_jsonl_streaming(path, lines_per_batch=1)
        assert result.vector_count == 1
    finally:
        dense_index._build_vector_params = original_build_vector_params

    messages = [record.getMessage() for record in caplog.records]
    assert "httpx-warning-dense" in messages
    assert "httpx-info-dense" not in messages


def test_dense_jsonl_streaming_respects_max_passages(tmp_path) -> None:
    path = tmp_path / "silver_cap.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for i in range(7):
            p = Passage(passage_id=f"p{i}", text=f"doc{i}")
            handle.write(p.model_dump_json())
            handle.write("\n")

    settings = Settings(
        qdrant_collection="c_cap",
        dense_read_batch_lines=3,
        embedding_batch_size=8,
    )
    fake_client = FakeQdrantClient()
    fake_model = FakeEmbeddingModel()
    indexer = DenseIndexer(settings=settings, client=fake_client, model=fake_model)
    original_build_vector_params = dense_index._build_vector_params

    def _stub_vector_params(size: int, distance_name: str) -> dict[str, str | int]:
        return {"size": size, "distance": distance_name}

    dense_index._build_vector_params = _stub_vector_params
    try:
        result = indexer.build_from_jsonl_streaming(path, lines_per_batch=3, max_passages=5)
        assert result.vector_count == 5
        assert fake_client.upsert_batches == [3, 2]
    finally:
        dense_index._build_vector_params = original_build_vector_params


def test_dense_jsonl_streaming_respects_max_index_rows_for_groups(tmp_path) -> None:
    path = tmp_path / "index_chunks_cap.jsonl"
    group1 = IndexChunkGroup(
        group_id="g1",
        source_row_ordinal=0,
        texts=[
            IndexText(text_idx=0, raw_idx=0, text="first chunk"),
            IndexText(text_idx=1, raw_idx=1, text="second chunk"),
        ],
        chunks=[
            IndexChunkItem(
                chunk_id="11111111-1111-1111-1111-111111111111",
                text_idxs=[0],
                context_idxs=[0],
                start_candidate_idx=0,
                end_candidate_idx=0,
            ),
            IndexChunkItem(
                chunk_id="22222222-2222-2222-2222-222222222222",
                text_idxs=[1],
                context_idxs=[1],
                start_candidate_idx=1,
                end_candidate_idx=1,
            ),
        ],
    )
    group2 = IndexChunkGroup(
        group_id="g2",
        source_row_ordinal=1,
        texts=[IndexText(text_idx=0, raw_idx=0, text="third chunk")],
        chunks=[
            IndexChunkItem(
                chunk_id="33333333-3333-3333-3333-333333333333",
                text_idxs=[0],
                context_idxs=[0],
                start_candidate_idx=0,
                end_candidate_idx=0,
            )
        ],
    )
    path.write_text(
        f"{group1.model_dump_json()}\n{group2.model_dump_json()}\n",
        encoding="utf-8",
    )

    settings = Settings(
        qdrant_collection="c_row_cap",
        dense_read_batch_lines=4,
        embedding_batch_size=8,
    )
    fake_client = FakeQdrantClient()
    fake_model = FakeEmbeddingModel()
    indexer = DenseIndexer(settings=settings, client=fake_client, model=fake_model)
    original_build_vector_params = dense_index._build_vector_params

    def _stub_vector_params(size: int, distance_name: str) -> dict[str, str | int]:
        return {"size": size, "distance": distance_name}

    dense_index._build_vector_params = _stub_vector_params
    try:
        assert count_index_records_jsonl(path, max_rows=1) == 2
        result = indexer.build_from_jsonl_streaming(
            path,
            lines_per_batch=4,
            max_index_rows=1,
        )
        assert result.vector_count == 2
        assert fake_client.upsert_batches == [2]
    finally:
        dense_index._build_vector_params = original_build_vector_params


def test_dense_jsonl_streaming_retries_transient_upsert_failure(tmp_path, monkeypatch) -> None:
    path = tmp_path / "silver_retry.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for i in range(2):
            p = Passage(passage_id=f"p{i}", text=f"doc{i}")
            handle.write(p.model_dump_json())
            handle.write("\n")

    settings = Settings(
        qdrant_collection="c_retry",
        dense_read_batch_lines=2,
        embedding_batch_size=8,
    )
    fake_client = FlakyFakeQdrantClient(fail_attempts=2)
    fake_model = FakeEmbeddingModel()
    indexer = DenseIndexer(settings=settings, client=fake_client, model=fake_model)
    original_build_vector_params = dense_index._build_vector_params

    def _stub_vector_params(size: int, distance_name: str) -> dict[str, str | int]:
        return {"size": size, "distance": distance_name}

    dense_index._build_vector_params = _stub_vector_params
    monkeypatch.setattr(dense_index.time, "sleep", lambda _seconds: None)
    try:
        result = indexer.build_from_jsonl_streaming(path, lines_per_batch=2)
        assert result.vector_count == 2
        assert fake_client.upsert_batches == [2]
    finally:
        dense_index._build_vector_params = original_build_vector_params


def test_dense_jsonl_streaming_resumes_from_checkpoint(tmp_path, monkeypatch) -> None:
    path = tmp_path / "silver_checkpoint.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for i in range(5):
            p = Passage(passage_id=f"p{i}", text=f"doc{i}")
            handle.write(p.model_dump_json())
            handle.write("\n")

    settings = Settings(
        output_dir=tmp_path / "artifacts",
        qdrant_collection="c_checkpoint",
        dense_read_batch_lines=2,
        embedding_batch_size=8,
    )
    failing_client = FailOnSecondUpsertQdrantClient()
    fake_model = FakeEmbeddingModel()
    indexer = DenseIndexer(settings=settings, client=failing_client, model=fake_model)
    original_build_vector_params = dense_index._build_vector_params

    def _stub_vector_params(size: int, distance_name: str) -> dict[str, str | int]:
        return {"size": size, "distance": distance_name}

    dense_index._build_vector_params = _stub_vector_params
    monkeypatch.setattr(dense_index.time, "sleep", lambda _seconds: None)
    try:
        try:
            indexer.build_from_jsonl_streaming(path, lines_per_batch=2)
            assert False, "Expected persistent upsert failure"
        except RuntimeError:
            pass
        assert settings.dense_checkpoint_path.is_file()

        resumed_client = FakeQdrantClient()
        resumed_indexer = DenseIndexer(settings=settings, client=resumed_client, model=fake_model)
        result = resumed_indexer.build_from_jsonl_streaming(path, lines_per_batch=2)
        assert result.vector_count == 5
        assert resumed_client.upsert_batches == [2, 1]
        assert not settings.dense_checkpoint_path.exists()
    finally:
        dense_index._build_vector_params = original_build_vector_params
