from __future__ import annotations

from src.config.settings import Settings
from src.ingestion.models import Passage
from src.retrieval import dense_index
from src.retrieval.dense_index import DenseIndexer


class FakeEmbeddingModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, *, batch_size: int, normalize_embeddings: bool):
        assert batch_size > 0
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
