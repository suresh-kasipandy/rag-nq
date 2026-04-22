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
        assert normalize_embeddings is True
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.exists = False
        self.points: list[dict] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(
        self, collection_name: str, vectors_config, sparse_vectors_config=None, **kwargs
    ) -> None:
        self.exists = True
        if isinstance(vectors_config, dict):
            named = vectors_config["dense"]
            if isinstance(named, dict):
                assert named["size"] == 3
            else:
                assert getattr(named, "size", None) == 3
        else:
            assert getattr(vectors_config, "size", None) == 3

    def upsert(self, collection_name: str, points: list[dict]) -> None:
        self.points.extend(points)

    def count(self, collection_name: str, exact: bool = True):
        return type("CountResult", (), {"count": len(self.points)})()


def test_dense_index_build_and_count() -> None:
    settings = Settings(qdrant_collection="test_collection")
    fake_client = FakeQdrantClient()
    fake_model = FakeEmbeddingModel()
    indexer = DenseIndexer(settings=settings, client=fake_client, model=fake_model)
    original_build_vector_params = dense_index._build_vector_params

    def _stub_vector_params(size: int, distance_name: str) -> dict[str, str | int]:
        return {"size": size, "distance": distance_name}

    dense_index._build_vector_params = _stub_vector_params

    try:
        passages = [
            Passage(passage_id="p1", text="A", source="x"),
            Passage(passage_id="p2", text="B", source="y"),
        ]
        result = indexer.build(passages)

        assert result.vector_count == 2
        assert result.vector_size == 3
        assert indexer.count() == 2
        assert fake_client.points[0]["payload"] == {"text": "A", "source": "x"}
        assert fake_client.points[1]["payload"] == {"text": "B", "source": "y"}
    finally:
        dense_index._build_vector_params = original_build_vector_params


def test_dense_index_payload_includes_optional_passage_metadata() -> None:
    settings = Settings(qdrant_collection="test_collection_meta")
    fake_client = FakeQdrantClient()
    fake_model = FakeEmbeddingModel()
    indexer = DenseIndexer(settings=settings, client=fake_client, model=fake_model)
    original_build_vector_params = dense_index._build_vector_params

    def _stub_vector_params(size: int, distance_name: str) -> dict[str, str | int]:
        return {"size": size, "distance": distance_name}

    dense_index._build_vector_params = _stub_vector_params

    try:
        passages = [
            Passage(
                passage_id="p1",
                text="body",
                source="src",
                title="T",
                question="Q?",
                passage_type="wiki",
                document_url="https://example.com",
                long_answers=["42"],
            )
        ]
        indexer.build(passages)
        assert fake_client.points[0]["payload"] == {
            "text": "body",
            "source": "src",
            "title": "T",
            "question": "Q?",
            "passage_type": "wiki",
            "document_url": "https://example.com",
            "long_answers": ["42"],
        }
    finally:
        dense_index._build_vector_params = original_build_vector_params
