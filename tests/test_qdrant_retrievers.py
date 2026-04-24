from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.config.settings import Settings
from src.models.query_schemas import PassageHit
from src.retrieval.qdrant_retrievers import (
    DenseQdrantRetriever,
    HybridQdrantRetriever,
    QdrantModeRetriever,
    SparseQdrantRetriever,
)
from src.retrieval.sparse_qdrant import SparsePass1Data


class FakeEmbeddingModel:
    def encode(
        self, texts: list[str], *, batch_size: int, normalize_embeddings: bool
    ) -> list[list[float]]:
        assert batch_size == 1
        assert normalize_embeddings is True
        return [[0.2, 0.8] for _ in texts]


class FakeQueryClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def query_points(
        self,
        collection_name: str,
        query: Any,
        *,
        using: str,
        limit: int,
        with_payload: bool,
    ) -> Any:
        self.calls.append(
            {
                "collection_name": collection_name,
                "using": using,
                "limit": limit,
                "with_payload": with_payload,
                "query": query,
            }
        )
        if using == "dense":
            return SimpleNamespace(
                points=[
                    SimpleNamespace(id="p1", score=0.9, payload={"text": "d1", "source": "a"}),
                    SimpleNamespace(id="p2", score=0.8, payload={"text": "d2", "source": "b"}),
                ]
            )
        if using == "sparse":
            return SimpleNamespace(
                points=[
                    SimpleNamespace(id="p2", score=5.0, payload={"text": "d2", "source": "b"}),
                    SimpleNamespace(id="p3", score=3.0, payload={"text": "d3"}),
                ]
            )
        raise AssertionError(f"unexpected using={using}")


def _write_sparse_pass1_artifact(path: Path, *, max_passages: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1",
        "silver_path": str((path.parent / "passages.jsonl").resolve()),
        "max_passages": max_passages,
        "document_count": 2,
        "total_tokens": 5,
        "term_to_id": {"alpha": 0, "beta": 1},
        "created_at_utc": "2026-01-01T00:00:00+00:00",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_dense_retriever_maps_scores_and_ranks() -> None:
    settings = Settings(qdrant_collection="c")
    client = FakeQueryClient()
    retriever = DenseQdrantRetriever(settings=settings, client=client, model=FakeEmbeddingModel())
    hits = retriever.retrieve("hello", top_k=2)
    assert [h.passage_id for h in hits] == ["p1", "p2"]
    assert [h.dense_rank for h in hits] == [1, 2]
    assert hits[0].dense_score == 0.9
    assert hits[0].sparse_score is None


def test_sparse_retriever_uses_term_to_id_and_sets_sparse_fields(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(output_dir=output_dir, qdrant_collection="c")
    (output_dir / "passages.jsonl").write_text("", encoding="utf-8")
    _write_sparse_pass1_artifact(settings.sparse_pass1_path)
    client = FakeQueryClient()
    retriever = SparseQdrantRetriever(settings=settings, client=client)
    hits = retriever.retrieve("alpha alpha beta", top_k=2)
    assert [h.passage_id for h in hits] == ["p2", "p3"]
    assert [h.sparse_rank for h in hits] == [1, 2]
    assert hits[0].sparse_score == 5.0
    assert hits[0].dense_score is None


def test_sparse_retriever_returns_empty_when_query_has_no_vocab_overlap() -> None:
    settings = Settings(qdrant_collection="c")
    client = FakeQueryClient()
    pass1 = SparsePass1Data(
        document_count=1,
        total_tokens=1,
        term_to_id={"alpha": 0},
        silver_path_resolved="/tmp/passages.jsonl",
        max_passages=None,
    )
    retriever = SparseQdrantRetriever(settings=settings, client=client, pass1_data=pass1)
    hits = retriever.retrieve("zzz", top_k=3)
    assert hits == []
    assert client.calls == []


def test_hybrid_retriever_rrf_combines_dense_and_sparse() -> None:
    settings = Settings(qdrant_collection="c", hybrid_rrf_k=60)
    client = FakeQueryClient()
    dense = DenseQdrantRetriever(settings=settings, client=client, model=FakeEmbeddingModel())
    sparse = SparseQdrantRetriever(
        settings=settings,
        client=client,
        pass1_data=SparsePass1Data(
            document_count=1,
            total_tokens=1,
            term_to_id={"hello": 0},
            silver_path_resolved="/tmp/passages.jsonl",
            max_passages=None,
        ),
    )
    retriever = HybridQdrantRetriever(settings=settings, dense=dense, sparse=sparse)
    hits = retriever.retrieve("hello", top_k=3)
    assert [h.passage_id for h in hits] == ["p2", "p1", "p3"]
    assert [h.fusion_rank for h in hits] == [1, 2, 3]
    assert hits[0].dense_rank == 2
    assert hits[0].sparse_rank == 1


def test_mode_retriever_routes_to_selected_mode() -> None:
    settings = Settings(qdrant_collection="c")
    dense = SimpleNamespace(retrieve=lambda query, top_k: [PassageHit(passage_id="dense", text="d")])
    sparse = SimpleNamespace(
        retrieve=lambda query, top_k: [PassageHit(passage_id="sparse", text="s")]
    )
    hybrid = SimpleNamespace(
        retrieve=lambda query, top_k: [PassageHit(passage_id="hybrid", text="h")]
    )
    retriever = QdrantModeRetriever(
        settings=settings,
        mode="hybrid",
        dense=dense,
        sparse=sparse,
        hybrid=hybrid,
    )
    out = retriever.retrieve("q", top_k=1)
    assert out[0].passage_id == "hybrid"


def test_mode_retriever_dense_mode_does_not_require_sparse_dependencies() -> None:
    settings = Settings(qdrant_collection="c")
    dense = SimpleNamespace(retrieve=lambda query, top_k: [PassageHit(passage_id="dense", text="d")])
    retriever = QdrantModeRetriever(
        settings=settings,
        mode="dense",
        dense=dense,
        sparse=None,
        hybrid=None,
    )
    out = retriever.retrieve("q", top_k=1)
    assert out[0].passage_id == "dense"
