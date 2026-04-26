from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from tests.conftest import FIXTURES_DIR, load_jsonl_fixture_rows

from src.config.settings import Settings
from src.ingestion.chunk_raw import run_chunk_ingest
from src.ingestion import nq_loader
from src.ingestion.ingest_silver import load_ingest_manifest, run_ingest
from src.ingestion.models import SparseIndexManifest
from src.retrieval.dense_index import DenseIndexer
from src.retrieval.qdrant_retrievers import (
    DenseQdrantRetriever,
    HybridQdrantRetriever,
    QdrantModeRetriever,
    SparseQdrantRetriever,
)
from src.retrieval.sparse_qdrant import SparseQdrantIndexer, load_sparse_pass1_artifact


class TinyEmbeddingModel:
    """Deterministic tiny embedding model for integration tests."""

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(
        self, texts: list[str], *, batch_size: int, normalize_embeddings: bool
    ) -> list[list[float]]:
        assert batch_size >= 1
        assert normalize_embeddings is True
        out: list[list[float]] = []
        for text in texts:
            s = text.lower()
            out.append(
                [
                    1.0 if "car" in s or "vehicle" in s else 0.0,
                    1.0 if "dog" in s or "animal" in s else 0.0,
                    1.0 if "apple" in s or "fruit" in s else 0.0,
                    1.0 if "red" in s else 0.0,
                ]
            )
        return out


class InMemoryQdrantAdapter:
    """Adapter that makes local Qdrant compatible with indexer client expectations."""

    def __init__(self) -> None:
        self._client = QdrantClient(location=":memory:")

    def collection_exists(self, collection_name: str) -> bool:
        return self._client.collection_exists(collection_name)

    def create_collection(self, collection_name: str, vectors_config: Any, **kwargs: Any) -> None:
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            **kwargs,
        )

    def upsert(self, collection_name: str, points: list[Any]) -> Any:
        if points and isinstance(points[0], dict):
            converted: list[qm.PointStruct] = []
            for point in points:
                converted.append(
                    qm.PointStruct(
                        id=point["id"],
                        vector=point["vector"],
                        payload=point.get("payload"),
                    )
                )
            return self._client.upsert(collection_name=collection_name, points=converted)
        return self._client.upsert(collection_name=collection_name, points=points)

    def count(self, collection_name: str, exact: bool = True) -> Any:
        return self._client.count(collection_name=collection_name, exact=exact)

    def get_collection(self, collection_name: str) -> Any:
        return self._client.get_collection(collection_name)

    def update_vectors(self, collection_name: str, points: Any) -> Any:
        return self._client.update_vectors(collection_name=collection_name, points=points)

    def query_points(
        self,
        collection_name: str,
        query: Any,
        *,
        using: str,
        limit: int,
        with_payload: bool,
    ) -> Any:
        return self._client.query_points(
            collection_name=collection_name,
            query=query,
            using=using,
            limit=limit,
            with_payload=with_payload,
        )


def _build_settings(tmp_path: Path, *, collection: str, max_passages: int | None) -> Settings:
    return Settings(
        output_dir=tmp_path / "artifacts",
        qdrant_collection=collection,
        qdrant_sparse_vector_name="sparse",
        qdrant_vector_name="dense",
        dataset_name="sentence-transformers/NQ-retrieval",
        dataset_split="train",
        max_passages=max_passages,
        ingest_show_progress=False,
        dense_read_batch_lines=2,
        sparse_upsert_batch_size=2,
        sparse_workers=1,
        sparse_write_concurrency=1,
        chunk_min_tokens_soft=1,
        chunk_min_tokens_hard=1,
        chunk_target_tokens=8,
        chunk_max_tokens=40,
    )


def _run_pipeline_to_m3(
    *,
    settings: Settings,
    client: InMemoryQdrantAdapter,
    model: TinyEmbeddingModel,
) -> None:
    chunk_manifest, skipped = run_chunk_ingest(settings, force=True)
    assert skipped is False
    assert chunk_manifest.chunk_count > 0
    dense_indexer = DenseIndexer(settings=settings, client=client, model=model)
    dense_indexer.build_from_jsonl_streaming(
        settings.index_chunks_path,
        lines_per_batch=settings.dense_read_batch_lines,
        max_index_rows=settings.max_index_rows,
        max_passages=settings.max_passages,
    )
    sparse_indexer = SparseQdrantIndexer(settings=settings, client=client)
    sparse_indexer.build_from_jsonl(
        settings.index_chunks_path,
        max_index_rows=settings.max_index_rows,
        max_passages=settings.max_passages,
    )


def _assert_retrieval_modes(
    settings: Settings, client: InMemoryQdrantAdapter, model: TinyEmbeddingModel
) -> None:
    dense = DenseQdrantRetriever(settings=settings, client=client, model=model)
    sparse = SparseQdrantRetriever(settings=settings, client=client)
    hybrid = HybridQdrantRetriever(settings=settings, dense=dense, sparse=sparse)

    dense_mode = QdrantModeRetriever(settings=settings, mode="dense", dense=dense)
    sparse_mode = QdrantModeRetriever(settings=settings, mode="sparse", sparse=sparse)
    hybrid_mode = QdrantModeRetriever(settings=settings, mode="hybrid", hybrid=hybrid)

    query = "red car"
    dense_hits = dense_mode.retrieve(query, top_k=3)
    sparse_hits = sparse_mode.retrieve(query, top_k=3)
    hybrid_hits = hybrid_mode.retrieve(query, top_k=3)

    assert dense_hits, "dense mode should return at least one hit"
    assert sparse_hits, "sparse mode should return at least one hit"
    assert hybrid_hits, "hybrid mode should return at least one hit"

    assert dense_hits[0].dense_rank is not None
    assert dense_hits[0].dense_score is not None
    assert sparse_hits[0].sparse_rank is not None
    assert sparse_hits[0].sparse_score is not None
    assert hybrid_hits[0].fusion_rank is not None


def test_pipeline_e2e_m3_fixture_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_rows = load_jsonl_fixture_rows(FIXTURES_DIR / "nq_sample_rows.jsonl")
    monkeypatch.setattr(nq_loader, "_load_dataset", lambda *_a, **_k: fixture_rows)

    settings = _build_settings(tmp_path, collection="e2e_m3_fixture", max_passages=None)
    ingest_manifest, skipped = run_ingest(settings, force=True)
    assert skipped is False
    assert ingest_manifest.line_count > 0
    assert settings.passages_path.is_file()
    assert settings.ingest_manifest_path.is_file()

    client = InMemoryQdrantAdapter()
    model = TinyEmbeddingModel()
    _run_pipeline_to_m3(settings=settings, client=client, model=model)

    assert settings.sparse_manifest_path.is_file()
    assert settings.sparse_pass1_path.is_file()
    assert not settings.sparse_checkpoint_path.exists()
    assert settings.index_chunks_path.is_file()
    assert settings.chunk_manifest_path.is_file()
    assert load_ingest_manifest(settings.ingest_manifest_path) is not None
    loaded_sparse_manifest = SparseIndexManifest.model_validate_json(
        settings.sparse_manifest_path.read_text(encoding="utf-8")
    )
    assert loaded_sparse_manifest.points_updated > 0
    pass1 = load_sparse_pass1_artifact(
        path=settings.sparse_pass1_path,
        silver_path=settings.index_chunks_path,
        max_index_rows=settings.max_index_rows,
        max_passages=settings.max_passages,
    )
    assert pass1.document_count == loaded_sparse_manifest.points_updated

    _assert_retrieval_modes(settings, client, model)


@pytest.mark.live_network
def test_pipeline_e2e_m3_live_network_opt_in(tmp_path: Path) -> None:
    if os.getenv("RAG_RUN_LIVE_NETWORK_TESTS", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        pytest.skip("Set RAG_RUN_LIVE_NETWORK_TESTS=1 to run live-network E2E tests.")

    settings = _build_settings(tmp_path, collection="e2e_m3_live", max_passages=50)
    ingest_manifest, _ = run_ingest(settings, force=True)
    assert ingest_manifest.line_count > 0

    client = InMemoryQdrantAdapter()
    model = TinyEmbeddingModel()
    _run_pipeline_to_m3(settings=settings, client=client, model=model)

    first_line = settings.passages_path.read_text(encoding="utf-8").splitlines()[0]
    first_payload: dict[str, Any] = json.loads(first_line)
    query = " ".join(str(first_payload["text"]).split()[:3])
    dense = DenseQdrantRetriever(settings=settings, client=client, model=model)
    sparse = SparseQdrantRetriever(settings=settings, client=client)
    hybrid = HybridQdrantRetriever(settings=settings, dense=dense, sparse=sparse)
    hits = hybrid.retrieve(query, top_k=5)
    assert hits
