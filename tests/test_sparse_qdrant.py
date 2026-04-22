from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.config.settings import Settings
from src.ingestion.models import SPARSE_INDEX_MANIFEST_VERSION, Passage, SparseIndexManifest
from src.retrieval.sparse_qdrant import (
    SparseQdrantIndexer,
    _pass1_scan,
    _sparse_vector_for_passage,
    compute_okapi_idf,
    encode_query_sparse_vector,
)


def _sparse_dot(
    q_indices: list[int], q_values: list[float], d_indices: list[int], d_values: list[float]
) -> float:
    dmap = dict(zip(d_indices, d_values, strict=True))
    return sum(w * dmap.get(i, 0.0) for i, w in zip(q_indices, q_values, strict=True))


def test_okapi_idf_matches_rank_bm25() -> None:
    from rank_bm25 import BM25Okapi

    corpus = [
        "the cat sat on mat".split(),
        "the dog sat on log".split(),
        "birds fly high".split(),
    ]
    bm25 = BM25Okapi(corpus)
    nd: dict[str, int] = {}
    for doc in corpus:
        for w in set(doc):
            nd[w] = nd.get(w, 0) + 1
    idf = compute_okapi_idf(nd, len(corpus), epsilon=0.25)
    for term in nd:
        assert idf[term] == pytest.approx(bm25.idf[term], rel=1e-9, abs=1e-9)


def test_sparse_dot_product_matches_bm25_scores(tmp_path: Path) -> None:
    from rank_bm25 import BM25Okapi

    texts = [
        "the cat sat on mat",
        "the dog sat on log",
        "birds fly high",
    ]
    corpus = [t.lower().split() for t in texts]
    path = tmp_path / "silver.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for i, t in enumerate(texts):
            p = Passage(passage_id=f"doc-{i}", text=t)
            handle.write(p.model_dump_json())
            handle.write("\n")

    bm25 = BM25Okapi(corpus)
    p1 = _pass1_scan(path, max_passages=None)
    assert p1.document_count == len(texts)
    idf = compute_okapi_idf(p1.nd, p1.document_count, epsilon=0.25)

    query_tokens = "the sat".split()
    expected = bm25.get_scores(query_tokens)
    q_i, q_v = encode_query_sparse_vector("the sat", term_to_id=p1.term_to_id)

    for i, t in enumerate(texts):
        d_i, d_v = _sparse_vector_for_passage(
            t,
            term_to_id=p1.term_to_id,
            idf=idf,
            corpus_size=p1.document_count,
            total_tokens=p1.total_tokens,
            k1=1.5,
            b=0.75,
        )
        got = _sparse_dot(q_i, q_v, d_i, d_v)
        assert got == pytest.approx(float(expected[i]), rel=1e-5, abs=1e-5)


def test_sparse_indexer_errors_when_collection_has_no_sparse_vectors() -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "c_only_dense",
        vectors_config={"dense": qm.VectorParams(size=2, distance=qm.Distance.COSINE)},
    )
    settings = Settings(qdrant_collection="c_only_dense", qdrant_sparse_vector_name="sparse")
    indexer = SparseQdrantIndexer(settings=settings, client=client)
    with pytest.raises(RuntimeError, match="no sparse vector"):
        indexer.assert_collection_sparse_ready()


def test_sparse_qdrant_end_to_end_in_memory(tmp_path: Path) -> None:
    client = QdrantClient(location=":memory:")
    collection = "coll"
    client.create_collection(
        collection,
        vectors_config={"dense": qm.VectorParams(size=2, distance=qm.Distance.COSINE)},
        sparse_vectors_config={"sparse": qm.SparseVectorParams()},
    )

    pid1 = str(uuid.uuid4())
    pid2 = str(uuid.uuid4())
    client.upsert(
        collection,
        points=[
            qm.PointStruct(id=pid1, vector={"dense": [1.0, 0.0]}, payload={"text": "the red car"}),
            qm.PointStruct(id=pid2, vector={"dense": [0.0, 1.0]}, payload={"text": "the blue bus"}),
        ],
    )

    path = tmp_path / "silver.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for pid, text in ((pid1, "the red car"), (pid2, "the blue bus")):
            handle.write(Passage(passage_id=pid, text=text).model_dump_json())
            handle.write("\n")

    settings = Settings(
        qdrant_collection=collection,
        qdrant_sparse_vector_name="sparse",
        sparse_upsert_batch_size=1,
        output_dir=tmp_path / "out",
    )
    indexer = SparseQdrantIndexer(settings=settings, client=client)
    result = indexer.build_from_jsonl(path)
    assert result.points_updated == 2
    assert settings.sparse_manifest_path.is_file()

    loaded = SparseIndexManifest.model_validate_json(
        settings.sparse_manifest_path.read_text(encoding="utf-8")
    )
    assert loaded.schema_version == SPARSE_INDEX_MANIFEST_VERSION
    assert loaded.points_updated == 2

    p1 = _pass1_scan(path, max_passages=None)
    q_i, q_v = encode_query_sparse_vector("the", term_to_id=p1.term_to_id)
    assert q_i, "fixture should yield a non-empty query sparse vector"
    hits = client.query_points(
        collection,
        query=qm.SparseVector(indices=q_i, values=q_v),
        using="sparse",
        limit=2,
    )
    assert len(hits.points) >= 1
    returned_ids = {str(p.id) for p in hits.points}
    assert pid1 in returned_ids and pid2 in returned_ids
