from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.config.settings import Settings
from src.ingestion.models import (
    SPARSE_INDEX_MANIFEST_VERSION,
    IndexChunkGroup,
    IndexChunkItem,
    IndexText,
    Passage,
    SparseIndexManifest,
)
from src.retrieval.sparse_index import SPARSE_ANALYZER_VERSION
from src.retrieval.sparse_qdrant import (
    SparseQdrantIndexer,
    _pass1_scan,
    _sparse_vector_for_passage,
    compute_okapi_idf,
    encode_query_sparse_vector,
    load_sparse_pass1_artifact,
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
    p1 = _pass1_scan(path, max_passages=None, analyzer="whitespace")
    assert p1.document_count == len(texts)
    idf = compute_okapi_idf(p1.nd, p1.document_count, epsilon=0.25)

    query_tokens = "the sat".split()
    expected = bm25.get_scores(query_tokens)
    q_i, q_v = encode_query_sparse_vector(
        "the sat",
        term_to_id=p1.term_to_id,
        analyzer="whitespace",
    )

    for i, t in enumerate(texts):
        d_i, d_v = _sparse_vector_for_passage(
            t,
            term_to_id=p1.term_to_id,
            idf=idf,
            corpus_size=p1.document_count,
            total_tokens=p1.total_tokens,
            k1=1.5,
            b=0.75,
            analyzer="whitespace",
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
        sparse_analyzer="whitespace",
    )
    indexer = SparseQdrantIndexer(settings=settings, client=client)
    result = indexer.build_from_jsonl(path)
    assert result.points_updated == 2
    assert settings.sparse_manifest_path.is_file()
    assert settings.sparse_pass1_path.is_file()

    loaded = SparseIndexManifest.model_validate_json(
        settings.sparse_manifest_path.read_text(encoding="utf-8")
    )
    assert loaded.schema_version == SPARSE_INDEX_MANIFEST_VERSION
    assert loaded.points_updated == 2

    p1 = _pass1_scan(path, max_passages=None, analyzer="whitespace")
    q_i, q_v = encode_query_sparse_vector("the", term_to_id=p1.term_to_id, analyzer="whitespace")
    assert q_i, "fixture should yield a non-empty query sparse vector"

    pass1_artifact = load_sparse_pass1_artifact(
        path=settings.sparse_pass1_path,
        silver_path=path,
        max_passages=None,
        sparse_analyzer="whitespace",
    )
    assert pass1_artifact.document_count == 2
    assert pass1_artifact.term_to_id == p1.term_to_id

    hits = client.query_points(
        collection,
        query=qm.SparseVector(indices=q_i, values=q_v),
        using="sparse",
        limit=2,
    )
    assert len(hits.points) >= 1
    returned_ids = {str(p.id) for p in hits.points}
    assert pid1 in returned_ids and pid2 in returned_ids


class FakeSparseClient:
    def __init__(self, *, fail_when_id_seen: str | None = None) -> None:
        self.fail_when_id_seen = fail_when_id_seen
        self.updated_ids: list[list[str]] = []

    def get_collection(self, collection_name: str) -> Any:
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(sparse_vectors={"sparse": object()}),
            )
        )

    def update_vectors(self, collection_name: str, points: Any) -> None:
        ids = [str(p.id) for p in points]
        self.updated_ids.append(ids)
        if self.fail_when_id_seen is not None and self.fail_when_id_seen in ids:
            raise RuntimeError(f"forced failure on ids={','.join(ids)}")


class HttpNoiseSparseClient(FakeSparseClient):
    def update_vectors(self, collection_name: str, points: Any) -> None:
        logging.getLogger("httpx").info("httpx-info-sparse")
        logging.getLogger("httpx").warning("httpx-warning-sparse")
        super().update_vectors(collection_name, points)


def _write_sparse_fixture(path: Path, ids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for i, pid in enumerate(ids):
            handle.write(Passage(passage_id=pid, text=f"text {i}").model_dump_json())
            handle.write("\n")


def test_sparse_indexer_uses_chunk_id_for_index_chunks(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    chunk_id = str(uuid.uuid4())
    group = IndexChunkGroup(
        group_id="group-1",
        source_row_ordinal=0,
        question="Which planet is smallest?",
        texts=[
            IndexText(
                text_idx=0,
                raw_idx=1,
                text="Mercury is the smallest planet.",
            ),
            IndexText(
                text_idx=1,
                raw_idx=None,
                text="Mercury is the smallest planet in the Solar System.",
            ),
        ],
        chunks=[
            IndexChunkItem(
                chunk_id=chunk_id,
                text_idxs=[0],
                context_idxs=[1],
                start_candidate_idx=1,
                end_candidate_idx=1,
            )
        ],
    )
    chunks = tmp_path / "index_chunks.jsonl"
    chunks.write_text(group.model_dump_json() + "\n", encoding="utf-8")
    settings = Settings(
        output_dir=tmp_path / "artifacts",
        sparse_upsert_batch_size=1,
        qdrant_sparse_vector_name="sparse",
        qdrant_collection="nq_passages",
        progress_log_every_records=1,
        progress_log_every_batches=1,
    )

    caplog.set_level(logging.INFO)
    client = FakeSparseClient()
    indexer = SparseQdrantIndexer(settings=settings, client=client)
    result = indexer.build_from_jsonl(chunks)

    assert result.points_updated == 1
    assert client.updated_ids == [[chunk_id]]
    progress_records = [
        (getattr(record, "stage", None), record.getMessage()) for record in caplog.records
    ]
    assert any(
        stage == "sparse_pass1" and message.startswith("start records=0 ")
        for stage, message in progress_records
    )
    assert any(
        stage == "sparse_pass1" and message.startswith("complete records=1 ")
        for stage, message in progress_records
    )
    assert any(
        stage == "sparse_pass2" and message.startswith("progress batches=1/1")
        for stage, message in progress_records
    )


def test_sparse_pass1_respects_max_index_rows_for_groups(tmp_path: Path) -> None:
    group1 = IndexChunkGroup(
        group_id="group-1",
        source_row_ordinal=0,
        texts=[
            IndexText(text_idx=0, raw_idx=0, text="alpha beta"),
            IndexText(text_idx=1, raw_idx=1, text="gamma delta"),
        ],
        chunks=[
            IndexChunkItem(
                chunk_id="chunk-1",
                text_idxs=[0],
                context_idxs=[0],
                start_candidate_idx=0,
                end_candidate_idx=0,
            ),
            IndexChunkItem(
                chunk_id="chunk-2",
                text_idxs=[1],
                context_idxs=[1],
                start_candidate_idx=1,
                end_candidate_idx=1,
            ),
        ],
    )
    group2 = IndexChunkGroup(
        group_id="group-2",
        source_row_ordinal=1,
        texts=[IndexText(text_idx=0, raw_idx=0, text="epsilon zeta")],
        chunks=[
            IndexChunkItem(
                chunk_id="chunk-3",
                text_idxs=[0],
                context_idxs=[0],
                start_candidate_idx=0,
                end_candidate_idx=0,
            )
        ],
    )
    chunks = tmp_path / "index_chunks.jsonl"
    chunks.write_text(
        f"{group1.model_dump_json()}\n{group2.model_dump_json()}\n",
        encoding="utf-8",
    )

    p1 = _pass1_scan(chunks, max_index_rows=1, analyzer="whitespace")

    assert p1.document_count == 2
    assert p1.term_to_id == {"alpha": 0, "beta": 1, "gamma": 2, "delta": 3}


def test_sparse_resume_uses_checkpoint_after_partial_failure(tmp_path: Path) -> None:
    ids = [str(uuid.uuid4()) for _ in range(3)]
    silver = tmp_path / "silver.jsonl"
    _write_sparse_fixture(silver, ids)
    settings = Settings(
        output_dir=tmp_path / "artifacts",
        sparse_upsert_batch_size=1,
        qdrant_sparse_vector_name="sparse",
        qdrant_collection="nq_passages",
    )

    first_client = FakeSparseClient(fail_when_id_seen=ids[1])
    first_indexer = SparseQdrantIndexer(settings=settings, client=first_client)
    with pytest.raises(RuntimeError, match="forced failure"):
        first_indexer.build_from_jsonl(silver)
    checkpoint_payload = settings.sparse_checkpoint_path.read_text(encoding="utf-8")
    assert '"indexed_count": 1' in checkpoint_payload

    second_client = FakeSparseClient()
    second_indexer = SparseQdrantIndexer(settings=settings, client=second_client)
    result = second_indexer.build_from_jsonl(silver)
    assert result.points_updated == 2
    assert second_client.updated_ids == [[ids[1]], [ids[2]]]
    assert not settings.sparse_checkpoint_path.exists()


def test_sparse_checkpoint_invalidated_when_inputs_change(tmp_path: Path) -> None:
    ids = [str(uuid.uuid4()) for _ in range(3)]
    silver = tmp_path / "silver.jsonl"
    _write_sparse_fixture(silver, ids)
    settings = Settings(
        output_dir=tmp_path / "artifacts",
        sparse_upsert_batch_size=1,
        qdrant_sparse_vector_name="sparse",
        qdrant_collection="nq_passages",
        bm25_k1=1.5,
    )
    settings.sparse_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    settings.sparse_checkpoint_path.write_text(
        "{\n"
        '  "schema_version": "1",\n'
        f'  "silver_path": "{silver.resolve()}",\n'
        '  "collection_name": "nq_passages",\n'
        '  "sparse_vector_name": "sparse",\n'
        '  "max_passages": null,\n'
        '  "bm25_k1": 1.2,\n'
        '  "bm25_b": 0.75,\n'
        '  "bm25_epsilon": 0.25,\n'
        '  "sparse_upsert_batch_size": 1,\n'
        '  "indexed_count": 2\n'
        "}\n",
        encoding="utf-8",
    )

    client = FakeSparseClient(fail_when_id_seen=ids[0])
    indexer = SparseQdrantIndexer(settings=settings, client=client)
    with pytest.raises(RuntimeError, match=f"ids={ids[0]}"):
        indexer.build_from_jsonl(silver)


def test_sparse_concurrency_smoke_keeps_batch_size(tmp_path: Path) -> None:
    ids = [str(uuid.uuid4()) for _ in range(4)]
    silver = tmp_path / "silver.jsonl"
    _write_sparse_fixture(silver, ids)
    settings = Settings(
        output_dir=tmp_path / "artifacts",
        sparse_upsert_batch_size=2,
        sparse_workers=2,
        sparse_write_concurrency=2,
        qdrant_sparse_vector_name="sparse",
        qdrant_collection="nq_passages",
    )
    client = FakeSparseClient()
    indexer = SparseQdrantIndexer(settings=settings, client=client)
    result = indexer.build_from_jsonl(silver)
    assert result.points_updated == 4
    assert len(client.updated_ids) == 2
    assert sorted(len(batch) for batch in client.updated_ids) == [2, 2]


def test_sparse_indexer_quiets_httpx_info_keeps_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ids = [str(uuid.uuid4())]
    silver = tmp_path / "silver_http_noise.jsonl"
    _write_sparse_fixture(silver, ids)
    settings = Settings(
        output_dir=tmp_path / "artifacts",
        sparse_upsert_batch_size=1,
        qdrant_sparse_vector_name="sparse",
        qdrant_collection="nq_passages",
    )
    caplog.set_level(logging.INFO)
    client = HttpNoiseSparseClient()
    indexer = SparseQdrantIndexer(settings=settings, client=client)
    result = indexer.build_from_jsonl(silver)

    assert result.points_updated == 1
    messages = [record.getMessage() for record in caplog.records]
    assert "httpx-warning-sparse" in messages
    assert "httpx-info-sparse" not in messages


def test_sparse_indexer_restores_httpx_logger_level(tmp_path: Path) -> None:
    ids = [str(uuid.uuid4())]
    silver = tmp_path / "silver_http_logger_restore.jsonl"
    _write_sparse_fixture(silver, ids)
    settings = Settings(
        output_dir=tmp_path / "artifacts",
        sparse_upsert_batch_size=1,
        qdrant_sparse_vector_name="sparse",
        qdrant_collection="nq_passages",
    )
    logger = logging.getLogger("httpx")
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        client = HttpNoiseSparseClient()
        indexer = SparseQdrantIndexer(settings=settings, client=client)
        result = indexer.build_from_jsonl(silver)
        assert result.points_updated == 1
        assert logger.level == logging.INFO
    finally:
        logger.setLevel(previous_level)


def test_sparse_build_reuses_existing_pass1_artifact(tmp_path: Path, monkeypatch) -> None:
    ids = [str(uuid.uuid4()) for _ in range(2)]
    silver = tmp_path / "silver.jsonl"
    _write_sparse_fixture(silver, ids)
    settings = Settings(
        output_dir=tmp_path / "artifacts",
        sparse_upsert_batch_size=1,
        qdrant_sparse_vector_name="sparse",
        qdrant_collection="nq_passages",
    )

    p1 = _pass1_scan(silver, max_passages=None)
    settings.sparse_pass1_path.parent.mkdir(parents=True, exist_ok=True)
    settings.sparse_pass1_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "silver_path": str(silver.resolve()),
                "max_passages": None,
                "document_count": p1.document_count,
                "total_tokens": p1.total_tokens,
                "term_to_id": p1.term_to_id,
                "sparse_analyzer": "regex_stem_stop",
                "sparse_analyzer_version": SPARSE_ANALYZER_VERSION,
                "created_at_utc": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    def _fail_pass1_scan(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("build_from_jsonl should reuse existing pass1 artifact")

    monkeypatch.setattr("src.retrieval.sparse_qdrant._pass1_scan", _fail_pass1_scan)
    client = FakeSparseClient()
    indexer = SparseQdrantIndexer(settings=settings, client=client)
    result = indexer.build_from_jsonl(silver)
    assert result.points_updated == 2
