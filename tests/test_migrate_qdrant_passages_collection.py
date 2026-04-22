from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client.http import models as qm

from src.scripts.migrate_qdrant_passages_collection import migrate_passages_collection


def _source_collection_info() -> SimpleNamespace:
    dense = qm.VectorParams(size=2, distance=qm.Distance.COSINE)
    return SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors={"dense": dense})),
    )


class FakeMigrateClient:
    """Minimal stand-in for Qdrant client operations used in migration."""

    def __init__(self) -> None:
        self.source_info = _source_collection_info()
        self.target_exists = False
        self.deleted: list[str] = []
        self.created: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
        self.upserts: list[tuple[str, list[Any]]] = []
        self._scroll_calls = 0
        self.source_count = 2
        self.target_count = 2

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name == "tgt" and self.target_exists

    def get_collection(self, collection_name: str) -> Any:
        assert collection_name == "src"
        return self.source_info

    def create_collection(
        self,
        collection_name: str,
        *,
        vectors_config: Any,
        sparse_vectors_config: Any | None = None,
    ) -> bool:
        self.created.append((collection_name, vectors_config, sparse_vectors_config))
        self.target_exists = True
        return True

    def delete_collection(self, collection_name: str) -> bool:
        self.deleted.append(collection_name)
        self.target_exists = False
        return True

    def scroll(
        self,
        collection_name: str,
        *,
        limit: int,
        offset: Any,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[list[Any], Any]:
        assert collection_name == "src"
        assert with_payload is True and with_vectors is True
        self._scroll_calls += 1
        if self._scroll_calls == 1:
            records = [
                SimpleNamespace(
                    id="p1",
                    vector={"dense": [0.1, 0.2]},
                    payload={"text": "a"},
                ),
                SimpleNamespace(
                    id="p2",
                    vector={"dense": [0.3, 0.4]},
                    payload={"text": "b"},
                ),
            ]
            return records, None
        return [], None

    def upsert(self, collection_name: str, points: list[Any]) -> bool:
        self.upserts.append((collection_name, points))
        return True

    def count(self, collection_name: str, *, exact: bool = True) -> Any:
        assert exact is True
        if collection_name == "src":
            return SimpleNamespace(count=self.source_count)
        if collection_name == "tgt":
            return SimpleNamespace(count=self.target_count)
        raise AssertionError(collection_name)


def test_migrate_creates_sparse_slot_and_upserts_dense_only() -> None:
    client = FakeMigrateClient()
    migrate_passages_collection(
        client,
        source_collection="src",
        target_collection="tgt",
        vector_name="dense",
        sparse_vector_name="sparse",
        batch_size=256,
        recreate_target=False,
        dry_run=False,
    )

    assert len(client.created) == 1
    name, vec_cfg, sparse_cfg = client.created[0]
    assert name == "tgt"
    assert "dense" in vec_cfg
    assert sparse_cfg is not None
    assert "sparse" in sparse_cfg
    assert isinstance(sparse_cfg["sparse"], qm.SparseVectorParams)

    assert len(client.upserts) == 1
    coll, points = client.upserts[0]
    assert coll == "tgt"
    assert [p.id for p in points] == ["p1", "p2"]
    assert points[0].vector == {"dense": [0.1, 0.2]}
    assert points[0].payload == {"text": "a"}
    assert "sparse" not in points[0].vector


def test_migrate_dry_run_does_not_write() -> None:
    client = FakeMigrateClient()
    migrate_passages_collection(
        client,
        source_collection="src",
        target_collection="tgt",
        vector_name="dense",
        sparse_vector_name="sparse",
        batch_size=10,
        recreate_target=False,
        dry_run=True,
    )
    assert client.created == []
    assert client.upserts == []
    assert client._scroll_calls == 1


def test_migrate_recreate_target_deletes_first() -> None:
    client = FakeMigrateClient()
    client.target_exists = True
    migrate_passages_collection(
        client,
        source_collection="src",
        target_collection="tgt",
        vector_name="dense",
        sparse_vector_name="sparse",
        batch_size=256,
        recreate_target=True,
        dry_run=False,
    )
    assert client.deleted == ["tgt"]


def test_migrate_raises_when_counts_mismatch() -> None:
    client = FakeMigrateClient()
    client.target_count = 1
    with pytest.raises(ValueError, match="Count mismatch"):
        migrate_passages_collection(
            client,
            source_collection="src",
            target_collection="tgt",
            vector_name="dense",
            sparse_vector_name="sparse",
            batch_size=256,
            recreate_target=False,
            dry_run=False,
        )


def test_migrate_raises_when_target_exists_without_recreate() -> None:
    client = FakeMigrateClient()
    client.target_exists = True
    with pytest.raises(ValueError, match="already exists"):
        migrate_passages_collection(
            client,
            source_collection="src",
            target_collection="tgt",
            vector_name="dense",
            sparse_vector_name="sparse",
            batch_size=256,
            recreate_target=False,
            dry_run=False,
        )


def test_migrate_raises_when_source_has_no_named_vector() -> None:
    client = FakeMigrateClient()
    with pytest.raises(ValueError, match="no vector named"):
        migrate_passages_collection(
            client,
            source_collection="src",
            target_collection="tgt",
            vector_name="wrong_name",
            sparse_vector_name="sparse",
            batch_size=256,
            recreate_target=False,
            dry_run=False,
        )


class FakeMigrateClientMultiPage(FakeMigrateClient):
    """Source yields two scroll pages before exhaustion (covers pagination)."""

    def __init__(self) -> None:
        super().__init__()
        self.source_count = self.target_count = 2

    def scroll(
        self,
        collection_name: str,
        *,
        limit: int,
        offset: Any,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[list[Any], Any]:
        assert collection_name == "src"
        assert with_payload is True and with_vectors is True
        self._scroll_calls += 1
        if self._scroll_calls == 1:
            assert offset is None
            return [
                SimpleNamespace(
                    id="p1",
                    vector={"dense": [0.1, 0.2]},
                    payload={"text": "a"},
                ),
            ], "next-page-token"
        if self._scroll_calls == 2:
            assert offset == "next-page-token"
            return [
                SimpleNamespace(
                    id="p2",
                    vector={"dense": [0.3, 0.4]},
                    payload={"text": "b"},
                ),
            ], None
        return [], None


def test_migrate_multi_page_scroll_upserts_each_batch() -> None:
    client = FakeMigrateClientMultiPage()
    migrate_passages_collection(
        client,
        source_collection="src",
        target_collection="tgt",
        vector_name="dense",
        sparse_vector_name="sparse",
        batch_size=256,
        recreate_target=False,
        dry_run=False,
    )
    assert client._scroll_calls == 2
    assert len(client.upserts) == 2
    assert [len(batch) for _, batch in client.upserts] == [1, 1]
    assert [p.id for _, batch in client.upserts for p in batch] == ["p1", "p2"]
