from __future__ import annotations

import json

from src.ingestion.models import IndexBuildManifest, Passage
from src.ingestion.passage_store import PassageStore


def test_passage_store_jsonl_roundtrip(tmp_path) -> None:
    passages = [
        Passage(passage_id="p1", text="alpha", source="doc1"),
        Passage(passage_id="p2", text="beta", source=None),
    ]
    path = tmp_path / "passages.jsonl"

    written = PassageStore.write_jsonl(passages, path)
    loaded = PassageStore.read_jsonl(path)

    assert written == 2
    assert [p.model_dump() for p in loaded] == [p.model_dump() for p in passages]


def test_write_manifest_is_stable_json(tmp_path) -> None:
    manifest = IndexBuildManifest(
        dataset_name="sentence-transformers/NQ-retrieval",
        dataset_split="train",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        passage_count=10,
        qdrant_collection="nq_passages",
        sparse_index_path="artifacts/sparse_index_manifest.json",
    )
    path = tmp_path / "manifest.json"
    PassageStore.write_manifest(manifest, path)

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["passage_count"] == 10
    assert payload["qdrant_collection"] == "nq_passages"
