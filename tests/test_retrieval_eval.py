from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.settings import Settings
from src.evaluation.retrieval_eval import (
    build_eval_cases_from_passages_jsonl,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    run_retrieval_evaluation,
)
from src.ingestion.models import Passage
from src.ingestion.passage_store import PassageStore
from src.models.query_schemas import PassageHit


def test_metric_functions_on_toy_ranked_list() -> None:
    retrieved = ["p1", "p2", "p3", "p4"]
    relevant = {"p2", "p4"}

    assert compute_recall_at_k(retrieved, relevant, k=3) == 0.5
    assert compute_mrr_at_k(retrieved, relevant, k=3) == 0.5
    ndcg = compute_ndcg_at_k(retrieved, relevant, k=3)
    assert ndcg == 0.38685280723454163


def test_metric_functions_reject_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        compute_recall_at_k(["p1"], {"p1"}, k=0)
    with pytest.raises(ValueError, match="k must be >= 1"):
        compute_mrr_at_k(["p1"], {"p1"}, k=0)
    with pytest.raises(ValueError, match="k must be >= 1"):
        compute_ndcg_at_k(["p1"], {"p1"}, k=0)


def test_build_eval_cases_keeps_only_question_with_relevance(tmp_path: Path) -> None:
    passages_path = tmp_path / "passages.jsonl"
    PassageStore.write_jsonl(
        [
            Passage(
                passage_id="p1",
                text="Paris is the capital of France.",
                question="What is the capital of France?",
                long_answers=["Paris"],
            ),
            Passage(
                passage_id="p2",
                text="Berlin is the capital of Germany.",
                question="What is the capital of France?",
                long_answers=["Paris"],
            ),
            Passage(
                passage_id="p3",
                text="No useful answer here.",
                question="Question with no labels",
                long_answers=[],
            ),
        ],
        passages_path,
    )

    cases = build_eval_cases_from_passages_jsonl(passages_path, max_queries=None)
    assert len(cases) == 1
    assert cases[0].query == "What is the capital of France?"
    assert cases[0].relevant_passage_ids == ["p1"]


def test_run_retrieval_eval_writes_deterministic_report(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "artifacts"
    settings = Settings(
        output_dir=output_dir,
        qdrant_url="http://localhost:6333",
        qdrant_collection="nq_passages",
        qdrant_vector_name="dense",
        qdrant_sparse_vector_name="sparse",
    )
    passages_path = settings.passages_path
    PassageStore.write_jsonl(
        [
            Passage(
                passage_id="p1",
                text="Paris is the capital of France.",
                question="What is the capital of France?",
                long_answers=["Paris"],
            ),
            Passage(
                passage_id="p2",
                text="Madrid is in Spain.",
                question="What is the capital of France?",
                long_answers=["Paris"],
            ),
        ],
        passages_path,
    )
    sparse_manifest_payload = {
        "schema_version": "1",
        "silver_path_resolved": str(passages_path.resolve()),
        "qdrant_url": "http://localhost:6333",
        "qdrant_collection": "nq_passages",
        "qdrant_sparse_vector_name": "sparse",
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "bm25_epsilon": 0.25,
        "sparse_upsert_batch_size": 8,
        "document_count": 2,
        "vocabulary_size": 4,
        "points_updated": 2,
        "avg_doc_len": 6.0,
        "idf_term_count": 4,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
    }
    settings.sparse_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.sparse_manifest_path.write_text(
        json.dumps(sparse_manifest_payload), encoding="utf-8"
    )

    class FakeModeRetriever:
        def __init__(self, *, settings: Settings, mode: str) -> None:
            self.mode = mode

        def retrieve(self, query: str, top_k: int) -> list[PassageHit]:
            if self.mode == "dense":
                return [PassageHit(passage_id="p1", text="Paris is the capital of France.")]
            if self.mode == "sparse":
                return [PassageHit(passage_id="p2", text="Madrid is in Spain.")]
            if self.mode == "hybrid":
                return [PassageHit(passage_id="p1", text="Paris is the capital of France.")]
            raise AssertionError("unexpected mode")

    monkeypatch.setattr(
        "src.evaluation.retrieval_eval.QdrantModeRetriever",
        FakeModeRetriever,
    )

    output_path = output_dir / "retrieval_eval.json"
    report = run_retrieval_evaluation(
        settings,
        top_k=1,
        max_queries=10,
        output_path=output_path,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert report.run_config.query_count == 1
    assert payload["run_config"]["qdrant_collection"] == "nq_passages"
    assert payload["run_config"]["sparse_identifiers"]["sparse_manifest_schema_version"] == "1"
    assert set(payload["modes"].keys()) == {"dense", "sparse", "hybrid"}
    assert payload["modes"]["dense"]["recall_at_k"] == 1.0
    assert payload["modes"]["sparse"]["recall_at_k"] == 0.0
    assert payload["modes"]["hybrid"]["mrr_at_k"] == 1.0


def test_run_retrieval_eval_rejects_non_positive_top_k(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path / "artifacts")
    PassageStore.write_jsonl([], settings.passages_path)
    with pytest.raises(ValueError, match="k must be >= 1"):
        run_retrieval_evaluation(
            settings,
            top_k=0,
            max_queries=10,
            output_path=settings.output_dir / "retrieval_eval.json",
        )
