from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.config.settings import Settings
from src.evaluation.retrieval_eval import (
    build_eval_cases_from_index_artifact,
    build_eval_cases_from_passages_jsonl,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    run_retrieval_evaluation,
)
from src.ingestion.models import IndexChunk, Passage
from src.ingestion.passage_store import PassageStore
from src.models.query_schemas import PassageHit
from src.scripts.eval_retrieval import _quiet_dependency_logs


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
    assert cases[0].answer_texts == ["Paris"]


def test_build_eval_cases_from_index_chunks_tracks_chunk_labels(tmp_path: Path) -> None:
    index_path = tmp_path / "index_chunks.jsonl"
    chunks = [
        IndexChunk(
            chunk_id="c1",
            group_id="g1",
            text="Paris is the capital of France.",
            context_text="Europe facts. Paris is the capital of France.",
            source_row_ordinal=7,
            start_candidate_idx=2,
            end_candidate_idx=3,
            question="What is the capital of France?",
            long_answers=["Paris"],
        ),
        IndexChunk(
            chunk_id="c2",
            group_id="g1",
            text="Madrid is in Spain.",
            context_text="Madrid is in Spain.",
            source_row_ordinal=7,
            start_candidate_idx=4,
            end_candidate_idx=4,
            question="What is the capital of France?",
            long_answers=["Paris"],
        ),
    ]
    index_path.write_text(
        "\n".join(chunk.model_dump_json() for chunk in chunks) + "\n",
        encoding="utf-8",
    )

    cases = build_eval_cases_from_index_artifact(index_path, max_queries=None)

    assert len(cases) == 1
    assert cases[0].relevant_passage_ids == ["c1"]
    assert cases[0].relevant_group_ids == ["g1"]
    assert cases[0].relevant_source_row_ordinals == [7]
    assert cases[0].answer_texts == ["Paris"]
    assert len(cases[0].relevant_candidate_spans) == 1
    assert cases[0].relevant_candidate_spans[0].start_candidate_idx == 2


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
                return [PassageHit(point_id="p1", text="Paris is the capital of France.")]
            if self.mode == "sparse":
                return [PassageHit(point_id="p2", text="Madrid is in Spain.")]
            if self.mode == "hybrid":
                return [PassageHit(point_id="p1", text="Paris is the capital of France.")]
            raise AssertionError("unexpected mode")

    monkeypatch.setattr(
        "src.evaluation.retrieval_eval.QdrantModeRetriever",
        FakeModeRetriever,
    )

    output_path = output_dir / "retrieval_eval.json"
    csv_output_path = output_dir / "retrieval_eval.csv"
    report = run_retrieval_evaluation(
        settings,
        k_values=[1, 2],
        modes=["dense", "hybrid"],
        max_queries=10,
        output_path=output_path,
        csv_output_path=csv_output_path,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    csv_text = csv_output_path.read_text(encoding="utf-8")

    assert report.run_config.query_count == 1
    assert report.run_config.k_values == [1, 2]
    assert report.run_config.modes == ["dense", "hybrid"]
    assert report.run_config.relevance_contract == "answer_overlap"
    assert payload["run_config"]["qdrant_collection"] == "nq_passages"
    assert payload["run_config"]["sparse_identifiers"]["sparse_manifest_schema_version"] == "1"
    assert set(payload["modes"].keys()) == {"dense", "hybrid"}
    assert payload["modes"]["dense"]["recall_at_k"] == 1.0
    assert payload["modes"]["dense"]["metrics_by_k"]["1"]["recall_at_k"] == 1.0
    assert payload["modes"]["hybrid"]["mrr_at_k"] == 1.0
    assert "mode,k,query_count,recall_at_k" in csv_text
    assert "dense,1,1,1.0" in csv_text


def test_run_retrieval_eval_logs_mode_progress(
    tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    output_dir = tmp_path / "artifacts"
    settings = Settings(
        output_dir=output_dir,
        progress_log_every_records=1,
        progress_log_every_seconds=60.0,
    )
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
                question="What is the capital of Germany?",
                long_answers=["Berlin"],
            ),
        ],
        settings.passages_path,
    )

    class FakeModeRetriever:
        def __init__(self, *, settings: Settings, mode: str) -> None:
            del settings, mode

        def retrieve(self, query: str, top_k: int) -> list[PassageHit]:
            del top_k
            if "France" in query:
                return [PassageHit(point_id="p1", text="Paris is the capital of France.")]
            return [PassageHit(point_id="p2", text="Berlin is the capital of Germany.")]

    monkeypatch.setattr(
        "src.evaluation.retrieval_eval.QdrantModeRetriever",
        FakeModeRetriever,
    )

    caplog.set_level(logging.INFO, logger="src.evaluation.retrieval_eval")
    run_retrieval_evaluation(
        settings,
        k_values=[1],
        modes=["dense"],
        max_queries=10,
        output_path=output_dir / "retrieval_eval.json",
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("[retrieval_eval_cases] start records=0" in msg for msg in messages)
    assert any("[retrieval_eval_cases] complete records=2" in msg for msg in messages)
    assert any("[retrieval_eval] retrieval eval start" in msg for msg in messages)
    assert any("[retrieval_eval_mode] start queries=0/2" in msg for msg in messages)
    assert any("[retrieval_eval_mode] progress queries=1/2" in msg for msg in messages)
    assert any("[retrieval_eval_mode] complete queries=2/2" in msg for msg in messages)


def test_eval_cli_quiets_dependency_logs() -> None:
    logger_names = [
        "httpx",
        "httpcore",
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
        "qdrant_client",
        "src.retrieval.qdrant_retrievers",
    ]
    previous_levels = {
        logger_name: logging.getLogger(logger_name).level for logger_name in logger_names
    }
    try:
        for logger_name in logger_names:
            logging.getLogger(logger_name).setLevel(logging.INFO)

        _quiet_dependency_logs()

        assert all(
            logging.getLogger(logger_name).level == logging.WARNING
            for logger_name in logger_names
        )
    finally:
        for logger_name, level in previous_levels.items():
            logging.getLogger(logger_name).setLevel(level)


def test_run_retrieval_eval_can_use_candidate_span_overlap(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "artifacts"
    settings = Settings(output_dir=output_dir)
    settings.index_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunk = IndexChunk(
        chunk_id="c1",
        group_id="g1",
        text="Paris is the capital of France.",
        context_text="Paris is the capital of France.",
        source_row_ordinal=7,
        start_candidate_idx=2,
        end_candidate_idx=4,
        question="What is the capital of France?",
        long_answers=["Paris"],
    )
    settings.index_chunks_path.write_text(chunk.model_dump_json() + "\n", encoding="utf-8")

    class FakeModeRetriever:
        def __init__(self, *, settings: Settings, mode: str) -> None:
            del settings, mode

        def retrieve(self, query: str, top_k: int) -> list[PassageHit]:
            del query, top_k
            return [
                PassageHit(
                    point_id="other",
                    text="overlapping candidate",
                    source_row_ordinal=7,
                    start_candidate_idx=3,
                    end_candidate_idx=5,
                )
            ]

    monkeypatch.setattr(
        "src.evaluation.retrieval_eval.QdrantModeRetriever",
        FakeModeRetriever,
    )

    report = run_retrieval_evaluation(
        settings,
        k_values=[1],
        modes=["dense"],
        relevance_contract="candidate_span_overlap",
        max_queries=None,
        output_path=output_dir / "retrieval_eval.json",
    )

    assert report.modes["dense"].recall_at_k == 1.0
    assert report.modes["dense"].mrr_at_k == 1.0


def test_answer_overlap_contract_matches_answer_bearing_retrieved_text(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "artifacts"
    settings = Settings(output_dir=output_dir)
    PassageStore.write_jsonl(
        [
            Passage(
                passage_id="gold",
                text="Paris is the capital of France.",
                question="What is the capital of France?",
                long_answers=["Paris"],
            )
        ],
        settings.passages_path,
    )

    class FakeModeRetriever:
        def __init__(self, *, settings: Settings, mode: str) -> None:
            del settings, mode

        def retrieve(self, query: str, top_k: int) -> list[PassageHit]:
            del query, top_k
            return [PassageHit(point_id="different-point", text="The answer is Paris.")]

    monkeypatch.setattr(
        "src.evaluation.retrieval_eval.QdrantModeRetriever",
        FakeModeRetriever,
    )

    report = run_retrieval_evaluation(
        settings,
        k_values=[1],
        modes=["dense"],
        relevance_contract="answer_overlap",
        max_queries=None,
        output_path=output_dir / "retrieval_eval.json",
    )

    assert report.modes["dense"].recall_at_k == 1.0


def test_run_retrieval_eval_rejects_missing_explicit_corpus(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path / "artifacts")

    with pytest.raises(FileNotFoundError, match="Evaluation corpus does not exist"):
        run_retrieval_evaluation(
            settings,
            k_values=[1],
            modes=["dense"],
            max_queries=None,
            output_path=settings.output_dir / "retrieval_eval.json",
            corpus_path=tmp_path / "missing.jsonl",
        )


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
