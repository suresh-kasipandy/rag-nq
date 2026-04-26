from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.query_schemas import (
    Citation,
    DedupeMetrics,
    DuplicateAlias,
    GroundedAnswer,
    PassageHit,
    QueryRequest,
    QueryResponse,
    RetrievalMetrics,
    RetrievalStageTimings,
)


def test_passage_hit_accepts_optional_scores() -> None:
    hit = PassageHit(
        point_id="p1",
        text="hello",
        dense_score=0.9,
        dense_rank=1,
        sparse_rank=2,
        dedupe_rank=1,
        duplicate_aliases=[DuplicateAlias(point_id="p2", fusion_rank=2)],
    )
    assert hit.sparse_score is None
    assert hit.duplicate_aliases[0].point_id == "p2"


def test_query_request_bounds() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="x", top_k=0)
    with pytest.raises(ValidationError):
        QueryRequest(query="", top_k=5)


def test_grounded_answer_citation_ids() -> None:
    ga = GroundedAnswer(
        answer="Based on sources.",
        citations=[Citation(passage_id="a"), Citation(passage_id="b")],
        abstained=False,
        supporting_passage_ids=["a", "b"],
    )
    assert len(ga.citations) == 2


def test_query_response_optional_sections() -> None:
    r = QueryResponse(query="q")
    assert r.retrieved_passages is None
    assert r.retrieval_metrics is None
    assert r.grounded is None


def test_query_response_accepts_retrieval_metrics() -> None:
    r = QueryResponse(
        query="q",
        retrieval_metrics=RetrievalMetrics(
            dedupe=DedupeMetrics(
                raw_count=5,
                unique_count=3,
                dedupe_drop_count=2,
                dedupe_drop_rate=0.4,
            ),
            timings=RetrievalStageTimings(
                retrieve_seconds=0.1,
                rerank_seconds=0.2,
                total_seconds=0.3,
            ),
        ),
    )
    assert r.retrieval_metrics is not None
    assert r.retrieval_metrics.dedupe is not None
    assert r.retrieval_metrics.dedupe.dedupe_drop_rate == 0.4
    assert r.retrieval_metrics.timings is not None
    assert r.retrieval_metrics.timings.rerank_seconds == 0.2
