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
    SupportedClaim,
    UnsupportedClaim,
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
    evidence = PassageHit(point_id="a", text="Evidence text")
    ga = GroundedAnswer(
        answer="Based on sources.",
        citations=[Citation(point_id="a"), Citation(point_id="b")],
        abstained=False,
        supported_claims=[SupportedClaim(claim="A is supported.", point_ids=["a"])],
        unsupported_claims=[UnsupportedClaim(claim="C is unsupported.", reason="No evidence.")],
        supporting_point_ids=["a", "b"],
        supporting_evidence=[evidence],
    )
    assert len(ga.citations) == 2
    assert ga.supported_claims[0].point_ids == ["a"]
    assert ga.unsupported_claims[0].reason == "No evidence."
    assert ga.supporting_evidence[0].point_id == "a"


def test_supported_claim_requires_point_ids() -> None:
    with pytest.raises(ValidationError):
        SupportedClaim(claim="Unsupported shape.", point_ids=[])


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
