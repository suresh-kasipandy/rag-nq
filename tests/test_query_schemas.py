from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.query_schemas import (
    Citation,
    GroundedAnswer,
    PassageHit,
    QueryRequest,
    QueryResponse,
)


def test_passage_hit_accepts_optional_scores() -> None:
    hit = PassageHit(
        passage_id="p1",
        text="hello",
        dense_score=0.9,
        dense_rank=1,
        sparse_rank=2,
    )
    assert hit.sparse_score is None


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
    assert r.grounded is None
