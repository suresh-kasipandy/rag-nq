from __future__ import annotations

from collections.abc import Sequence

from src.contracts import Generator, Reranker, Retriever
from src.models.query_schemas import GroundedAnswer, PassageHit


class _DummyRetriever:
    def retrieve(self, query: str, top_k: int) -> list[PassageHit]:
        return [
            PassageHit(passage_id="1", text="t", source=None),
        ]


class _DummyReranker:
    def rerank(self, query: str, hits: Sequence[PassageHit], top_k: int) -> list[PassageHit]:
        return list(hits)[:top_k]


class _DummyGenerator:
    def generate(self, query: str, hits: Sequence[PassageHit]) -> GroundedAnswer:
        return GroundedAnswer(abstained=True)


def test_protocols_runtime_checkable() -> None:
    assert isinstance(_DummyRetriever(), Retriever)
    assert isinstance(_DummyReranker(), Reranker)
    assert isinstance(_DummyGenerator(), Generator)


def test_dummy_retriever_contract() -> None:
    r: Retriever = _DummyRetriever()
    out = r.retrieve("q", top_k=3)
    assert len(out) == 1
    assert out[0].passage_id == "1"
