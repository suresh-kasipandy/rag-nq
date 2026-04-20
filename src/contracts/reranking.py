"""Reranker protocol (implementation deferred to later milestones)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from src.models.query_schemas import PassageHit


@runtime_checkable
class Reranker(Protocol):
    """Second-stage reranking over candidate passage hits."""

    def rerank(self, query: str, hits: Sequence[PassageHit], top_k: int) -> list[PassageHit]:
        """Return up to ``top_k`` passages reordered by the reranker."""
