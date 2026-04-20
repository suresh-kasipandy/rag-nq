"""Retriever protocol (implementation deferred to later milestones)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.models.query_schemas import PassageHit


@runtime_checkable
class Retriever(Protocol):
    """Dense, sparse, or hybrid retrieval behind a single interface."""

    def retrieve(self, query: str, top_k: int) -> list[PassageHit]:
        """Return up to ``top_k`` passages ranked by this retriever.

        Aligns with :class:`~src.models.query_schemas.QueryRequest` ``top_k``.
        """
