"""Generator protocol (implementation deferred to later milestones)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from src.models.query_schemas import GroundedAnswer, PassageHit


@runtime_checkable
class Generator(Protocol):
    """Grounded answer generation from retrieved passages."""

    def generate(self, query: str, hits: Sequence[PassageHit]) -> GroundedAnswer:
        """Produce an answer with citations; may abstain if evidence is insufficient."""
