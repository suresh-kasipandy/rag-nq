"""Sparse index build and persistence for Milestone 1."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.ingestion.models import Passage
from src.ingestion.nq_loader import tokenized_corpus


@dataclass(slots=True)
class SparseBuildResult:
    """Sparse build metadata."""

    document_count: int


class SparseIndexer:
    """BM25 sparse index wrapper with persistence."""

    def __init__(self, bm25: Any, passage_ids: list[str]) -> None:
        self._bm25 = bm25
        self._passage_ids = passage_ids

    @classmethod
    def build(cls, passages: list[Passage]) -> tuple[SparseIndexer, SparseBuildResult]:
        """Build BM25 from normalized passages."""

        try:
            from rank_bm25 import BM25Okapi
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "rank-bm25 is required to build the sparse index. "
                "Install project dependencies first."
            ) from exc

        tokens = tokenized_corpus(passages)
        bm25 = BM25Okapi(tokens)
        index = cls(bm25=bm25, passage_ids=[p.passage_id for p in passages])
        return index, SparseBuildResult(document_count=len(passages))

    @property
    def document_count(self) -> int:
        """Return count of indexed passages."""
        return len(self._passage_ids)

    def save(self, path: Path) -> None:
        """Persist sparse index."""

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"passage_ids": self._passage_ids, "bm25": self._bm25}
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: Path) -> SparseIndexer:
        """Load sparse index from disk."""

        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return cls(bm25=payload["bm25"], passage_ids=payload["passage_ids"])
