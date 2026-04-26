"""Sparse index build and persistence for Milestone 1."""

from __future__ import annotations

import pickle
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.ingestion.models import Passage
from src.ingestion.nq_loader import tokenized_corpus

SparseAnalyzerName = Literal["whitespace", "regex", "regex_stem", "regex_stem_stop"]
SPARSE_ANALYZER_VERSION = "1"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


@dataclass(slots=True)
class SparseBuildResult:
    """Sparse build metadata."""

    document_count: int


@dataclass(slots=True)
class SparseCorpusStats:
    """Global corpus statistics from pass 1 over silver JSONL (Milestone 2)."""

    document_count: int
    total_tokens: int
    term_document_frequency: dict[str, int]

    @property
    def vocabulary_size(self) -> int:
        return len(self.term_document_frequency)

    @property
    def avg_doc_len(self) -> float:
        if self.document_count == 0:
            return 0.0
        return self.total_tokens / self.document_count


def _tokenize_passage_text(
    text: str, *, analyzer: SparseAnalyzerName = "regex_stem_stop"
) -> list[str]:
    """Analyze text for sparse lexical retrieval."""

    if analyzer == "whitespace":
        return text.lower().split()
    tokens = [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]
    if analyzer in {"regex_stem", "regex_stem_stop"}:
        tokens = [_stem_token(token) for token in tokens]
    if analyzer == "regex_stem_stop":
        tokens = [token for token in tokens if token not in _STOPWORDS]
    return [token for token in tokens if token]


def _stem_token(token: str) -> str:
    """Small deterministic stemmer for sparse retrieval.

    This is intentionally conservative; it captures common English suffix variants without
    changing short/entity-like tokens aggressively.
    """

    if len(token) <= 4 or token.isdigit():
        return token
    for suffix, replacement in (
        ("ies", "y"),
        ("ing", ""),
        ("edly", ""),
        ("ed", ""),
        ("ers", "er"),
        ("es", ""),
        ("s", ""),
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + replacement
    return token


def compute_sparse_corpus_stats_pass1(jsonl_path: Path) -> SparseCorpusStats:
    """First pass: scan silver JSONL and accumulate document-frequency style stats."""

    df: dict[str, int] = {}
    doc_count = 0
    total_tokens = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            passage = Passage.model_validate_json(line)
            tokens = _tokenize_passage_text(passage.text)
            doc_count += 1
            total_tokens += len(tokens)
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
    return SparseCorpusStats(
        document_count=doc_count,
        total_tokens=total_tokens,
        term_document_frequency=df,
    )


def compute_sparse_corpus_stats_from_passages(passages: Iterable[Passage]) -> SparseCorpusStats:
    """Reference stats for tests (same semantics as :func:`compute_sparse_corpus_stats_pass1`)."""

    df: dict[str, int] = {}
    doc_count = 0
    total_tokens = 0
    for passage in passages:
        tokens = _tokenize_passage_text(passage.text)
        doc_count += 1
        total_tokens += len(tokens)
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    return SparseCorpusStats(
        document_count=doc_count,
        total_tokens=total_tokens,
        term_document_frequency=df,
    )


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

    @classmethod
    def build_from_jsonl_two_pass(
        cls, jsonl_path: Path
    ) -> tuple[SparseIndexer, SparseBuildResult, SparseCorpusStats]:
        """Two-pass silver read with global stats before materializing ``BM25Okapi`` tokens.

        Pass 2 still loads all token lists into RAM because ``rank_bm25.BM25Okapi`` expects the
        full corpus in memory. Pass 1 enables streaming validation of global statistics and tests
        without building the BM25 object twice.
        """

        stats = compute_sparse_corpus_stats_pass1(jsonl_path)

        try:
            from rank_bm25 import BM25Okapi
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "rank-bm25 is required to build the sparse index. "
                "Install project dependencies first."
            ) from exc

        corpus_tokens: list[list[str]] = []
        passage_ids: list[str] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                passage = Passage.model_validate_json(line)
                corpus_tokens.append(_tokenize_passage_text(passage.text))
                passage_ids.append(passage.passage_id)

        bm25 = BM25Okapi(corpus_tokens)
        index = cls(bm25=bm25, passage_ids=passage_ids)
        return index, SparseBuildResult(document_count=len(passage_ids)), stats

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
