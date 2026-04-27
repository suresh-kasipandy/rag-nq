"""Milestone 4 reranking and retrieval-result deduplication helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any, Protocol, cast

from src.models.query_schemas import DedupeMetrics, DuplicateAlias, PassageHit

_WHITESPACE_RE = re.compile(r"\s+")


class CrossEncoderLike(Protocol):
    """Minimal protocol implemented by sentence-transformers CrossEncoder."""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


@dataclass(slots=True, frozen=True)
class DedupeResult:
    """Deduped hits plus per-query counters."""

    hits: list[PassageHit]
    metrics: DedupeMetrics


def build_rerank_input(hit: PassageHit, *, context_token_budget: int) -> str:
    """Return bounded text shown to the cross-encoder reranker."""

    body = hit.context_text.strip() if hit.context_text and hit.context_text.strip() else hit.text
    tokens = body.split()
    if len(tokens) > context_token_budget:
        body = " ".join(tokens[:context_token_budget])
    if hit.title and hit.title.strip():
        return f"Title: {hit.title.strip()}\n{body}"
    return body


def rerank_hits(
    *,
    query: str,
    hits: Sequence[PassageHit],
    model: CrossEncoderLike,
    context_token_budget: int,
) -> list[PassageHit]:
    """Score hits with a cross-encoder and return a reranked copy."""

    if not hits:
        return []
    pairs = [
        (query, build_rerank_input(hit, context_token_budget=context_token_budget))
        for hit in hits
    ]
    scores = _predict_scores(model, pairs)
    ranked = [
        hit.model_copy(update={"rerank_score": score}, deep=True)
        for hit, score in zip(hits, scores, strict=True)
    ]
    ranked.sort(
        key=lambda hit: (
            -(hit.rerank_score or 0.0),
            hit.fusion_rank if hit.fusion_rank is not None else 1_000_000,
            hit.point_id,
        )
    )
    for rank, hit in enumerate(ranked, start=1):
        hit.rerank_rank = rank
    return ranked


def dedupe_hits(hits: Sequence[PassageHit], *, top_k: int) -> DedupeResult:
    """Collapse exact duplicate evidence by normalized title + context body."""

    representatives: list[PassageHit] = []
    by_key: dict[str, PassageHit] = {}
    raw_count = len(hits)

    for hit in hits:
        key = _dedupe_key(hit)
        existing = by_key.get(key)
        if existing is None:
            copied = hit.model_copy(deep=True)
            copied.duplicate_aliases = []
            by_key[key] = copied
            representatives.append(copied)
            continue
        existing.duplicate_aliases.append(_duplicate_alias(hit))

    final_hits = representatives[:top_k]
    for rank, hit in enumerate(final_hits, start=1):
        hit.dedupe_rank = rank

    unique_count = len(representatives)
    dropped = raw_count - unique_count
    metrics = DedupeMetrics(
        raw_count=raw_count,
        unique_count=unique_count,
        dedupe_drop_count=dropped,
        dedupe_drop_rate=(float(dropped) / float(raw_count)) if raw_count else 0.0,
    )
    return DedupeResult(hits=final_hits, metrics=metrics)


def _dedupe_key(hit: PassageHit) -> str:
    body = hit.context_text if hit.context_text and hit.context_text.strip() else hit.text
    return f"{_normalize(hit.title)}\x1f{_normalize(body)}"


def _normalize(value: str | None) -> str:
    if not value:
        return ""
    return _WHITESPACE_RE.sub(" ", value).strip().casefold()


def _predict_scores(
    model: CrossEncoderLike, pairs: Sequence[tuple[str, str]]
) -> list[float]:
    predict = cast(Any, model.predict)
    if _accepts_keyword(predict, "show_progress_bar"):
        return [float(score) for score in predict(pairs, show_progress_bar=False)]
    return [float(score) for score in model.predict(pairs)]


def _accepts_keyword(callable_object: Any, keyword: str) -> bool:
    try:
        parameters = signature(callable_object).parameters
    except (TypeError, ValueError):
        return False
    return keyword in parameters or any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _duplicate_alias(hit: PassageHit) -> DuplicateAlias:
    return DuplicateAlias(
        point_id=hit.point_id,
        dense_score=hit.dense_score,
        dense_rank=hit.dense_rank,
        sparse_score=hit.sparse_score,
        sparse_rank=hit.sparse_rank,
        fusion_rank=hit.fusion_rank,
        rerank_score=hit.rerank_score,
        rerank_rank=hit.rerank_rank,
    )


def build_default_cross_encoder(model_name: str) -> CrossEncoderLike:
    """Load the configured local cross-encoder lazily."""

    try:
        from sentence_transformers import CrossEncoder
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentence-transformers is required for reranking. Install project dependencies first."
        ) from exc
    return CrossEncoder(model_name)
