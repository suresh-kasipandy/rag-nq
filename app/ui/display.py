"""Presentation helpers shared by the Streamlit app and tests."""

from __future__ import annotations

from src.models.query_schemas import GroundedAnswer, PassageHit

DEFAULT_ABSTENTION_MESSAGE = "I don't have enough retrieved evidence to answer that reliably."


def answer_display_text(grounded: GroundedAnswer | None) -> str:
    """Return user-facing answer text without weakening the grounded-answer contract."""

    if grounded is None:
        return ""
    if grounded.abstained:
        if grounded.abstention_reason:
            return f"{DEFAULT_ABSTENTION_MESSAGE}\n\nReason: {grounded.abstention_reason}"
        return DEFAULT_ABSTENTION_MESSAGE
    return grounded.answer.strip()


def hit_rank_summary(hit: PassageHit) -> str:
    """Compact rank summary for evidence cards."""

    ranks: list[str] = []
    if hit.dense_rank is not None:
        ranks.append(f"dense #{hit.dense_rank}")
    if hit.sparse_rank is not None:
        ranks.append(f"sparse #{hit.sparse_rank}")
    if hit.fusion_rank is not None:
        ranks.append(f"fusion #{hit.fusion_rank}")
    if hit.rerank_rank is not None:
        ranks.append(f"rerank #{hit.rerank_rank}")
    if hit.dedupe_rank is not None:
        ranks.append(f"dedupe #{hit.dedupe_rank}")
    return " | ".join(ranks) if ranks else "rank metadata unavailable"


def hit_title(hit: PassageHit) -> str:
    """Return a stable evidence card title."""

    prefix = hit.title.strip() if hit.title and hit.title.strip() else "Untitled evidence"
    return f"{prefix} ({hit.point_id})"
