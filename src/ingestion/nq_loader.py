"""Load NQ-retrieval as deterministic, passage-oriented records."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping

from src.config.settings import Settings
from src.ingestion.models import Passage

# Stable namespace for synthetic passage IDs (UUID v5).
_PASSAGE_ID_NAMESPACE = uuid.UUID("018f0884-e1c7-7e4f-a9c2-0f6b2a8d4e10")


def load_nq_passages(settings: Settings) -> list[Passage]:
    """Load passages from the configured NQ-retrieval split.

    Rows that expose a ``candidates`` list (HuggingFace ``sentence-transformers/NQ-retrieval``)
    are expanded into one :class:`Passage` per non-empty candidate, with deterministic UUID
    passage IDs. Legacy flat rows (``id`` + ``text`` without ``candidates``) remain supported
    for tests and alternate snapshots.
    """

    raw_split = _load_dataset(settings.dataset_name, settings.dataset_split)

    passages: list[Passage] = []
    for row_ordinal, row in enumerate(raw_split):
        row_mapping = _as_mapping(row)
        for passage in _passages_from_row(row_mapping, row_ordinal, settings):
            passages.append(passage)
            if settings.max_passages is not None and len(passages) >= settings.max_passages:
                return passages
    return passages


def _as_mapping(row: object) -> Mapping[str, object]:
    if isinstance(row, Mapping):
        return row
    raise TypeError(f"Dataset row must be a mapping, got {type(row)!r}")


def _passages_from_row(
    row: Mapping[str, object], row_ordinal: int, settings: Settings
) -> list[Passage]:
    if isinstance(row.get("candidates"), list):
        return _expand_candidate_passages(row, row_ordinal, settings)
    return [_legacy_row_to_passage(row, settings)]


def _expand_candidate_passages(
    row: Mapping[str, object], row_ordinal: int, settings: Settings
) -> list[Passage]:
    raw_candidates = row["candidates"]
    if not isinstance(raw_candidates, list):
        raise TypeError("candidates must be a list")
    candidates: list[object] = raw_candidates
    passage_types = row.get("passage_types")
    type_list: list[object] | None = passage_types if isinstance(passage_types, list) else None

    title = _optional_str(row.get("title"))
    question = _optional_str(row.get("question"))
    document_url = _optional_str(row.get("document_url"))
    long_answers = _normalize_long_answers(row.get("long_answers"))

    out: list[Passage] = []
    for cand_idx, raw_cand in enumerate(candidates):
        text = str(raw_cand).strip()
        if not text:
            continue
        passage_type: str | None = None
        if type_list is not None and cand_idx < len(type_list):
            passage_type = _optional_str(type_list[cand_idx])
        passage_id = _synthetic_passage_id(settings, row_ordinal, cand_idx)
        out.append(
            Passage(
                passage_id=passage_id,
                text=text,
                source=title,
                title=title,
                question=question,
                passage_type=passage_type,
                document_url=document_url,
                long_answers=long_answers,
            )
        )
    return out


def _legacy_row_to_passage(row: Mapping[str, object], settings: Settings) -> Passage:
    """Convert one flat dataset row into a validated `Passage`."""

    raw_id = row.get(settings.dataset_passage_id_field)
    raw_text = row.get(settings.dataset_passage_text_field)
    raw_source = row.get(settings.dataset_source_field)

    if raw_id is None:
        raise ValueError(f"Missing passage id field: {settings.dataset_passage_id_field}")
    if raw_text is None:
        raise ValueError(f"Missing passage text field: {settings.dataset_passage_text_field}")

    text = str(raw_text).strip()
    if not text:
        raise ValueError("Passage text is empty after stripping")

    source = None if raw_source is None else str(raw_source).strip() or None
    title = source
    return Passage(passage_id=str(raw_id), text=text, source=source, title=title)


def _synthetic_passage_id(settings: Settings, row_ordinal: int, candidate_index: int) -> str:
    key = (
        f"{settings.dataset_name}\x1f{settings.dataset_split}\x1f{row_ordinal}\x1f{candidate_index}"
    )
    return str(uuid.uuid5(_PASSAGE_ID_NAMESPACE, key))


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _normalize_long_answers(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str) and not raw.strip():
        return []
    return [str(raw)]


def tokenized_corpus(passages: Iterable[Passage]) -> list[list[str]]:
    """Build BM25 token lists from passage text."""

    return [passage.text.lower().split() for passage in passages]


def _load_dataset(dataset_name: str, dataset_split: str) -> Iterable[object]:
    from datasets import load_dataset

    return load_dataset(dataset_name, split=dataset_split)
