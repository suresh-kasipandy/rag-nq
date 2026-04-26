"""Load NQ-retrieval as deterministic, passage-oriented records."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import json
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from src.config.settings import Settings
from src.ingestion.models import Passage
from src.observability.logging_setup import quiet_http_clients

# Stable namespace for synthetic passage IDs (UUID v5).
_PASSAGE_ID_NAMESPACE = uuid.UUID("018f0884-e1c7-7e4f-a9c2-0f6b2a8d4e10")


class _HFRowModel(BaseModel):
    """Validation boundary used only for first-load rows from Hugging Face."""

    model_config = ConfigDict(extra="allow")

    id: object | None = None
    text: object | None = None
    title: object | None = None
    question: object | None = None
    candidates: list[object] | None = None
    passage_types: list[object] | None = None
    long_answers: object | None = None
    document_url: object | None = None


def iter_nq_passages(settings: Settings) -> Iterator[Passage]:
    """Yield passages from the configured NQ-retrieval split without materializing the full list.

    Semantics match :func:`load_nq_passages` but stream one :class:`Passage` at a time for
    Milestone 2 silver writes and bounded-memory ingest.
    """

    yield from iter_nq_passages_from_rows(_iter_hf_rows(settings), settings=settings)


def iter_nq_passages_from_raw_artifact(settings: Settings) -> Iterator[Passage]:
    """Yield passages from local raw dataset JSONL artifact."""

    rows = iter_raw_rows(settings.raw_dataset_path)
    yield from iter_nq_passages_from_rows(rows, settings=settings)


def iter_nq_passages_from_rows(
    rows: Iterable[Mapping[str, object]], *, settings: Settings
) -> Iterator[Passage]:
    """Expand row stream into deterministic passage stream."""

    row_iter: Iterable[Mapping[str, object]] = rows
    if _ingest_progress_enabled(settings):
        row_total = _try_split_num_examples(settings.dataset_name, settings.dataset_split)
        row_iter = tqdm(
            rows,
            desc="Dataset rows",
            unit="row",
            total=row_total,
            mininterval=0.5,
        )

    yielded = 0
    for row_ordinal, row in enumerate(row_iter):
        for passage in _passages_from_row(row, row_ordinal, settings):
            yield passage
            yielded += 1
            if settings.max_passages is not None and yielded >= settings.max_passages:
                return


def load_nq_passages(settings: Settings) -> list[Passage]:
    """Load passages from the configured NQ-retrieval split.

    Rows that expose a ``candidates`` list (HuggingFace ``sentence-transformers/NQ-retrieval``)
    are expanded into one :class:`Passage` per non-empty candidate, with deterministic UUID
    passage IDs. Legacy flat rows (``id`` + ``text`` without ``candidates``) remain supported
    for tests and alternate snapshots.

    Hugging Face hub I/O temporarily lowers ``httpx`` / ``httpcore`` log levels to WARNING so
    per-chunk INFO lines do not drown real pipeline logs. Optional row ``tqdm`` uses split size
    from dataset metadata when available (percent); otherwise shows row counts only.

    This is a convenience wrapper around :func:`iter_nq_passages` for callers that still want
    a list (tests, small capped runs).
    """

    return list(iter_nq_passages(settings))


def _ingest_progress_enabled(settings: Settings) -> bool:
    if not settings.ingest_show_progress:
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return True


def _try_split_num_examples(dataset_name: str, split: str) -> int | None:
    """Return published row count for a split when the hub exposes it (no full data download)."""

    try:
        from datasets import load_dataset_builder

        builder = load_dataset_builder(dataset_name)
        splits = builder.info.splits
        if splits is None or split not in splits:
            return None
        n = splits[split].num_examples
        return int(n) if n is not None else None
    except Exception:
        return None


def _as_mapping(row: object) -> Mapping[str, object]:
    if isinstance(row, Mapping):
        return row
    raise TypeError(f"Dataset row must be a mapping, got {type(row)!r}")


def iter_raw_rows(path: Path) -> Iterator[Mapping[str, object]]:
    """Yield validated row mappings from local raw JSONL artifact."""

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("Raw dataset JSONL line must be a JSON object.")
            yield {str(k): v for k, v in payload.items()}


def _iter_hf_rows(settings: Settings) -> Iterator[Mapping[str, object]]:
    """Load and validate raw rows from Hugging Face stream."""

    with quiet_http_clients():
        raw_split = _load_dataset(
            settings.dataset_name,
            settings.dataset_split,
            streaming=settings.dataset_streaming,
        )
        for row in raw_split:
            mapping = _as_mapping(row)
            validated = _HFRowModel.model_validate(dict(mapping))
            yield validated.model_dump()


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


def _load_dataset(dataset_name: str, dataset_split: str, *, streaming: bool) -> Iterable[object]:
    from datasets import load_dataset

    # Non-streaming ``load_dataset`` may prepare every split; NQ-retrieval's dev rows can include
    # ``document_url`` while the hub-declared Features omit it, triggering CastError. Streaming
    # reads only ``dataset_split`` and sidesteps that path.
    return load_dataset(dataset_name, split=dataset_split, streaming=streaming)
