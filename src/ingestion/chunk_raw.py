"""Build ``index_chunks.jsonl`` from raw NQ rows (Milestone 3.6)."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from src.config.settings import Settings
from src.ingestion.chunk_store import flatten_index_chunk_group
from src.ingestion.ingest_raw import (
    load_raw_manifest,
    raw_inputs_match_manifest,
    run_raw_ingest,
)
from src.ingestion.models import (
    CHUNK_MANIFEST_SCHEMA_VERSION,
    INDEX_CHUNK_SCHEMA_VERSION,
    RAW_DATASET_SCHEMA_VERSION,
    ChunkManifest,
    IndexChunk,
    IndexChunkGroup,
    IndexChunkItem,
    IndexText,
)
from src.ingestion.nq_loader import iter_raw_rows
from src.observability.logging_setup import get_stage_logger
from src.observability.progress import ProgressTicker

LOGGER = get_stage_logger(__name__)

_CHUNK_ID_NAMESPACE = uuid.UUID("5eb87ec6-06da-4ad2-8a61-dff55c8b11ee")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_BOILERPLATE_EXACT = {
    "vte",
    "v t e",
    "v\nt\ne",
    "external links",
    "references",
    "see also",
    "notes",
}
_LIST_OR_TABLE_TYPES = {"list", "table"}
_HEADER_TYPES = {"list_definition", "section", "header"}
_INTRO_MARKERS = ("including", "include", "includes", "such as", "the following", "for example")

CandidateRole = Literal[
    "paragraph",
    "table_parent",
    "table_child",
    "list_parent",
    "list_item",
    "header",
    "intro",
    "boilerplate",
    "duplicate_child",
    "short_fact_candidate",
]


@dataclass(slots=True)
class CandidateAnnotation:
    """Internal candidate annotation used between preprocessing and chunk construction."""

    index: int
    text: str
    normalized: str
    passage_type: str | None
    token_count: int
    role: CandidateRole
    parent_candidate_idx: int | None = None


@dataclass(slots=True)
class _ChunkBuildState:
    rows_processed: int = 0
    chunks_emitted: int = 0
    candidates: int = 0
    boilerplate: int = 0
    duplicate_child: int = 0
    tiny_candidates: int = 0
    tiny_suppressed: int = 0
    short_facts: int = 0
    context_expanded: int = 0
    role_counts: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.role_counts is None:
            self.role_counts = Counter()


def count_jsonl_lines(path: Path) -> int:
    """Count non-empty JSONL lines."""

    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_chunk_manifest(path: Path) -> ChunkManifest | None:
    """Return parsed chunk manifest, or ``None`` if missing/invalid."""

    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return ChunkManifest.model_validate_json(handle.read())
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def should_skip_chunk_ingest(settings: Settings, *, force: bool = False) -> bool:
    """Return True when existing chunks and manifest match current inputs."""

    if force:
        return False
    if not settings.index_chunks_path.is_file() or not settings.chunk_manifest_path.is_file():
        return False
    manifest = load_chunk_manifest(settings.chunk_manifest_path)
    if manifest is None:
        return False
    raw_manifest = load_raw_manifest(settings.raw_manifest_path)
    if raw_manifest is None or not raw_inputs_match_manifest(settings, raw_manifest):
        return False
    expected = (
        manifest.schema_version == CHUNK_MANIFEST_SCHEMA_VERSION
        and manifest.chunk_schema_version == INDEX_CHUNK_SCHEMA_VERSION
        and manifest.dataset_name == settings.dataset_name
        and manifest.dataset_split == settings.dataset_split
        and manifest.max_chunk_rows == settings.max_chunk_rows
        and manifest.raw_schema_version == RAW_DATASET_SCHEMA_VERSION
        and manifest.raw_row_count == raw_manifest.row_count
        and manifest.min_tokens_soft == settings.chunk_min_tokens_soft
        and manifest.min_tokens_hard == settings.chunk_min_tokens_hard
        and manifest.target_tokens == settings.chunk_target_tokens
        and manifest.max_tokens == settings.chunk_max_tokens
        and manifest.context_text_token_cap == settings.chunk_context_text_token_cap
    )
    if not expected:
        return False
    try:
        return count_jsonl_lines(settings.index_chunks_path) == manifest.line_count
    except OSError:
        return False


def run_chunk_ingest(settings: Settings, *, force: bool = False) -> tuple[ChunkManifest, bool]:
    """Write ``index_chunks.jsonl`` + chunk manifest from raw rows."""

    raw_manifest, _ = run_raw_ingest(settings, force=Settings.force_raw_ingest_from_env())
    if should_skip_chunk_ingest(settings, force=force):
        manifest = load_chunk_manifest(settings.chunk_manifest_path)
        assert manifest is not None
        LOGGER.info(
            "skip chunk ingest: chunk_manifest matches (%s groups, %s chunks)",
            manifest.line_count,
            manifest.chunk_count,
            extra={"stage": "chunk"},
        )
        return manifest, True

    tmp_jsonl = settings.index_chunks_path.with_suffix(settings.index_chunks_path.suffix + ".tmp")
    tmp_manifest = settings.chunk_manifest_path.with_suffix(
        settings.chunk_manifest_path.suffix + ".tmp"
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    ticker = ProgressTicker(
        logger=LOGGER,
        stage="chunk",
        label="rows",
        total=(
            min(raw_manifest.row_count, settings.max_chunk_rows)
            if settings.max_chunk_rows is not None
            else raw_manifest.row_count
        ),
        every_items=settings.progress_log_every_records,
        every_seconds=settings.progress_log_every_seconds,
    )
    ticker.start(
        raw_path=settings.raw_dataset_path,
        output=settings.index_chunks_path,
        max_chunk_rows=settings.max_chunk_rows,
        min_tokens_soft=settings.chunk_min_tokens_soft,
        max_tokens=settings.chunk_max_tokens,
    )
    state = _ChunkBuildState()
    group_count = 0
    with tmp_jsonl.open("w", encoding="utf-8") as handle:
        groups = iter_index_chunk_groups_from_raw_artifact(
            settings,
            state=state,
            ticker=ticker,
        )
        for group in groups:
            handle.write(group.model_dump_json(exclude_none=True))
            handle.write("\n")
            group_count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_jsonl, settings.index_chunks_path)

    manifest = ChunkManifest(
        schema_version=CHUNK_MANIFEST_SCHEMA_VERSION,
        chunk_schema_version=INDEX_CHUNK_SCHEMA_VERSION,
        dataset_name=settings.dataset_name,
        dataset_split=settings.dataset_split,
        max_passages=settings.max_passages,
        max_chunk_rows=settings.max_chunk_rows,
        line_count=group_count,
        raw_schema_version=raw_manifest.schema_version,
        raw_row_count=raw_manifest.row_count,
        created_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
        chunk_count=state.chunks_emitted,
        min_tokens_soft=settings.chunk_min_tokens_soft,
        min_tokens_hard=settings.chunk_min_tokens_hard,
        target_tokens=settings.chunk_target_tokens,
        max_tokens=settings.chunk_max_tokens,
        context_text_token_cap=settings.chunk_context_text_token_cap,
        candidate_count=state.candidates,
        boilerplate_count=state.boilerplate,
        duplicate_child_count=state.duplicate_child,
        tiny_candidate_count=state.tiny_candidates,
        tiny_suppressed_count=state.tiny_suppressed,
        short_fact_count=state.short_facts,
        context_expanded_count=state.context_expanded,
        role_counts=dict(state.role_counts or {}),
    )
    with tmp_manifest.open("w", encoding="utf-8") as handle:
        handle.write(manifest.model_dump_json(indent=2, exclude_none=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_manifest, settings.chunk_manifest_path)
    ticker.finish(
        state.rows_processed,
        groups=group_count,
        chunks=state.chunks_emitted,
        candidates=state.candidates,
        tiny=state.tiny_candidates,
        tiny_suppressed=state.tiny_suppressed,
        duplicate_child=state.duplicate_child,
        boilerplate=state.boilerplate,
        context_expanded=state.context_expanded,
        output=settings.index_chunks_path,
        manifest=settings.chunk_manifest_path,
    )
    return manifest, False


def iter_index_chunks_from_raw_artifact(
    settings: Settings,
    *,
    state: _ChunkBuildState | None = None,
    ticker: ProgressTicker | None = None,
) -> Iterator[IndexChunk]:
    """Yield deterministic index chunks from local raw dataset JSONL."""

    for group in iter_index_chunk_groups_from_raw_artifact(settings, state=state, ticker=ticker):
        yield from flatten_index_chunk_group(group)


def iter_index_chunk_groups_from_raw_artifact(
    settings: Settings,
    *,
    state: _ChunkBuildState | None = None,
    ticker: ProgressTicker | None = None,
) -> Iterator[IndexChunkGroup]:
    """Yield grouped index chunks from local raw dataset JSONL."""

    build_state = state or _ChunkBuildState()
    for row_ordinal, row in enumerate(iter_raw_rows(settings.raw_dataset_path)):
        if settings.max_chunk_rows is not None and row_ordinal >= settings.max_chunk_rows:
            break
        build_state.rows_processed = row_ordinal + 1
        annotations = annotate_row_candidates(row, row_ordinal=row_ordinal, settings=settings)
        for annotation in annotations:
            build_state.candidates += 1
            if annotation.token_count <= settings.chunk_min_tokens_soft:
                build_state.tiny_candidates += 1
            if annotation.role == "boilerplate":
                build_state.boilerplate += 1
            if annotation.role == "duplicate_child":
                build_state.duplicate_child += 1
            assert build_state.role_counts is not None
            build_state.role_counts[annotation.role] += 1
        chunks = build_chunks_for_row(row, row_ordinal, annotations, settings, build_state)
        build_state.chunks_emitted += len(chunks)
        if ticker is not None:
            ticker.tick(
                row_ordinal + 1,
                chunks=build_state.chunks_emitted,
                candidates=build_state.candidates,
                tiny=build_state.tiny_candidates,
                tiny_suppressed=build_state.tiny_suppressed,
                duplicate_child=build_state.duplicate_child,
                boilerplate=build_state.boilerplate,
                context_expanded=build_state.context_expanded,
            )
        if chunks:
            yield _group_from_chunks(row, row_ordinal, chunks, settings)


def annotate_row_candidates(
    row: Mapping[str, object], *, row_ordinal: int, settings: Settings
) -> list[CandidateAnnotation]:
    """Preprocess and role-tag candidates for one raw row."""

    raw_candidates = row.get("candidates")
    if not isinstance(raw_candidates, list):
        return []
    raw_types = row.get("passage_types")
    type_list = raw_types if isinstance(raw_types, list) else []

    annotations: list[CandidateAnnotation] = []
    for idx, raw_candidate in enumerate(raw_candidates):
        text = normalize_text(raw_candidate)
        if not text:
            continue
        passage_type = str(type_list[idx]).strip() if idx < len(type_list) else None
        passage_type = passage_type or None
        tokens = token_count(text)
        role = _initial_role(text, passage_type, tokens)
        annotations.append(
            CandidateAnnotation(
                index=idx,
                text=text,
                normalized=text.casefold(),
                passage_type=passage_type,
                token_count=tokens,
                role=role,
            )
        )

    for pos, annotation in enumerate(annotations):
        if (
            annotation.role == "boilerplate"
            or annotation.token_count > settings.chunk_min_tokens_hard
        ):
            continue
        parent = _find_containing_parent(annotations, pos)
        if parent is not None:
            annotation.role = "duplicate_child"
            annotation.parent_candidate_idx = parent.index
    return annotations


def build_chunks_for_row(
    row: Mapping[str, object],
    row_ordinal: int,
    annotations: list[CandidateAnnotation],
    settings: Settings,
    state: _ChunkBuildState,
) -> list[IndexChunk]:
    """Build chunks from preprocessed candidates for one row."""

    chunks: list[IndexChunk] = []
    i = 0
    while i < len(annotations):
        current = annotations[i]
        if current.role in {"boilerplate", "duplicate_child"}:
            i += 1
            continue
        span = [current]
        total = current.token_count
        while total < settings.chunk_min_tokens_soft and i + len(span) < len(annotations):
            nxt = annotations[i + len(span)]
            if nxt.role in {"boilerplate", "duplicate_child"}:
                span.append(nxt)
                continue
            if not _compatible(span[-1], nxt):
                break
            next_total = total + nxt.token_count
            if next_total > settings.chunk_max_tokens:
                break
            span.append(nxt)
            total = next_total
        emitted = _emit_span(row, row_ordinal, span, annotations, settings, state)
        chunks.extend(emitted)
        i += len(span)
    return _resolve_trailing_tiny_chunks(chunks, settings, state)


def normalize_text(value: object) -> str:
    """Normalize candidate text before role tagging/chunking."""

    text = str(value).replace("\u00a0", " ").strip()
    return _WHITESPACE_RE.sub(" ", text)


def token_count(text: object) -> int:
    """Approximate token count for chunking decisions."""

    return len(re.findall(r"\w+", str(text)))


def _initial_role(text: str, passage_type: str | None, tokens: int) -> CandidateRole:
    normalized = text.casefold()
    if normalized in _BOILERPLATE_EXACT or len(normalized) <= 2:
        return "boilerplate"
    if _looks_like_intro(text):
        return "intro"
    if passage_type == "table":
        return "table_parent" if tokens >= 40 else "table_child"
    if passage_type == "list":
        return "list_parent" if tokens >= 40 else "list_item"
    if passage_type in _HEADER_TYPES:
        return "header"
    if _is_self_contained_fact(text, tokens):
        return "short_fact_candidate" if tokens < 60 else "paragraph"
    return "paragraph"


def _find_containing_parent(
    annotations: list[CandidateAnnotation], position: int
) -> CandidateAnnotation | None:
    child = annotations[position]
    if len(child.normalized) <= 5:
        return None
    lo = max(0, position - 25)
    hi = min(len(annotations), position + 26)
    for candidate in annotations[lo:hi]:
        if candidate.index == child.index or candidate.token_count < 8:
            continue
        if child.normalized in candidate.normalized:
            return candidate
    return None


def _emit_span(
    row: Mapping[str, object],
    row_ordinal: int,
    span: list[CandidateAnnotation],
    all_annotations: list[CandidateAnnotation],
    settings: Settings,
    state: _ChunkBuildState,
) -> list[IndexChunk]:
    usable = [
        candidate
        for candidate in span
        if candidate.role not in {"boilerplate", "duplicate_child"}
    ]
    if not usable:
        return []
    text = " ".join(candidate.text for candidate in usable).strip()
    if not text:
        return []
    tokens = token_count(text)
    if tokens < settings.chunk_min_tokens_hard and not _is_self_contained_fact(text, tokens):
        state.tiny_suppressed += 1
        return []
    if tokens > settings.chunk_max_tokens:
        return _split_oversized_span(row, row_ordinal, usable, all_annotations, settings, state)
    chunk_kind = "short_fact" if tokens < settings.chunk_min_tokens_soft else _chunk_kind(usable)
    if chunk_kind == "short_fact":
        state.short_facts += 1
    return [
        _make_chunk(
            row,
            row_ordinal,
            usable,
            all_annotations,
            text,
            chunk_kind,
            settings,
            state,
        )
    ]


def _split_oversized_span(
    row: Mapping[str, object],
    row_ordinal: int,
    span: list[CandidateAnnotation],
    all_annotations: list[CandidateAnnotation],
    settings: Settings,
    state: _ChunkBuildState,
) -> list[IndexChunk]:
    text = " ".join(candidate.text for candidate in span)
    parts = _split_text_by_sentences(text, settings.chunk_target_tokens, settings.chunk_max_tokens)
    chunks: list[IndexChunk] = []
    for part_no, part in enumerate(parts):
        if token_count(part) < settings.chunk_min_tokens_hard:
            state.tiny_suppressed += 1
            continue
        chunks.append(
            _make_chunk(
                row,
                row_ordinal,
                span,
                all_annotations,
                part,
                f"split_{part_no + 1}",
                settings,
                state,
            )
        )
    return chunks


def _split_text_by_sentences(text: str, target_tokens: int, max_tokens: int) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text)
    out: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and token_count(" ".join(current + [sentence])) > max_tokens:
            out.append(" ".join(current))
            overlap = (
                current[-1:]
                if token_count(current[-1]) <= max(10, target_tokens // 5)
                else []
            )
            current = overlap + [sentence]
        else:
            current.append(sentence)
    if current:
        out.append(" ".join(current))
    return out


def _make_chunk(
    row: Mapping[str, object],
    row_ordinal: int,
    span: list[CandidateAnnotation],
    all_annotations: list[CandidateAnnotation],
    text: str,
    chunk_kind: str,
    settings: Settings,
    state: _ChunkBuildState,
) -> IndexChunk:
    context_text, parent_idx = _context_text_for_span(span, all_annotations, text, settings)
    if context_text != text:
        state.context_expanded += 1
    start = min(candidate.index for candidate in span)
    end = max(candidate.index for candidate in span)
    title = _optional_str(row.get("title"))
    key = "\x1f".join(
        [
            settings.dataset_name,
            settings.dataset_split,
            str(row_ordinal),
            str(start),
            str(end),
            text[:120],
        ]
    )
    return IndexChunk(
        chunk_id=str(uuid.uuid5(_CHUNK_ID_NAMESPACE, key)),
        group_id=f"{settings.dataset_name}:{settings.dataset_split}:{row_ordinal}",
        text=text,
        context_text=context_text,
        source_row_ordinal=row_ordinal,
        start_candidate_idx=start,
        end_candidate_idx=end,
        parent_candidate_idx=parent_idx,
        passage_types=[candidate.passage_type for candidate in span if candidate.passage_type],
        title=title,
        source=title,
        question=_optional_str(row.get("question")),
        document_url=_optional_str(row.get("document_url")),
        chunk_kind=chunk_kind,
        token_count=token_count(text),
        context_token_count=token_count(context_text),
        long_answers=_long_answers_from_row(row),
    )


def _chunk_items_from_chunks(
    chunks: list[IndexChunk],
    raw_texts: dict[int, str],
    passage_types: dict[int, str | None],
    text_pool: _TextPool,
) -> list[IndexChunkItem]:
    items: list[IndexChunkItem] = []
    for chunk in chunks:
        text_idxs = _refs_for_materialized_text(
            chunk.text,
            raw_texts=raw_texts,
            passage_types=passage_types,
            start_idx=chunk.start_candidate_idx,
            end_idx=chunk.end_candidate_idx,
            text_pool=text_pool,
        )
        context_idxs = _refs_for_materialized_text(
            chunk.context_text,
            raw_texts=raw_texts,
            passage_types=passage_types,
            start_idx=chunk.start_candidate_idx,
            end_idx=chunk.end_candidate_idx,
            text_pool=text_pool,
        )
        items.append(
            IndexChunkItem(
                chunk_id=chunk.chunk_id,
                text_idxs=text_idxs,
                context_idxs=context_idxs,
                start_candidate_idx=chunk.start_candidate_idx,
                end_candidate_idx=chunk.end_candidate_idx,
                parent_candidate_idx=chunk.parent_candidate_idx,
                passage_types=chunk.passage_types,
                chunk_kind=chunk.chunk_kind,
                token_count=chunk.token_count,
                context_token_count=chunk.context_token_count,
            )
        )
    return items


@dataclass(slots=True)
class _TextPool:
    texts: list[IndexText] = field(default_factory=list)
    _by_key: dict[tuple[str, int | None], int] = field(default_factory=dict)

    def add(self, *, text: str, raw_idx: int | None, passage_type: str | None = None) -> int:
        key = (text, raw_idx)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing
        text_idx = len(self.texts)
        self.texts.append(
            IndexText(
                text_idx=text_idx,
                raw_idx=raw_idx,
                text=text,
                passage_type=passage_type,
            )
        )
        self._by_key[key] = text_idx
        return text_idx


def _raw_candidate_texts(
    row: Mapping[str, object],
) -> tuple[dict[int, str], dict[int, str | None]]:
    raw_candidates = row.get("candidates")
    if not isinstance(raw_candidates, list):
        return {}, {}
    raw_types = row.get("passage_types")
    type_list = raw_types if isinstance(raw_types, list) else []
    texts: dict[int, str] = {}
    passage_types: dict[int, str | None] = {}
    for idx, raw_candidate in enumerate(raw_candidates):
        text = normalize_text(raw_candidate)
        if not text:
            continue
        passage_type = str(type_list[idx]).strip() if idx < len(type_list) else None
        texts[idx] = text
        passage_types[idx] = passage_type or None
    return texts, passage_types


def _refs_for_materialized_text(
    text: str,
    *,
    raw_texts: dict[int, str],
    passage_types: dict[int, str | None],
    start_idx: int,
    end_idx: int,
    text_pool: _TextPool,
) -> list[int]:
    span_candidates = [
        (idx, raw_texts[idx])
        for idx in range(start_idx, end_idx + 1)
        if idx in raw_texts
    ]
    matched = _match_candidate_texts(text, span_candidates)
    if matched is None:
        matched = _match_candidate_texts(text, sorted(raw_texts.items()))
    if matched is None:
        return [text_pool.add(text=text, raw_idx=None)]
    return [
        text_pool.add(
            text=raw_texts[idx],
            raw_idx=idx,
            passage_type=passage_types.get(idx),
        )
        for idx in matched
    ]


def _match_candidate_texts(
    text: str, candidates: list[tuple[int, str]]
) -> list[int] | None:
    matched: list[int] = []
    text_pos = 0
    for raw_idx, candidate_text in candidates:
        prefix = candidate_text if text_pos == 0 else f" {candidate_text}"
        if not text.startswith(prefix, text_pos):
            continue
        matched.append(raw_idx)
        text_pos += len(prefix)
        if text_pos == len(text):
            return matched
    return None


def _group_from_chunks(
    row: Mapping[str, object],
    row_ordinal: int,
    chunks: list[IndexChunk],
    settings: Settings,
) -> IndexChunkGroup:
    title = _optional_str(row.get("title"))
    text_pool = _TextPool()
    raw_texts, passage_types = _raw_candidate_texts(row)
    items = _chunk_items_from_chunks(chunks, raw_texts, passage_types, text_pool)
    return IndexChunkGroup(
        group_id=f"{settings.dataset_name}:{settings.dataset_split}:{row_ordinal}",
        source_row_ordinal=row_ordinal,
        texts=text_pool.texts,
        title=title,
        source=title,
        question=_optional_str(row.get("question")),
        document_url=_optional_str(row.get("document_url")),
        long_answers=_long_answers_from_row(row),
        chunks=items,
    )


def _context_text_for_span(
    span: list[CandidateAnnotation],
    all_annotations: list[CandidateAnnotation],
    text: str,
    settings: Settings,
) -> tuple[str, int | None]:
    needs_context = (
        token_count(text) < settings.chunk_min_tokens_soft
        or any(
            candidate.role in {"table_child", "list_item", "header", "intro"}
            for candidate in span
        )
        or len(span) > 1
    )
    if not needs_context:
        return text, span[0].parent_candidate_idx
    context, parent_idx = _local_context_for_span(span, all_annotations, text)
    if token_count(context) <= settings.chunk_context_text_token_cap:
        return context, parent_idx
    words = context.split()
    return " ".join(words[: settings.chunk_context_text_token_cap]), parent_idx


def _local_context_for_span(
    span: list[CandidateAnnotation],
    all_annotations: list[CandidateAnnotation],
    text: str,
) -> tuple[str, int | None]:
    """Return local parent/header context for structurally dependent chunks."""

    start = min(candidate.index for candidate in span)
    by_index = {candidate.index: candidate for candidate in all_annotations}
    parent_idx = next(
        (
            candidate.parent_candidate_idx
            for candidate in span
            if candidate.parent_candidate_idx is not None
        ),
        None,
    )
    if parent_idx is not None:
        parent = by_index.get(parent_idx)
        if parent is not None and parent.text not in text:
            return f"{parent.text} {text}", parent_idx

    prefix: list[str] = []
    for candidate in reversed(all_annotations):
        if candidate.index >= start:
            continue
        if candidate.role not in {"header", "intro", "list_parent", "table_parent"}:
            if prefix:
                break
            continue
        if candidate.text not in text:
            prefix.insert(0, candidate.text)
        if token_count(" ".join(prefix)) >= 80:
            break
    if prefix:
        return f"{' '.join(prefix)} {text}", None
    return text, parent_idx


def _resolve_trailing_tiny_chunks(
    chunks: list[IndexChunk], settings: Settings, state: _ChunkBuildState
) -> list[IndexChunk]:
    if not chunks:
        return chunks
    out: list[IndexChunk] = []
    for chunk in chunks:
        if chunk.token_count >= settings.chunk_min_tokens_hard or chunk.chunk_kind == "short_fact":
            out.append(chunk)
            continue
        if out:
            prev = out[-1]
            prev.text = f"{prev.text} {chunk.text}"
            prev.context_text = prev.text
            prev.end_candidate_idx = max(prev.end_candidate_idx, chunk.end_candidate_idx)
            prev.token_count = token_count(prev.text)
            prev.context_token_count = token_count(prev.context_text)
        else:
            state.tiny_suppressed += 1
    return out


def _chunk_kind(span: list[CandidateAnnotation]) -> str:
    roles = {candidate.role for candidate in span}
    if roles & {"table_parent", "table_child"}:
        return "table_span"
    if roles & {"list_parent", "list_item"}:
        return "list_span"
    if len(span) > 1:
        return "minimum_context_span"
    return "paragraph"


def _compatible(left: CandidateAnnotation, right: CandidateAnnotation) -> bool:
    if left.role in {"header", "intro"}:
        return True
    if left.passage_type == right.passage_type:
        return True
    if {left.passage_type, right.passage_type}.issubset(_LIST_OR_TABLE_TYPES):
        return True
    return left.token_count <= 30 and right.token_count <= 30


def _looks_like_intro(text: str) -> bool:
    normalized = text.casefold()
    return normalized.endswith(":") or (
        token_count(normalized) <= 40 and any(marker in normalized for marker in _INTRO_MARKERS)
    )


def _is_self_contained_fact(text: str, tokens: int) -> bool:
    if tokens < 8:
        return False
    normalized = f" {text.casefold()} "
    has_predicate = any(
        marker in normalized
        for marker in (" is ", " was ", " are ", " were ", " has ", " had ", " consists ")
    )
    return has_predicate and text.rstrip().endswith((".", "!", "?"))


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _long_answers_from_row(row: Mapping[str, object]) -> list[str]:
    raw = row.get("long_answers")
    if not isinstance(raw, list):
        return []
    return [str(value) for value in raw if value is not None]
