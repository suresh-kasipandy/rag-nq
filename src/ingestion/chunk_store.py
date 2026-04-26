"""Read and flatten grouped chunk artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from src.ingestion.models import IndexChunk, IndexChunkGroup, Passage

IndexRecord = IndexChunk | Passage


def iter_index_records_jsonl(
    path: Path, *, max_records: int | None = None, max_rows: int | None = None
) -> Iterator[IndexRecord]:
    """Yield flattened index records from grouped chunks, flat chunks, or legacy passages.

    ``max_rows`` caps non-empty JSONL lines before flattening, preserving row-based
    throttling for grouped ``index_chunks.jsonl`` artifacts. ``max_records`` remains
    available for legacy/debug caps over flattened index records.
    """

    yielded = 0
    groups_seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if max_rows is not None and groups_seen >= max_rows:
                return
            groups_seen += 1
            for record in _records_from_json_line(line):
                if max_records is not None and yielded >= max_records:
                    return
                yielded += 1
                yield record


def count_index_records_jsonl(
    path: Path, *, max_records: int | None = None, max_rows: int | None = None
) -> int:
    """Count flattened index records in an index JSONL artifact."""

    return sum(
        1
        for _ in iter_index_records_jsonl(
            path,
            max_records=max_records,
            max_rows=max_rows,
        )
    )


def flatten_index_chunk_group(group: IndexChunkGroup) -> Iterator[IndexChunk]:
    """Expand a grouped chunk artifact row into indexable chunks."""

    texts_by_idx = {entry.text_idx: entry.text for entry in group.texts}
    for item in group.chunks:
        text = " ".join(texts_by_idx[idx] for idx in item.text_idxs)
        context_text = " ".join(texts_by_idx[idx] for idx in item.context_idxs)
        yield IndexChunk(
            chunk_id=item.chunk_id,
            group_id=group.group_id,
            text=text,
            context_text=context_text,
            source_row_ordinal=group.source_row_ordinal,
            start_candidate_idx=item.start_candidate_idx,
            end_candidate_idx=item.end_candidate_idx,
            passage_types=item.passage_types,
            title=group.title,
            source=group.source,
            question=group.question,
            document_url=group.document_url,
            parent_candidate_idx=item.parent_candidate_idx,
            chunk_kind=item.chunk_kind,
            token_count=item.token_count,
            context_token_count=item.context_token_count,
            long_answers=group.long_answers,
        )


def _records_from_json_line(line: str) -> Iterator[IndexRecord]:
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL line must decode to an object.")
    if "chunks" in payload:
        yield from flatten_index_chunk_group(IndexChunkGroup.model_validate(payload))
    elif "chunk_id" in payload:
        yield IndexChunk.model_validate(payload)
    else:
        yield Passage.model_validate(payload)
