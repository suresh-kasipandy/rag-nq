from __future__ import annotations

import pytest

from src.ingestion.chunk_store import count_index_records_jsonl, iter_index_records_jsonl
from src.ingestion.models import (
    IndexChunk,
    IndexChunkGroup,
    IndexChunkItem,
    IndexText,
    Passage,
)


def test_iter_index_records_flattens_grouped_chunks(tmp_path) -> None:
    path = tmp_path / "index_chunks.jsonl"
    group = IndexChunkGroup(
        group_id="g1",
        source_row_ordinal=7,
        title="Mercury",
        question="Which planet is smallest?",
        texts=[
            IndexText(
                text_idx=0,
                raw_idx=2,
                text="Mercury is the smallest planet.",
                passage_type="paragraph",
            ),
            IndexText(
                text_idx=1,
                raw_idx=3,
                text="It has no natural moons.",
            ),
            IndexText(
                text_idx=2,
                raw_idx=None,
                text="Mercury overview.",
            ),
        ],
        chunks=[
            IndexChunkItem(
                chunk_id="11111111-1111-1111-1111-111111111111",
                text_idxs=[0, 1],
                context_idxs=[2, 0, 1],
                start_candidate_idx=2,
                end_candidate_idx=3,
                chunk_kind="minimum_context_span",
            )
        ],
    )
    path.write_text(group.model_dump_json(exclude_none=True) + "\n", encoding="utf-8")

    records = list(iter_index_records_jsonl(path))

    assert len(records) == 1
    assert isinstance(records[0], IndexChunk)
    assert records[0].chunk_id == group.chunks[0].chunk_id
    assert records[0].group_id == "g1"
    assert records[0].text == (
        "Mercury is the smallest planet. It has no natural moons."
    )
    assert records[0].context_text == (
        "Mercury overview. Mercury is the smallest planet. "
        "It has no natural moons."
    )
    assert records[0].source_row_ordinal == 7
    assert records[0].question == "Which planet is smallest?"
    assert count_index_records_jsonl(path) == 1


def test_iter_index_records_preserves_legacy_flat_records(tmp_path) -> None:
    path = tmp_path / "legacy.jsonl"
    flat_chunk = IndexChunk(
        chunk_id="22222222-2222-2222-2222-222222222222",
        group_id="g2",
        text="flat chunk",
        context_text="flat chunk",
        source_row_ordinal=0,
        start_candidate_idx=0,
        end_candidate_idx=0,
    )
    passage = Passage(passage_id="p1", text="legacy passage")
    path.write_text(
        flat_chunk.model_dump_json() + "\n" + passage.model_dump_json() + "\n",
        encoding="utf-8",
    )

    records = list(iter_index_records_jsonl(path, max_records=1))

    assert len(records) == 1
    assert isinstance(records[0], IndexChunk)
    assert records[0].chunk_id == flat_chunk.chunk_id
    assert count_index_records_jsonl(path) == 2


def test_index_chunk_group_rejects_ambiguous_text_pool() -> None:
    with pytest.raises(ValueError, match="text_idx values"):
        IndexChunkGroup(
            group_id="g1",
            source_row_ordinal=0,
            texts=[IndexText(text_idx=1, text="orphaned position")],
            chunks=[
                IndexChunkItem(
                    chunk_id="33333333-3333-3333-3333-333333333333",
                    text_idxs=[1],
                    context_idxs=[1],
                    start_candidate_idx=0,
                    end_candidate_idx=0,
                )
            ],
        )


def test_index_chunk_group_rejects_unknown_text_reference() -> None:
    with pytest.raises(ValueError, match="text_idxs"):
        IndexChunkGroup(
            group_id="g1",
            source_row_ordinal=0,
            texts=[IndexText(text_idx=0, text="known text")],
            chunks=[
                IndexChunkItem(
                    chunk_id="44444444-4444-4444-4444-444444444444",
                    text_idxs=[1],
                    context_idxs=[0],
                    start_candidate_idx=0,
                    end_candidate_idx=0,
                )
            ],
        )
