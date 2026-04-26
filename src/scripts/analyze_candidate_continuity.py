"""Quick inspection utility for NQ candidate continuity and duplicate text reuse.

Reports:
1) likely adjacent candidate continuations within the same row/group
2) repeated candidate text across different rows/questions

Data source options:
- ``artifact`` (default): read local ``artifacts/passages.jsonl`` and infer contiguous groups
- ``hf``: stream raw Hugging Face rows from ``sentence-transformers/NQ-retrieval``
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from src.config.settings import Settings


@dataclass(slots=True)
class ContinuationExample:
    row_index: int | str
    candidate_index: int
    title: str | None
    question: str | None
    left_type: str | None
    right_type: str | None
    left_text: str
    right_text: str


@dataclass(slots=True)
class TextOccurrence:
    row_index: int | str
    candidate_index: int
    title: str | None
    question: str | None


def _to_clean_text(raw: object) -> str:
    return str(raw).strip()


def _looks_like_intro_fragment(text: str) -> bool:
    lower = text.lower()
    intro_markers = (
        "including",
        "include",
        "includes",
        "such as",
        "the following",
        "for example",
    )
    if text.endswith(":"):
        return True
    return any(marker in lower for marker in intro_markers) and text.endswith((".", ":"))


def _looks_like_followup_fragment(text: str) -> bool:
    if not text:
        return False
    if "\n" in text:
        return True
    # Short single-line noun phrases are common in list splits.
    words = text.split()
    return 1 <= len(words) <= 16


def _iter_hf_rows(dataset_name: str, split: str) -> Iterable[object]:
    from datasets import load_dataset

    return load_dataset(dataset_name, split=split, streaming=True)


def _group_key_from_passage_obj(obj: dict[str, object]) -> tuple[str | None, str | None, str | None]:
    title_raw = obj.get("title")
    question_raw = obj.get("question")
    doc_url_raw = obj.get("document_url")
    title = str(title_raw).strip() if title_raw is not None else None
    question = str(question_raw).strip() if question_raw is not None else None
    document_url = str(doc_url_raw).strip() if doc_url_raw is not None else None
    return title, question, document_url


def _iter_artifact_groups(path: Path, *, max_rows: int) -> Iterator[tuple[int, dict[str, object]]]:
    """Yield pseudo-rows reconstructed from contiguous JSONL lines with same row key."""

    with path.open("r", encoding="utf-8") as handle:
        group_index = 0
        rows_seen = 0
        current_key: tuple[str | None, str | None, str | None] | None = None
        current: dict[str, object] | None = None

        for raw_line in handle:
            if rows_seen >= max_rows:
                break
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            key = _group_key_from_passage_obj(obj)
            passage_text = _to_clean_text(obj.get("text", ""))
            if not passage_text:
                continue
            passage_type_raw = obj.get("passage_type")
            passage_type = (
                str(passage_type_raw).strip() if passage_type_raw is not None else None
            )

            if current is None or key != current_key:
                if current is not None:
                    yield group_index, current
                    group_index += 1
                    rows_seen += 1
                    if rows_seen >= max_rows:
                        break
                current_key = key
                current = {
                    "title": key[0],
                    "question": key[1],
                    "document_url": key[2],
                    "candidates": [],
                    "passage_types": [],
                }

            assert current is not None
            candidates = current["candidates"]
            types = current["passage_types"]
            assert isinstance(candidates, list)
            assert isinstance(types, list)
            candidates.append(passage_text)
            types.append(passage_type)

        if current is not None and rows_seen < max_rows:
            yield group_index, current


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("artifact", "hf"),
        default="artifact",
        help="Data source for analysis.",
    )
    parser.add_argument("--dataset-name", default=Settings().dataset_name)
    parser.add_argument("--split", default=Settings().dataset_split)
    parser.add_argument(
        "--artifact-path",
        default=str(Settings().passages_path),
        help="Path to local passages.jsonl (used when --source artifact).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20000,
        help="Maximum streamed rows to inspect.",
    )
    parser.add_argument(
        "--max-continuation-examples",
        type=int,
        default=12,
        help="How many likely continuation pairs to print.",
    )
    parser.add_argument(
        "--max-duplicate-groups",
        type=int,
        default=10,
        help="How many duplicated text groups to print.",
    )
    parser.add_argument(
        "--max-occurrences-per-duplicate",
        type=int,
        default=4,
        help="Max occurrences to print under each duplicated text group.",
    )
    args = parser.parse_args()

    scanned_rows = 0
    rows_with_candidates = 0
    total_candidates = 0
    adjacent_candidate_pairs = 0
    likely_continuations = 0
    continuation_examples: list[ContinuationExample] = []

    # Keep a bounded list of occurrences per exact candidate text.
    text_occurrences: dict[str, list[TextOccurrence]] = defaultdict(list)

    if args.source == "hf":
        row_iter = enumerate(_iter_hf_rows(args.dataset_name, args.split))
    else:
        row_iter = _iter_artifact_groups(Path(args.artifact_path), max_rows=args.max_rows)

    for row_index, row in row_iter:
        scanned_rows += 1

        candidates_raw = row.get("candidates")
        if not isinstance(candidates_raw, list):
            continue
        rows_with_candidates += 1

        passage_types_raw = row.get("passage_types")
        passage_types = passage_types_raw if isinstance(passage_types_raw, list) else []

        title = str(row.get("title")).strip() if row.get("title") is not None else None
        question = str(row.get("question")).strip() if row.get("question") is not None else None

        cleaned: list[tuple[int, str]] = []
        for idx, cand in enumerate(candidates_raw):
            text = _to_clean_text(cand)
            if not text:
                continue
            total_candidates += 1
            cleaned.append((idx, text))
            text_occurrences[text].append(
                TextOccurrence(
                    row_index=row_index,
                    candidate_index=idx,
                    title=title,
                    question=question,
                )
            )

        for pair_i in range(len(cleaned) - 1):
            left_idx, left_text = cleaned[pair_i]
            right_idx, right_text = cleaned[pair_i + 1]
            adjacent_candidate_pairs += 1

            if _looks_like_intro_fragment(left_text) and _looks_like_followup_fragment(right_text):
                likely_continuations += 1
                if len(continuation_examples) < args.max_continuation_examples:
                    left_type = (
                        str(passage_types[left_idx]).strip()
                        if left_idx < len(passage_types)
                        else None
                    )
                    right_type = (
                        str(passage_types[right_idx]).strip()
                        if right_idx < len(passage_types)
                        else None
                    )
                    continuation_examples.append(
                        ContinuationExample(
                            row_index=row_index,
                            candidate_index=left_idx,
                            title=title,
                            question=question,
                            left_type=left_type,
                            right_type=right_type,
                            left_text=left_text,
                            right_text=right_text,
                        )
                    )

    duplicate_groups: list[tuple[str, list[TextOccurrence]]] = []
    for text, occs in text_occurrences.items():
        distinct_rows = {occ.row_index for occ in occs}
        if len(occs) >= 2 and len(distinct_rows) >= 2:
            duplicate_groups.append((text, occs))
    duplicate_groups.sort(key=lambda item: len(item[1]), reverse=True)

    print("=== Candidate Continuity Audit ===")
    print(f"source={args.source}")
    if args.source == "hf":
        print(f"dataset={args.dataset_name} split={args.split}")
    else:
        print(f"artifact_path={Path(args.artifact_path).resolve()}")
    print(f"scanned_rows={scanned_rows}")
    print(f"rows_with_candidates={rows_with_candidates}")
    print(f"total_non_empty_candidates={total_candidates}")
    print(f"adjacent_candidate_pairs={adjacent_candidate_pairs}")
    print(f"likely_continuation_pairs={likely_continuations}")
    ratio = (likely_continuations / adjacent_candidate_pairs) if adjacent_candidate_pairs else 0.0
    print(f"likely_continuation_ratio={ratio:.4f}")

    print("\n=== Sample Likely Continuations ===")
    if not continuation_examples:
        print("(none found in scanned sample)")
    for ex in continuation_examples:
        print(
            f"- row={ex.row_index} cand={ex.candidate_index}->{ex.candidate_index + 1} "
            f"title={ex.title!r} left_type={ex.left_type!r} right_type={ex.right_type!r}"
        )
        print(f"  question={ex.question!r}")
        print(f"  left={ex.left_text!r}")
        print(f"  right={ex.right_text!r}")

    print("\n=== Duplicate Candidate Text Groups ===")
    if not duplicate_groups:
        print("(none found in scanned sample)")
    for text, occs in duplicate_groups[: args.max_duplicate_groups]:
        distinct_rows = len({occ.row_index for occ in occs})
        print(f"- count={len(occs)} distinct_rows={distinct_rows} text={text[:140]!r}")
        for occ in occs[: args.max_occurrences_per_duplicate]:
            print(
                f"  row={occ.row_index} cand={occ.candidate_index} "
                f"title={occ.title!r} question={occ.question!r}"
            )


if __name__ == "__main__":
    main()
