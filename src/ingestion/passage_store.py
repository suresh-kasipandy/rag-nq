"""Persistence helpers for normalized passages and manifests."""

from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.models import IndexBuildManifest, Passage


class PassageStore:
    """Persistence helpers for passage-oriented artifacts."""

    @staticmethod
    def write_jsonl(passages: list[Passage], path: Path) -> int:
        """Write normalized passages as JSONL and return written count."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for passage in passages:
                handle.write(passage.model_dump_json())
                handle.write("\n")
        return len(passages)

    @staticmethod
    def read_jsonl(path: Path) -> list[Passage]:
        """Read normalized passages from JSONL file."""

        passages: list[Passage] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                passages.append(Passage.model_validate_json(line))
        return passages

    @staticmethod
    def write_manifest(manifest: IndexBuildManifest, path: Path) -> None:
        """Write build manifest as stable JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest.model_dump(), handle, indent=2, sort_keys=True)
