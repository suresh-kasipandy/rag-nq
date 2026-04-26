from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.observability.progress import ProgressTicker, count_non_empty_jsonl


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.extras: list[dict[str, Any]] = []

    def info(self, msg: str, *args: object, **kwargs: Any) -> None:
        self.messages.append(msg % args if args else msg)
        extra = kwargs.get("extra")
        self.extras.append(extra if isinstance(extra, dict) else {})


def test_progress_ticker_logs_start_item_interval_time_interval_and_finish() -> None:
    clock = FakeClock()
    logger = FakeLogger()
    ticker = ProgressTicker(
        logger=logger,
        stage="chunk",
        label="rows",
        total=10,
        every_items=5,
        every_seconds=60.0,
        clock=clock,
    )

    ticker.start(source="raw")
    ticker.tick(4)
    ticker.tick(5, chunks=8)
    clock.advance(61.0)
    ticker.tick(6, chunks=9)
    ticker.finish(6, chunks=9)

    assert len(logger.messages) == 4
    assert logger.messages[0].startswith("start rows=0/10")
    assert "progress rows=5/10" in logger.messages[1]
    assert "chunks=8" in logger.messages[1]
    assert "progress rows=6/10" in logger.messages[2]
    assert logger.messages[3].startswith("complete rows=6/10")
    assert all(extra == {"stage": "chunk"} for extra in logger.extras)


def test_progress_ticker_rejects_invalid_thresholds() -> None:
    logger = FakeLogger()
    with pytest.raises(ValueError, match="every_items"):
        ProgressTicker(logger=logger, stage="x", label="rows", every_items=0)
    with pytest.raises(ValueError, match="every_seconds"):
        ProgressTicker(logger=logger, stage="x", label="rows", every_seconds=0)


def test_count_non_empty_jsonl_respects_max_records(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")

    assert count_non_empty_jsonl(path) == 3
    assert count_non_empty_jsonl(path, max_records=2) == 2
    assert count_non_empty_jsonl(path, max_records=0) == 0
