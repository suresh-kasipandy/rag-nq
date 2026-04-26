"""Small progress logging helper for long-running pipeline stages."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol


class ProgressLogger(Protocol):
    """Logger protocol shared by ``logging.Logger`` and ``LoggerAdapter``."""

    def info(self, msg: str, *args: object, **kwargs: Any) -> None: ...


class ProgressTicker:
    """Emit start/progress/finish logs using item and elapsed-time throttles."""

    def __init__(
        self,
        *,
        logger: ProgressLogger,
        stage: str,
        label: str,
        total: int | None = None,
        every_items: int = 10_000,
        every_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if every_items < 1:
            raise ValueError("every_items must be >= 1.")
        if every_seconds <= 0:
            raise ValueError("every_seconds must be > 0.")
        self._logger = logger
        self._stage = stage
        self._label = label
        self._total = total
        self._every_items = every_items
        self._every_seconds = every_seconds
        self._clock = clock
        self._started_at = clock()
        self._last_logged_at = self._started_at
        self._last_count = 0
        self._started = False

    def start(self, **metrics: object) -> None:
        """Emit the initial stage log."""

        self._started = True
        self._log("start", 0, metrics)

    def tick(self, count: int, **metrics: object) -> None:
        """Emit a progress log when item or elapsed-time thresholds are crossed."""

        if count < 0:
            raise ValueError("count must be >= 0.")
        now = self._clock()
        item_due = count >= self._last_count + self._every_items
        time_due = now >= self._last_logged_at + self._every_seconds
        if count == 0 or not (item_due or time_due):
            return
        if not self._started:
            self._started = True
        self._log("progress", count, metrics, now=now)

    def finish(self, count: int, **metrics: object) -> None:
        """Emit the final stage log."""

        if count < 0:
            raise ValueError("count must be >= 0.")
        if not self._started:
            self.start()
        self._log("complete", count, metrics)

    def _log(
        self,
        event: str,
        count: int,
        metrics: dict[str, object],
        *,
        now: float | None = None,
    ) -> None:
        logged_at = self._clock() if now is None else now
        elapsed_seconds = logged_at - self._started_at
        fields: dict[str, object] = {
            self._label: _format_count(count, self._total),
            "elapsed_seconds": f"{elapsed_seconds:.1f}",
        }
        fields.update(metrics)
        self._logger.info(
            "%s %s",
            event,
            _format_metrics(fields),
            extra={"stage": self._stage},
        )
        self._last_count = count
        self._last_logged_at = logged_at


def count_non_empty_jsonl(path: Path, *, max_records: int | None = None) -> int:
    """Count non-empty JSONL records, optionally capped by ``max_records``."""

    if max_records is not None and max_records <= 0:
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            count += 1
            if max_records is not None and count >= max_records:
                break
    return count


def _format_count(count: int, total: int | None) -> str:
    if total is None:
        return str(count)
    return f"{count}/{total}"


def _format_metrics(metrics: dict[str, object]) -> str:
    return " ".join(f"{key}={value}" for key, value in metrics.items())
