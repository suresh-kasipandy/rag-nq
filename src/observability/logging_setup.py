"""Stage-oriented logging helpers for pipeline visibility."""

from __future__ import annotations

import logging
from typing import Any


class StageLoggerAdapter(logging.LoggerAdapter):
    """Prepends ``[stage]`` when ``extra`` contains ``stage`` (e.g. ``stage`` = ``ingest``)."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.get("extra")
        stage = extra.get("stage") if isinstance(extra, dict) else None
        if stage:
            return f"[{stage}] {msg}", kwargs
        return msg, kwargs


def get_logger(name: str) -> logging.Logger:
    """Return a module logger; use with :class:`StageLoggerAdapter` for stage tags."""

    return logging.getLogger(name)


def get_stage_logger(name: str) -> StageLoggerAdapter:
    """Return a logger that supports ``logger.info(..., extra={\"stage\": \"ingest\"})``."""

    return StageLoggerAdapter(get_logger(name), {})


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once (idempotent for repeated calls in CLI)."""

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
