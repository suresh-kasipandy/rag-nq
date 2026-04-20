from __future__ import annotations

import io
import logging

from src.observability.logging_setup import StageLoggerAdapter, setup_logging


def test_stage_logger_adapter_prepends_stage() -> None:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    base = logging.getLogger("test_obs_stage")
    base.handlers.clear()
    base.addHandler(handler)
    base.setLevel(logging.INFO)

    adapter = StageLoggerAdapter(base, {})
    adapter.info("hello", extra={"stage": "ingest"})

    assert "[ingest]" in buf.getvalue()
    assert "hello" in buf.getvalue()


def test_setup_logging_idempotent() -> None:
    setup_logging()
    setup_logging()
    root = logging.getLogger()
    assert root.level <= logging.INFO
