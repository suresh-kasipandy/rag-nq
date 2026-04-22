from __future__ import annotations

import io
import logging

from src.observability.logging_setup import StageLoggerAdapter, quiet_http_clients, setup_logging


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


def test_quiet_http_clients_restores_levels() -> None:
    httpx_log = logging.getLogger("httpx")
    httpx_log.setLevel(logging.DEBUG)
    with quiet_http_clients():
        assert httpx_log.level == logging.WARNING
    assert httpx_log.level == logging.DEBUG
