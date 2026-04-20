from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.scripts.doctor import CheckResult, check_qdrant_ready, run_doctor


def test_check_qdrant_ready_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    monkeypatch.setattr("src.scripts.doctor.urlopen", lambda *_a, **_k: mock_resp)

    result = check_qdrant_ready("http://localhost:6333/")
    assert result.ok
    assert "readyz" in result.detail


def test_check_qdrant_ready_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from urllib.error import URLError

    monkeypatch.setattr(
        "src.scripts.doctor.urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(URLError("refused")),
    )

    result = check_qdrant_ready("http://127.0.0.1:6333")
    assert not result.ok
    assert "URLError" in result.detail


def test_run_doctor_includes_qdrant_when_env_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_DOCTOR_CHECK_QDRANT", "1")
    monkeypatch.setattr(
        "src.scripts.doctor._try_import",
        lambda module: CheckResult(name=f"import:{module}", ok=True, detail="ok"),
    )
    monkeypatch.setattr(
        "src.scripts.doctor.check_qdrant_ready",
        lambda url: CheckResult(name="qdrant:readyz", ok=True, detail=f"stub {url}"),
    )

    results = run_doctor()
    names = [r.name for r in results]
    assert "qdrant:readyz" in names
