from __future__ import annotations

import pytest

from src.scripts.doctor import CheckResult, run_doctor


def test_run_doctor_config_failure_reports_check(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise RuntimeError("simulated config failure")

    monkeypatch.setattr("src.scripts.doctor.Settings.from_env", boom)
    monkeypatch.setattr(
        "src.scripts.doctor._try_import",
        lambda module: CheckResult(name=f"import:{module}", ok=True, detail="ok"),
    )
    results = run_doctor()
    config = next(r for r in results if r.name == "config")
    assert not config.ok
    assert "simulated config failure" in config.detail


def test_run_doctor_import_failure_reported(monkeypatch: pytest.MonkeyPatch) -> None:

    def flaky_import(module: str) -> CheckResult:
        if module == "rank_bm25":
            return CheckResult(name=f"import:{module}", ok=False, detail="missing module")
        return CheckResult(name=f"import:{module}", ok=True, detail="ok")

    monkeypatch.setattr("src.scripts.doctor._try_import", flaky_import)
    results = run_doctor()
    rank = next(r for r in results if r.name == "import:rank_bm25")
    assert not rank.ok
    assert "missing module" in rank.detail


def test_run_doctor_structure_when_imports_patched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.scripts.doctor._try_import",
        lambda module: CheckResult(name=f"import:{module}", ok=True, detail="ok"),
    )
    results = run_doctor()
    names = {r.name for r in results}
    assert "config" in names
    assert "import:pydantic" in names
    assert "import:datasets" in names
    assert all(r.ok for r in results)


def test_run_doctor_live_optional() -> None:
    """If dependencies are installed, all checks pass; otherwise skip."""

    results = run_doctor()
    failed = [r for r in results if not r.ok]
    if failed:
        pytest.skip(f"environment missing deps or config: {failed!r}")
    assert len(results) >= 6
