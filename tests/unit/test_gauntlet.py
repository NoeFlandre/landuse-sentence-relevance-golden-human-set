from __future__ import annotations

import fcntl
import os
import sys
from pathlib import Path

import pytest
from scripts.gauntlet import (
    MutationGateBusyError,
    StepFailedError,
    main,
    mutation_lock,
    mutation_lock_path,
    run_step,
)


def test_mutation_lock_path_uses_the_project_state_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state_root = tmp_path / "state"
    monkeypatch.setenv("PROJECT_STATE_ROOT", str(state_root))

    assert mutation_lock_path() == state_root / "mutation.lock"


def test_mutation_lock_rejects_a_concurrent_gate(tmp_path: Path) -> None:
    lock_path = tmp_path / "mutation.lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(MutationGateBusyError, match="already running"), mutation_lock(lock_path):
                pass
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_mutation_lock_is_released_after_the_gate_finishes(tmp_path: Path) -> None:
    lock_path = tmp_path / "mutation.lock"

    with mutation_lock(lock_path):
        assert lock_path.is_file()

    with mutation_lock(lock_path):
        pass


def test_run_step_reports_the_failing_step_and_its_exit_status() -> None:
    with pytest.raises(StepFailedError, match=r"gauntlet step failed: boom \(exit status 7\)") as raised:
        run_step("boom", [sys.executable, "-c", "raise SystemExit(7)"], os.environ.copy())

    assert raised.value.returncode == 7


def test_main_returns_the_failed_step_exit_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(name: str, command: list[str], environment: dict[str, str]) -> None:
        raise StepFailedError(name, 3)

    monkeypatch.setattr("scripts.gauntlet.run_step", fail)

    assert main([]) == 3
    assert "gauntlet step failed: lock (exit status 3)" in capsys.readouterr().err
