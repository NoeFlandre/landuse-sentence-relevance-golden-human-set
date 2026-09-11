from __future__ import annotations

import fcntl
from pathlib import Path

import pytest
from scripts.gauntlet import MutationGateBusyError, mutation_lock, mutation_lock_path


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
