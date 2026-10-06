from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from scripts import gauntlet
from scripts.gauntlet import suite_steps


def _step_name(command: list[str]) -> str:
    if command[:3] == ["uv", "run", "pytest"]:
        return command[3]
    if command[:3] == ["uv", "run", "python"]:
        return command[3]
    if command[:3] == ["uv", "run", "mutmut"]:
        return "mutmut"
    return " ".join(command[:3])


class _Recorder:
    """Stand-in for subprocess.run that records commands and can fail one of them."""

    def __init__(self, failing: str | None = None, returncode: int = 5) -> None:
        self.commands: list[list[str]] = []
        self.environments: list[dict[str, str]] = []
        self._failing = failing
        self._returncode = returncode

    def __call__(self, command: list[str], check: bool, env: dict[str, str]) -> None:
        assert check is True
        self.commands.append(command)
        self.environments.append(env)
        if self._failing is not None and _step_name(command) == self._failing:
            raise subprocess.CalledProcessError(self._returncode, command)

    @property
    def names(self) -> list[str]:
        return [_step_name(command) for command in self.commands]


@pytest.fixture
def isolated_gauntlet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROJECT_STATE_ROOT", str(tmp_path / "state"))


def test_gauntlet_runs_every_step_in_order_with_deterministic_environment(
    monkeypatch: pytest.MonkeyPatch, isolated_gauntlet: None, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(gauntlet.subprocess, "run", recorder)

    assert gauntlet.main(["--mutation-workers", "3"]) == 0

    assert recorder.names == [
        "uv lock --check",
        "uv run ruff",
        "uv run ruff",
        "uv run ty",
        "tests/unit",
        "tests/property",
        "tests/acceptance",
        "scripts/check_coverage.py",
        "scripts/check_architecture.py",
        "scripts/check_crap.py",
        "mutmut",
        "scripts/check_mutants.py",
        "uv build --wheel",
        "uv run mkdocs",
        "scripts/streaming_smoke.py",
        "docker build --tag",
        "git diff --check",
    ]
    assert recorder.commands[1][3:5] == ["format", "--check"]
    assert recorder.commands[2][3] == "check"
    assert recorder.commands[10][-2:] == ["--max-children", "3"]
    assert all(env["PYTHONHASHSEED"] == "0" for env in recorder.environments)
    assert all(env["MUTMUT_MAX_CHILDREN"] == "3" for env in recorder.environments)
    assert "GAUNTLET PASSED" in capsys.readouterr().out


def test_gauntlet_skip_flags_remove_only_the_network_and_docker_steps(
    monkeypatch: pytest.MonkeyPatch, isolated_gauntlet: None
) -> None:
    full = _Recorder()
    monkeypatch.setattr(gauntlet.subprocess, "run", full)
    gauntlet.main([])
    skipped = _Recorder()
    monkeypatch.setattr(gauntlet.subprocess, "run", skipped)

    assert gauntlet.main(["--skip-network", "--skip-docker"]) == 0

    assert [name for name in full.names if name not in skipped.names] == [
        "scripts/streaming_smoke.py",
        "docker build --tag",
    ]


@pytest.mark.parametrize("failing", ["uv lock --check", "tests/property", "scripts/check_crap.py", "mutmut"])
def test_gauntlet_stops_at_the_first_failing_step_and_returns_its_status(
    monkeypatch: pytest.MonkeyPatch,
    isolated_gauntlet: None,
    capsys: pytest.CaptureFixture[str],
    failing: str,
) -> None:
    recorder = _Recorder(failing=failing, returncode=5)
    monkeypatch.setattr(gauntlet.subprocess, "run", recorder)

    assert gauntlet.main(["--skip-network", "--skip-docker"]) == 5

    assert recorder.names[-1] == failing
    assert recorder.names.count(failing) == 1
    assert "git diff --check" not in recorder.names
    assert "GAUNTLET PASSED" not in capsys.readouterr().out


def test_gauntlet_runs_test_suites_in_distinct_coverage_accumulating_stages() -> None:
    steps = suite_steps()

    assert tuple(name for name, _ in steps) == ("unit tests", "property tests", "acceptance tests")
    assert all("--cov=landuse_sentence_relevance" in command for _, command in steps)
    assert "--cov-append" not in steps[0][1]
    assert "--cov-append" in steps[1][1]
    assert "--cov-append" in steps[2][1]
    assert "--cov-report=json:coverage.json" in steps[2][1]
