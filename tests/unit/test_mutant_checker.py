import subprocess

import pytest
from scripts import check_mutants
from scripts.check_mutants import has_unfinished_mutants
from scripts.gauntlet import quality_paths

# Independent of production: removing a marker from check_mutants must make these fail.
UNFINISHED_STATUSES = ("survived", "🙁", "timeout", "untested", "no tests")


@pytest.mark.parametrize("marker", UNFINISHED_STATUSES)
def test_mutation_checker_rejects_every_unfinished_status(marker: str) -> None:
    assert has_unfinished_mutants(f"storage.x_save__mutmut_1: {marker}") is True


@pytest.mark.parametrize("marker", UNFINISHED_STATUSES)
def test_main_rejects_every_unfinished_status_through_public_path(monkeypatch, marker: str) -> None:
    _fake_run(monkeypatch, stdout=f"storage.x_save__mutmut_1: {marker}\n")

    assert check_mutants.main([]) == 1


@pytest.mark.parametrize("marker", UNFINISHED_STATUSES)
def test_mutation_checker_is_case_insensitive(marker: str) -> None:
    assert has_unfinished_mutants(f"storage.x_save__mutmut_1: {marker.upper()}") is True


def test_mutation_checker_rejects_unfinished_line_among_killed_ones() -> None:
    output = "a.x_f__mutmut_1: killed\na.x_f__mutmut_2: survived\na.x_f__mutmut_3: killed\n"

    assert has_unfinished_mutants(output) is True


@pytest.mark.parametrize(
    "output",
    [
        "",
        "storage mutant: killed",
        "a.x_f__mutmut_1: killed\na.x_f__mutmut_2: killed\n",
        "survived_marker.x_f__mutmut_1: killed",
        "timeouts.x_untested__mutmut_1: killed",
        "no tests.x_f__mutmut_1: killed",
    ],
)
def test_mutation_checker_accepts_killed_results_even_when_names_contain_markers(output: str) -> None:
    assert has_unfinished_mutants(output) is False


def _fake_run(monkeypatch: pytest.MonkeyPatch, *, stdout: str, stderr: str = "", returncode: int = 0):
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(check_mutants.subprocess, "run", run)
    return calls


def test_main_passes_when_all_mutants_are_killed(monkeypatch, capsys) -> None:
    calls = _fake_run(monkeypatch, stdout="a.x_f__mutmut_1: killed\n")

    assert check_mutants.main([]) == 0
    assert calls == [["mutmut", "results", "--all=true"]]
    assert "killed" in capsys.readouterr().out


def test_main_fails_on_surviving_mutant(monkeypatch) -> None:
    _fake_run(monkeypatch, stdout="a.x_f__mutmut_1: survived\n")

    assert check_mutants.main([]) == 1


def test_main_fails_on_unfinished_status_in_stderr(monkeypatch) -> None:
    _fake_run(monkeypatch, stdout="", stderr="a.x_f__mutmut_1: no tests\n")

    assert check_mutants.main([]) == 1


def test_main_fails_on_nonzero_returncode_even_when_output_is_clean(monkeypatch) -> None:
    _fake_run(monkeypatch, stdout="a.x_f__mutmut_1: killed\n", returncode=2)

    assert check_mutants.main([]) == 1


@pytest.mark.parametrize("stdout", ["", "   \n\t\n"])
def test_main_fails_when_mutmut_reports_no_mutant_records(monkeypatch, capsys, stdout: str) -> None:
    _fake_run(monkeypatch, stdout=stdout)

    assert check_mutants.main([]) == 1
    assert "no mutation results" in capsys.readouterr().err


def test_main_fails_on_malformed_records_without_a_status(monkeypatch) -> None:
    _fake_run(monkeypatch, stdout="a.x_f__mutmut_1\nnoise line without status\n")

    assert check_mutants.main([]) == 1


def test_quality_paths_skip_missing_roots(tmp_path) -> None:
    (tmp_path / "src").mkdir()

    assert quality_paths(tmp_path) == ("src",)
