from scripts.check_mutants import has_unfinished_mutants
from scripts.gauntlet import quality_paths


def test_mutation_checker_rejects_no_tests_results() -> None:
    assert has_unfinished_mutants("storage mutant: no tests") is True


def test_mutation_checker_accepts_killed_results() -> None:
    assert has_unfinished_mutants("storage mutant: killed") is False


def test_quality_paths_skip_missing_roots(tmp_path) -> None:
    (tmp_path / "src").mkdir()

    assert quality_paths(tmp_path) == ("src",)
