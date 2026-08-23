from scripts.check_mutants import has_unfinished_mutants


def test_mutation_checker_rejects_no_tests_results() -> None:
    assert has_unfinished_mutants("storage mutant: no tests") is True


def test_mutation_checker_accepts_killed_results() -> None:
    assert has_unfinished_mutants("storage mutant: killed") is False
