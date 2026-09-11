from __future__ import annotations

from scripts.gauntlet import test_steps as gauntlet_test_steps


def test_gauntlet_runs_test_suites_in_distinct_coverage_accumulating_stages() -> None:
    steps = gauntlet_test_steps()

    assert tuple(name for name, _ in steps) == ("unit tests", "property tests", "acceptance tests")
    assert all("--cov=landuse_sentence_relevance" in command for _, command in steps)
    assert "--cov-append" not in steps[0][1]
    assert "--cov-append" in steps[1][1]
    assert "--cov-append" in steps[2][1]
    assert "--cov-report=json:coverage.json" in steps[2][1]
