from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.check_coverage import coverage_percentages, coverage_violations


def _write_report(path: Path, *, lines: int, statements: int, branches: int, total_branches: int) -> None:
    path.write_text(
        json.dumps(
            {
                "totals": {
                    "covered_lines": lines,
                    "num_statements": statements,
                    "covered_branches": branches,
                    "num_branches": total_branches,
                }
            }
        ),
        encoding="utf-8",
    )


def test_coverage_percentages_report_line_and_branch_coverage(tmp_path: Path) -> None:
    report = tmp_path / "coverage.json"
    _write_report(report, lines=95, statements=100, branches=90, total_branches=100)

    assert coverage_percentages(report) == pytest.approx((95.0, 90.0))
    assert coverage_violations(report) == ()


def test_coverage_gate_reports_each_threshold_that_is_missed(tmp_path: Path) -> None:
    report = tmp_path / "coverage.json"
    _write_report(report, lines=94, statements=100, branches=89, total_branches=100)

    assert coverage_violations(report) == (
        "line coverage 94.00% is below 95.00%",
        "branch coverage 89.00% is below 90.00%",
    )


def test_coverage_gate_reports_malformed_reports(tmp_path: Path) -> None:
    report = tmp_path / "coverage.json"
    report.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=r"coverage report is missing totals\.covered_lines"):
        coverage_percentages(report)
