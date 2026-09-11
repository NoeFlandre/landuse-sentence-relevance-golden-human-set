from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

MIN_LINE_COVERAGE = 95.0
MIN_BRANCH_COVERAGE = 90.0


def coverage_percentages(path: Path) -> tuple[float, float]:
    """Read line and branch percentages from a coverage.py JSON report."""

    try:
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read coverage report {path}: {error}") from error

    totals = report.get("totals") if isinstance(report, dict) else None
    if not isinstance(totals, Mapping):
        raise ValueError("coverage report is missing totals.covered_lines")
    covered_lines = _integer(totals, "covered_lines")
    statements = _integer(totals, "num_statements")
    covered_branches = _integer(totals, "covered_branches")
    branches = _integer(totals, "num_branches")
    if statements == 0:
        raise ValueError("coverage report has no statements")
    line_percentage = covered_lines / statements * 100
    branch_percentage = 100.0 if branches == 0 else covered_branches / branches * 100
    return line_percentage, branch_percentage


def coverage_violations(path: Path) -> tuple[str, ...]:
    """Return threshold violations for a coverage.py JSON report."""

    line_percentage, branch_percentage = coverage_percentages(path)
    violations: list[str] = []
    if line_percentage < MIN_LINE_COVERAGE:
        violations.append(f"line coverage {line_percentage:.2f}% is below {MIN_LINE_COVERAGE:.2f}%")
    if branch_percentage < MIN_BRANCH_COVERAGE:
        violations.append(f"branch coverage {branch_percentage:.2f}% is below {MIN_BRANCH_COVERAGE:.2f}%")
    return tuple(violations)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"coverage report is missing totals.{key}")
    if value < 0:
        raise ValueError(f"coverage report has a negative totals.{key}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Check line and branch coverage thresholds.")
    parser.add_argument("--report", type=Path, default=Path("coverage.json"))
    args = parser.parse_args()
    line_percentage, branch_percentage = coverage_percentages(args.report)
    print(f"line coverage: {line_percentage:.2f}%")
    print(f"branch coverage: {branch_percentage:.2f}%")
    violations = coverage_violations(args.report)
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        return 1
    print("Coverage thresholds passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
