from __future__ import annotations

import tomllib
from pathlib import Path

from scripts.gauntlet import test_steps as gauntlet_test_steps

ROOT = Path(__file__).parents[2]


def test_quality_docs_are_reachable_from_mkdocs_navigation() -> None:
    configuration = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert "Architecture decisions:" in configuration
    assert "adr/0001-quality-gauntlet.md" in configuration
    assert "adr/0002-layered-architecture.md" in configuration
    assert "Technical debt: technical-debt.md" in configuration


def test_ci_runs_the_unskipped_gauntlet() -> None:
    workflow = (ROOT / ".github/workflows/qa.yml").read_text(encoding="utf-8")

    assert "uv run python scripts/gauntlet.py" in workflow
    assert "--skip-network" not in workflow
    assert "--skip-docker" not in workflow


def test_quality_configuration_covers_property_tests_and_branch_thresholds() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        configuration = tomllib.load(project_file)

    assert any("hypothesis" in dependency for dependency in configuration["dependency-groups"]["dev"])
    assert "tests/property" in configuration["tool"]["mutmut"]["pytest_add_cli_args_test_selection"]
    assert gauntlet_test_steps()[1][0] == "property tests"
    assert "scripts/check_coverage.py" in (ROOT / "scripts/gauntlet.py").read_text(encoding="utf-8")
