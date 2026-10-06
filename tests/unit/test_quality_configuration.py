from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parents[2]


def _load_yaml(relative_path: str) -> Any:
    return yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))


def _nav_pages(node: Any) -> set[str]:
    if isinstance(node, str):
        return {node}
    if isinstance(node, list):
        return set().union(*(_nav_pages(item) for item in node)) if node else set()
    if isinstance(node, dict):
        return set().union(*(_nav_pages(value) for value in node.values())) if node else set()
    return set()


def test_quality_docs_are_reachable_from_mkdocs_navigation() -> None:
    pages = _nav_pages(_load_yaml("mkdocs.yml")["nav"])

    assert {
        "adr/0001-quality-gauntlet.md",
        "adr/0002-layered-architecture.md",
        "technical-debt.md",
    } <= pages
    assert all((ROOT / "docs" / page).is_file() for page in pages)


def test_ci_runs_the_unskipped_gauntlet() -> None:
    workflow = _load_yaml(".github/workflows/qa.yml")

    commands = [
        step["run"] for step in workflow["jobs"]["gauntlet"]["steps"] if isinstance(step.get("run"), str)
    ]
    assert "uv run python scripts/gauntlet.py" in commands
    assert not any("--skip-" in command for command in commands)


def test_quality_configuration_covers_property_tests_and_branch_thresholds() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        configuration = tomllib.load(project_file)

    assert any("hypothesis" in dependency for dependency in configuration["dependency-groups"]["dev"])
    assert "tests/property" in configuration["tool"]["mutmut"]["pytest_add_cli_args_test_selection"]
