import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_mkdocs_configuration_is_strict_and_has_public_navigation() -> None:
    configuration = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert "theme:" in configuration
    assert (
        "site_url: https://noeflandre.github.io/landuse-sentence-relevance-golden-human-set/" in configuration
    )
    assert "Overview: index.md" in configuration
    assert "QA: qa.md" in configuration


def test_pages_workflow_builds_strictly_and_deploys_with_least_privilege() -> None:
    workflow = (ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")

    assert "mkdocs build --strict" in workflow
    assert "upload-pages-artifact" in workflow
    assert "deploy-pages" in workflow
    assert "pages: write" in workflow
    assert "id-token: write" in workflow


def test_qa_workflow_installs_the_browser_used_by_acceptance_tests() -> None:
    workflow = (ROOT / ".github/workflows/qa.yml").read_text(encoding="utf-8")

    assert "uv run playwright install --with-deps chromium" in workflow


def test_qa_workflow_installs_model_dependencies_used_by_model_tests() -> None:
    workflow = (ROOT / ".github/workflows/qa.yml").read_text(encoding="utf-8")

    assert "uv sync --locked --extra models" in workflow


def test_docker_image_uses_locked_uv_dependencies_without_source_data() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "uv sync --frozen --no-dev --extra models" in dockerfile
    assert "HF_HOME" in dockerfile
    assert "COPY data" not in dockerfile


def test_public_attribution_files_use_apache_code_license() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")

    assert "license: Apache-2.0" in citation
    assert "Apache License" in license_text


def test_local_model_cache_rule_does_not_hide_source_model_adapters() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "\n/models/\n" in gitignore
    assert "\nmodels/\n" not in gitignore


def test_mutation_gate_includes_storage_modules() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        configuration = tomllib.load(project_file)

    source_paths = configuration["tool"]["mutmut"]["source_paths"]

    assert "src/landuse_sentence_relevance/domain" in source_paths
    assert "src/landuse_sentence_relevance/storage" in source_paths


def test_pytest_pythonpath_leaves_mutmut_source_precedence_intact() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        configuration = tomllib.load(project_file)

    assert configuration["tool"]["pytest"]["ini_options"]["pythonpath"] == ["."]


def test_mutation_gate_rebuilds_results_when_project_dependencies_change() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        configuration = tomllib.load(project_file)

    assert configuration["tool"]["mutmut"]["on_dependency_change"] == "rerun"


def test_mutation_gate_copies_qa_helpers_into_isolated_test_tree() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        configuration = tomllib.load(project_file)

    assert "scripts" in configuration["tool"]["mutmut"]["also_copy"]
