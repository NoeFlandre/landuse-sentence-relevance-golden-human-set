import csv
import os
import tomllib
from collections import Counter
from pathlib import Path

from landuse_sentence_relevance.config import V3Settings

ROOT = Path(__file__).parents[2]


def test_mkdocs_configuration_is_strict_and_has_public_navigation() -> None:
    configuration = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert "theme:" in configuration
    assert (
        "site_url: https://noeflandre.github.io/landuse-sentence-relevance-golden-human-set/" in configuration
    )
    assert "Overview: index.md" in configuration
    assert "Interrater agreement: interrater-agreement.md" in configuration
    assert "QA: qa.md" in configuration


def test_llm_evaluation_round_workflow_is_publicly_documented() -> None:
    configuration = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert "LLM evaluation rounds: llm-evaluation-rounds.md" in configuration
    assert (ROOT / "docs/llm-evaluation-rounds.md").is_file()


def test_paused_v3_workflow_is_publicly_documented() -> None:
    configuration = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    page = (ROOT / "docs/v3.md").read_text(encoding="utf-8")

    assert "V3 workflow: v3.md" in configuration
    assert "osm-polygon-description-tag" in page
    assert "Wikivoyage is excluded" in page
    assert "300" in page
    assert "paused and not implemented" in page


def test_v3_page_does_not_deny_the_settings_profile_the_repository_ships() -> None:
    page = (ROOT / "docs/v3.md").read_text(encoding="utf-8")

    assert "V3Settings" in page
    assert "V3 dataset revisions and settings are not in `config.py`" not in page
    assert "No V3 code, V3 settings, or V3 benchmark exists" not in page
    assert "no V3 setting in `config.py`" not in page


def test_v3_page_documents_the_streaming_adapters_while_the_workflow_stays_paused() -> None:
    page = " ".join((ROOT / "docs/v3.md").read_text(encoding="utf-8").casefold().split())

    assert "streaming adapters are implemented" in page
    assert "no local splitter" in page
    assert "bounded streaming joins" in page
    assert "candidate pool and annotation workflow remain" in page
    assert "paused and not implemented" in page


def test_v3_page_documents_how_the_projections_are_checked_against_the_pinned_schemas() -> None:
    page = " ".join((ROOT / "docs/v3.md").read_text(encoding="utf-8").casefold().split())

    assert "v3_upstream_schema.json" in page
    assert "publishes no `is_lead` or `is_title` column" in page
    assert "publishes no `region`" in page
    assert "scripts/record_v3_schema.py" in page
    assert "scripts/streaming_smoke.py" in page


def test_v3_page_pins_the_same_revisions_and_website_records_as_the_settings() -> None:
    page = (ROOT / "docs/v3.md").read_text(encoding="utf-8")
    settings = V3Settings()

    for revision in (
        settings.description_dataset_revision,
        settings.wikipedia_dataset_revision,
        settings.website_dataset_revision,
    ):
        assert revision in page
    assert f"`{settings.website_config}`/`{settings.website_split}`" in page
    assert "`website_sentences` and `contact_website_sentences`" not in page


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


def test_qa_workflow_keeps_optional_model_dependencies_inside_docker_only() -> None:
    workflow = (ROOT / ".github/workflows/qa.yml").read_text(encoding="utf-8")

    assert "uv sync --locked" in workflow
    assert "uv sync --locked --extra models" not in workflow


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


def test_mutation_gate_covers_deterministic_sources() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        configuration = tomllib.load(project_file)

    source_paths = configuration["tool"]["mutmut"]["source_paths"]

    assert {
        "src/landuse_sentence_relevance/analysis",
        "src/landuse_sentence_relevance/domain",
        "src/landuse_sentence_relevance/storage",
        "src/landuse_sentence_relevance/sources/validation.py",
        "src/landuse_sentence_relevance/sources/website_text.py",
        "src/landuse_sentence_relevance/sources/website_discovery.py",
        "src/landuse_sentence_relevance/sources/v3.py",
    }.issubset(source_paths)

    selected_tests = configuration["tool"]["mutmut"]["pytest_add_cli_args_test_selection"]
    assert "tests/unit/analysis" in selected_tests
    assert "tests/unit/test_interrater_script.py" in selected_tests
    assert "tests/unit/sources/test_website_helpers.py" in selected_tests
    assert "tests/unit/sources/test_v3.py" in selected_tests
    assert "tests/unit/test_models.py" in selected_tests


def test_pytest_pythonpath_leaves_mutmut_source_precedence_intact() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        configuration = tomllib.load(project_file)

    assert configuration["tool"]["pytest"]["ini_options"]["pythonpath"] == ["."]


def test_mutation_gate_rebuilds_results_when_project_dependencies_change() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        configuration = tomllib.load(project_file)

    assert configuration["tool"]["mutmut"]["on_dependency_change"] == "rerun"


def test_mutation_cache_ignores_non_runtime_project_files() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        configuration = tomllib.load(project_file)

    assert configuration["tool"]["mutmut"]["cache_invalidation_exclude"] == [
        "mkdocs.yml",
        ".github/*",
        "Dockerfile",
        "CITATION.cff",
        "scripts/uv-seagate",
    ]


def test_mutation_gate_copies_qa_helpers_into_isolated_test_tree() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        configuration = tomllib.load(project_file)

    assert "scripts" in configuration["tool"]["mutmut"]["also_copy"]


def test_source_distribution_is_bounded_to_project_files() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        configuration = tomllib.load(project_file)

    sdist = configuration["tool"]["hatch"]["build"]["targets"]["sdist"]

    assert "src" in sdist["only-include"]
    assert "results/**" in sdist["exclude"]
    assert "state/**" in sdist["exclude"]
    assert "tmp/**" not in sdist["exclude"]
    assert "uv-environment/**" not in sdist["exclude"]
    assert "runtime-cache/**" not in sdist["exclude"]
    assert "site/**" in sdist["exclude"]


def test_mutation_gate_uses_every_core_by_default() -> None:
    from scripts.gauntlet import default_mutation_workers

    assert default_mutation_workers() >= 1
    assert default_mutation_workers() == max(1, os.cpu_count() or 1)


def test_gauntlet_passes_the_worker_count_to_mutmut_and_its_environment() -> None:
    source = (ROOT / "scripts/gauntlet.py").read_text(encoding="utf-8")

    assert '"--mutation-workers"' in source
    assert '["uv", "run", "mutmut", "run", "--max-children", workers]' in source
    assert 'environment["MUTMUT_MAX_CHILDREN"] = workers' in source


def test_committed_adjudicated_benchmark_is_the_complete_final_export() -> None:
    benchmark = ROOT / "data/benchmark/v2-adjudicated.csv"

    with benchmark.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "sentence",
        "label",
        "polygon_name",
        "h3_cell",
        "latitude",
        "longitude",
        "source",
        "region",
        "source_url",
    ]
    assert len(rows) == 154
    assert Counter(row["label"] for row in rows) == {"yes": 80, "no": 74}
    assert Counter(row["source"] for row in rows) == {"wikipedia": 100, "website": 54}
    assert len({row["sentence"] for row in rows}) == 154


def test_mutation_gate_copies_the_committed_data_the_guards_read() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        configuration = tomllib.load(project_file)

    assert "data" in configuration["tool"]["mutmut"]["also_copy"]
