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
