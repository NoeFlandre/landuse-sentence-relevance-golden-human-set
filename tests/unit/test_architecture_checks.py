from __future__ import annotations

from pathlib import Path

from scripts.check_architecture import architecture_violations

ROOT = Path(__file__).parents[2]


def test_project_architecture_has_no_violations() -> None:
    assert architecture_violations(ROOT) == ()


def test_architecture_check_reports_forbidden_layer_dependency(tmp_path: Path) -> None:
    package = tmp_path / "src" / "landuse_sentence_relevance"
    (package / "domain").mkdir(parents=True)
    (package / "storage").mkdir()
    (package / "domain" / "models.py").write_text(
        "from landuse_sentence_relevance.storage import atomic\n",
        encoding="utf-8",
    )
    (package / "storage" / "atomic.py").write_text("VALUE = 1\n", encoding="utf-8")

    violations = architecture_violations(tmp_path)

    assert violations == ("domain.models imports forbidden layer storage",)


def test_architecture_check_reports_circular_imports(tmp_path: Path) -> None:
    package = tmp_path / "src" / "landuse_sentence_relevance"
    (package / "domain").mkdir(parents=True)
    (package / "domain" / "first.py").write_text(
        "from landuse_sentence_relevance.domain.second import value\n",
        encoding="utf-8",
    )
    (package / "domain" / "second.py").write_text(
        "from landuse_sentence_relevance.domain.first import value\n",
        encoding="utf-8",
    )

    violations = architecture_violations(tmp_path)

    assert violations == (
        "circular import: landuse_sentence_relevance.domain.first -> "
        "landuse_sentence_relevance.domain.second -> landuse_sentence_relevance.domain.first",
    )
