from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

import landuse_sentence_relevance.bootstrap as bootstrap
import landuse_sentence_relevance.bootstrap.v3 as bootstrap_v3
import landuse_sentence_relevance.config as config
import landuse_sentence_relevance.web.app as app
from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.domain.models import Annotation, Label, Source
from landuse_sentence_relevance.domain.profile import balanced_quotas
from landuse_sentence_relevance.domain.v3_annotation import (
    V3AnnotationSeed,
    V3SeedRow,
    V3SelectionMetadata,
)
from landuse_sentence_relevance.storage.session import AnnotationStore
from landuse_sentence_relevance.workflow import V3AnnotationWorkflow
from tests.builders import make_candidate


def _forbidden(version: str) -> Callable[[object], None]:
    def fail(settings: object) -> None:
        raise AssertionError(f"{version} must not be built")

    return fail


def _seed() -> V3AnnotationSeed:
    quotas = balanced_quotas((Source.WIKIPEDIA,), rows_per_source_label=1)
    pending = replace(make_candidate("wiki-pending"), source=Source.WIKIPEDIA, h3_cell="pending-cell")
    seeded = replace(make_candidate("wiki-seeded"), source=Source.WIKIPEDIA, h3_cell="seeded-cell")
    return V3AnnotationSeed(
        rows=(
            V3SeedRow(
                candidate=pending,
                quota_source=Source.WIKIPEDIA,
                quota_label=Label.YES,
                origin="v3",
                annotation=None,
                selection=V3SelectionMetadata(seed="entrypoint", rank="0" * 64, slot_index=0),
            ),
            V3SeedRow(
                candidate=seeded,
                quota_source=Source.WIKIPEDIA,
                quota_label=Label.NO,
                origin="v2",
                annotation=Annotation(seeded, Label.NO),
                selection=V3SelectionMetadata(seed="entrypoint", rank="1" * 64, slot_index=1),
            ),
        ),
        excluded_v2_rows=(),
        reserved_v2_cells=frozenset({"seeded-cell"}),
        quotas=quotas,
        benchmark_sha256="0" * 64,
        seed="entrypoint",
    )


def test_build_v3_workflow_resumes_from_the_seed_and_persists_to_the_v3_session_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = V3Settings(data_root=tmp_path)
    received: list[V3Settings] = []

    def build_seed(received_settings: V3Settings) -> V3AnnotationSeed:
        received.append(received_settings)
        return _seed()

    monkeypatch.setattr(bootstrap_v3, "build_v3_annotation_seed", build_seed)

    workflow = bootstrap_v3.build_v3_workflow(settings)

    assert received == [settings]
    assert isinstance(workflow, V3AnnotationWorkflow)
    current = workflow.current_candidate()
    assert current is not None
    assert current.candidate_id == "wiki-pending"
    assert workflow.state().total_count == 2

    workflow.annotate("wiki-pending", Label.YES)

    assert settings.session_path.is_file()
    assert settings.session_path != tmp_path / "results/annotations/sessions/v2-wikipedia.jsonl"
    assert tuple(AnnotationStore(settings.session_path).load()) == ("wiki-pending",)


def test_build_v3_workflow_refuses_to_start_when_seed_preflight_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = V3Settings(data_root=tmp_path)

    def fail_preflight(received_settings: V3Settings) -> V3AnnotationSeed:
        assert received_settings is settings
        raise bootstrap_v3.V3PreflightError("candidate preflight failed")

    monkeypatch.setattr(bootstrap_v3, "build_v3_annotation_seed", fail_preflight)

    with pytest.raises(bootstrap_v3.V3PreflightError, match="candidate preflight failed"):
        bootstrap_v3.build_v3_workflow(settings)


def test_annotation_entrypoint_selects_v3_only_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[object] = []
    monkeypatch.setenv("ANNOTATION_VERSION", "v3")
    monkeypatch.setenv("PROJECT_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(bootstrap, "build_v3_workflow", lambda settings: calls.append(settings))
    monkeypatch.setattr(bootstrap, "build_workflow", _forbidden("v2"))

    app.build_annotation_workflow()

    assert calls == [V3Settings.from_env({"PROJECT_DATA_ROOT": str(tmp_path)})]


def test_annotation_entrypoint_keeps_v2_as_the_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[object] = []
    monkeypatch.delenv("ANNOTATION_VERSION", raising=False)
    monkeypatch.setenv("PROJECT_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(bootstrap, "build_workflow", lambda settings: calls.append(settings))
    monkeypatch.setattr(bootstrap, "build_v3_workflow", _forbidden("v3"))

    app.build_annotation_workflow()

    assert calls == [config.Settings.from_env({"PROJECT_DATA_ROOT": str(tmp_path)})]


def test_annotation_entrypoint_rejects_an_unknown_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANNOTATION_VERSION", "v4")

    with pytest.raises(ValueError, match="ANNOTATION_VERSION must be v2 or v3"):
        app.build_annotation_workflow()
