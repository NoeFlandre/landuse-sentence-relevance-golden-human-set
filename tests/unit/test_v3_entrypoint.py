from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import landuse_sentence_relevance.bootstrap as bootstrap
import landuse_sentence_relevance.bootstrap.v3 as bootstrap_v3
import landuse_sentence_relevance.config as config
import landuse_sentence_relevance.web.app as app
from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.domain.v3_annotation import V3AnnotationSeed
from landuse_sentence_relevance.storage.session import AnnotationStore


def test_build_v3_workflow_uses_the_seed_and_v3_session_path(monkeypatch, tmp_path: Path) -> None:
    settings = V3Settings(data_root=tmp_path)
    sentinel_seed = cast(V3AnnotationSeed, SimpleNamespace(total_rows=6, pending_row_count=5))
    sentinel_workflow = object()
    captured: dict[str, object] = {}

    def build_seed(received_settings: V3Settings) -> V3AnnotationSeed:
        assert received_settings is settings
        return sentinel_seed

    def build_workflow(seed: V3AnnotationSeed, store: object) -> object:
        captured["seed"] = seed
        captured["store"] = store
        return sentinel_workflow

    monkeypatch.setattr(bootstrap_v3, "build_v3_annotation_seed", build_seed)
    monkeypatch.setattr(bootstrap_v3, "V3AnnotationWorkflow", build_workflow)

    result = bootstrap_v3.build_v3_workflow(settings)

    assert result is sentinel_workflow
    assert captured["seed"] is sentinel_seed
    assert cast(AnnotationStore, captured["store"])._path == settings.session_path
    assert settings.session_path != tmp_path / "results/annotations/sessions/v2-wikipedia.jsonl"


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = object()
    monkeypatch.setenv("ANNOTATION_VERSION", "v3")

    def build_v3(settings: V3Settings) -> object:
        assert isinstance(settings, V3Settings)
        return marker

    monkeypatch.setattr(bootstrap, "build_v3_workflow", build_v3)

    assert app.build_annotation_workflow() is marker


def test_annotation_entrypoint_keeps_v2_as_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = object()
    monkeypatch.delenv("ANNOTATION_VERSION", raising=False)

    def build_v2(settings: config.Settings) -> object:
        assert isinstance(settings, config.Settings)
        return marker

    monkeypatch.setattr(bootstrap, "build_workflow", build_v2)

    assert app.build_annotation_workflow() is marker


def test_annotation_entrypoint_rejects_an_unknown_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANNOTATION_VERSION", "v4")

    with pytest.raises(ValueError, match="ANNOTATION_VERSION must be v2 or v3"):
        app.build_annotation_workflow()
