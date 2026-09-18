from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

import landuse_sentence_relevance.bootstrap.v3 as bootstrap
from landuse_sentence_relevance.bootstrap import V3CandidatePoolResult
from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.domain.profile import V3_QUOTAS
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.domain.v3_preflight import preflight_v3_candidate_pool
from landuse_sentence_relevance.storage.v3_candidate_pool import load_v2_seed_plan
from tests.builders import make_candidate

ROOT = Path(__file__).parents[2]


def _pool() -> FinalizedCandidatePool:
    candidates = tuple(
        replace(
            make_candidate(f"fresh-{source.value}-{index:03d}"),
            source=source,
            h3_cell=f"fresh-{source.value}-{index:03d}",
        )
        for source in V3_QUOTAS.sources
        for index in range(400)
    )
    return FinalizedCandidatePool(candidates, tuple(candidate.h3_cell for candidate in candidates))


def _settings(tmp_path: Path) -> V3Settings:
    benchmark = tmp_path / "data" / "benchmark" / "v2-adjudicated.csv"
    benchmark.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "data" / "benchmark" / "v2-adjudicated.csv", benchmark)
    return V3Settings(data_root=tmp_path, benchmark_path=benchmark)


def _pool_result(settings: V3Settings) -> V3CandidatePoolResult:
    seed_plan = load_v2_seed_plan(settings.benchmark_path, quotas=V3_QUOTAS, seed=settings.seed)
    pool = _pool()
    return V3CandidatePoolResult(
        pool=pool,
        preflight=preflight_v3_candidate_pool(
            pool,
            seed_plan,
            quotas=V3_QUOTAS,
            candidate_cells_per_source=settings.candidate_cells_per_source,
        ),
        seed_plan=seed_plan,
    )


def test_build_v3_annotation_seed_persists_exact_quota_slots_without_labels(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    benchmark_before = settings.benchmark_path.read_bytes()
    result = _pool_result(settings)

    state = bootstrap.build_v3_annotation_seed(settings, pool_result=result)

    assert settings.benchmark_path.read_bytes() == benchmark_before
    assert settings.annotation_seed_path.is_file()
    assert state.total_rows == 300
    assert Counter(row.candidate.source for row in state.seeded_rows) == {
        Source.WIKIPEDIA: 99,
        Source.WEBSITE: 54,
    }
    assert Counter(row.candidate.source for row in state.pending_rows) == {
        Source.WIKIPEDIA: 1,
        Source.WEBSITE: 46,
        Source.DESCRIPTION: 100,
    }
    assert state.pending_counts == result.seed_plan.remaining
    assert all(row.annotation is None for row in state.pending_rows)
    assert len(state.excluded_v2_reasons) == 1
    assert next(iter(state.excluded_v2_reasons.values()))

    payload = json.loads(settings.annotation_seed_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["benchmark"]["sha256"]
    assert payload["metadata"]["candidate_pool"]["candidate_count"] == 1200


def test_build_v3_annotation_seed_resumes_from_the_saved_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    result = _pool_result(settings)
    first = bootstrap.build_v3_annotation_seed(settings, pool_result=result)

    def fail_if_selected(*args, **kwargs):
        raise AssertionError("a persisted V3 seed must be reused")

    monkeypatch.setattr(bootstrap, "select_v3_annotation_seed", fail_if_selected)

    assert bootstrap.build_v3_annotation_seed(settings, pool_result=result) == first


def test_v3_annotation_seed_rejects_a_changed_benchmark_on_resume(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    result = _pool_result(settings)
    bootstrap.build_v3_annotation_seed(settings, pool_result=result)
    settings.benchmark_path.write_bytes(settings.benchmark_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="benchmark SHA-256"):
        bootstrap.build_v3_annotation_seed(settings, pool_result=result)


def test_v3_settings_reserves_a_distinct_annotation_seed_path(tmp_path: Path) -> None:
    settings = V3Settings(data_root=tmp_path)

    assert settings.annotation_seed_path == tmp_path / "results/annotations/seeds/v3.json"
    assert settings.annotation_seed_path != settings.session_path
