from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from pytest_bdd import given, scenarios, then, when

import landuse_sentence_relevance.bootstrap.v3 as bootstrap
from landuse_sentence_relevance.bootstrap import V3CandidatePoolResult
from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.domain.models import Label, Source
from landuse_sentence_relevance.domain.profile import V3_QUOTAS
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.domain.v3_annotation import V3AnnotationSeed
from landuse_sentence_relevance.domain.v3_preflight import preflight_v3_candidate_pool
from landuse_sentence_relevance.storage.v3_candidate_pool import load_v2_seed_plan
from tests.builders import make_candidate

scenarios("features/v3_annotation_seed.feature")
pytestmark = pytest.mark.acceptance

ROOT = Path(__file__).parents[2]


@dataclass(slots=True)
class _Scenario:
    settings: V3Settings
    pool_result: V3CandidatePoolResult
    state: V3AnnotationSeed | None = None


def _pool() -> FinalizedCandidatePool:
    candidates = tuple(
        replace(
            make_candidate(f"acceptance-seed-{source.value}-{index:03d}"),
            source=source,
            h3_cell=f"acceptance-seed-{source.value}-cell-{index:03d}",
        )
        for source in V3_QUOTAS.sources
        for index in range(400)
    )
    return FinalizedCandidatePool(candidates, tuple(candidate.h3_cell for candidate in candidates))


def _scenario(tmp_path: Path) -> _Scenario:
    benchmark = tmp_path / "data" / "benchmark" / "v2-adjudicated.csv"
    benchmark.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "data" / "benchmark" / "v2-adjudicated.csv", benchmark)
    settings = V3Settings(data_root=tmp_path, benchmark_path=benchmark)
    seed_plan = load_v2_seed_plan(settings.benchmark_path, quotas=V3_QUOTAS, seed=settings.seed)
    pool = _pool()
    return _Scenario(
        settings=settings,
        pool_result=V3CandidatePoolResult(
            pool=pool,
            preflight=preflight_v3_candidate_pool(
                pool,
                seed_plan,
                quotas=V3_QUOTAS,
                candidate_cells_per_source=settings.candidate_cells_per_source,
            ),
            seed_plan=seed_plan,
        ),
    )


@given("the frozen V2 benchmark and a finalized V3 candidate reservoir")
def scenario_has_a_benchmark_and_pool(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    request.node._v3_scenario = _scenario(tmp_path)  # type: ignore[attr-defined]


@when("I build the V3 annotation seed")
def build_seed(request: pytest.FixtureRequest) -> None:
    scenario: _Scenario = request.node._v3_scenario  # type: ignore[attr-defined]
    scenario.state = bootstrap.build_v3_annotation_seed(scenario.settings, pool_result=scenario.pool_result)


def _state(request: pytest.FixtureRequest) -> V3AnnotationSeed:
    scenario: _Scenario = request.node._v3_scenario  # type: ignore[attr-defined]
    if scenario.state is None:
        raise AssertionError("the V3 annotation seed has not been built")
    return scenario.state


@then("the seed has 300 rows and exact source-by-label quotas")
def seed_has_exact_quotas(request: pytest.FixtureRequest) -> None:
    state = _state(request)
    assert state.total_rows == 300
    assert state.rows_by_source == {source: 100 for source in V3_QUOTAS.sources}
    assert state.quota_counts == {
        (source, label): 50 for source in V3_QUOTAS.sources for label in (Label.YES, Label.NO)
    }
    assert state.pending_row_count == 147
    assert all(row.annotation is None for row in state.pending_rows)


@then("every V2 cell is reserved and the excluded surplus is explained")
def v2_cells_are_reserved(request: pytest.FixtureRequest) -> None:
    state = _state(request)
    assert len(state.reserved_v2_cells) == 154
    assert len(state.excluded_v2_rows) == 1
    assert set(state.excluded_v2_reasons) == {
        annotation.candidate.candidate_id for annotation in state.excluded_v2_rows
    }
    assert all(reason.strip() for reason in state.excluded_v2_reasons.values())
    assert Counter(row.candidate.source for row in state.seeded_rows) == {
        Source.WIKIPEDIA: 99,
        Source.WEBSITE: 54,
    }


@when("I rerun the V3 annotation seed")
def rerun_seed(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario: _Scenario = request.node._v3_scenario  # type: ignore[attr-defined]

    def fail_if_selected(*args: object, **kwargs: object) -> None:
        raise AssertionError("the persisted V3 seed must be reused")

    monkeypatch.setattr(bootstrap, "select_v3_annotation_seed", fail_if_selected)
    scenario.state = bootstrap.build_v3_annotation_seed(
        scenario.settings,
        pool_result=scenario.pool_result,
    )


@then("the saved seed is reused without selecting new rows")
def saved_seed_is_reused(request: pytest.FixtureRequest) -> None:
    state = _state(request)
    scenario: _Scenario = request.node._v3_scenario  # type: ignore[attr-defined]
    assert state == bootstrap.build_v3_annotation_seed(
        scenario.settings,
        pool_result=scenario.pool_result,
    )
