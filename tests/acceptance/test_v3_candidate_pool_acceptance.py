from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from pytest_bdd import given, scenarios, then, when
from tests.unit.test_models import make_candidate

from landuse_sentence_relevance.bootstrap import V3CandidatePoolResult, build_v3_candidate_pool
from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.profile import V3_SOURCES, SourceLabelQuotas, balanced_quotas
from landuse_sentence_relevance.sources.v3 import V3SourceAdapters

scenarios("features/v3_candidate_pool.feature")
pytestmark = pytest.mark.acceptance


@dataclass(frozen=True, slots=True)
class _StaticSource:
    candidates: tuple[Candidate, ...]

    def iter_candidates(self) -> Iterator[Candidate]:
        return iter(self.candidates)


@dataclass(slots=True)
class _Scenario:
    settings: V3Settings
    quotas: SourceLabelQuotas
    adapters: V3SourceAdapters
    centers: dict[str, tuple[float, float]]
    result: V3CandidatePoolResult | None = None


def _candidate(source: Source, index: int, cell: str | None = None) -> Candidate:
    return replace(
        make_candidate(f"acceptance-{source.value}-{index}"),
        source=source,
        h3_cell=cell or f"{source.value}-cell-{index}",
    )


def _adapters() -> V3SourceAdapters:
    return V3SourceAdapters(
        description=_StaticSource(tuple(_candidate(Source.DESCRIPTION, index) for index in range(2))),
        wikipedia=_StaticSource(
            (
                _candidate(Source.WIKIPEDIA, 0, "v2-cell"),
                _candidate(Source.WIKIPEDIA, 1),
                _candidate(Source.WIKIPEDIA, 2),
            )
        ),
        website=_StaticSource(tuple(_candidate(Source.WEBSITE, index) for index in range(2))),
    )


def _make_scenario(tmp_path: Path) -> _Scenario:
    benchmark = tmp_path / "v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        "seed,yes,Seed,v2-cell,1,2,wikipedia,region,https://example.test/seed\n",
        encoding="utf-8",
    )
    candidates = (
        _candidate(Source.WIKIPEDIA, 1),
        _candidate(Source.WIKIPEDIA, 2),
        _candidate(Source.WEBSITE, 0),
        _candidate(Source.WEBSITE, 1),
        _candidate(Source.DESCRIPTION, 0),
        _candidate(Source.DESCRIPTION, 1),
    )
    return _Scenario(
        settings=V3Settings(
            data_root=tmp_path,
            benchmark_path=benchmark,
            candidate_cells_per_source=2,
        ),
        quotas=balanced_quotas(V3_SOURCES, rows_per_source_label=1),
        adapters=_adapters(),
        centers={candidate.h3_cell: (0.0, float(index)) for index, candidate in enumerate(candidates)},
    )


@given("a V2 seed and three bounded V3 source streams")
def scenario_has_sources(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    request.node._v3_scenario = _make_scenario(tmp_path)  # type: ignore[attr-defined]


@when("I build the V3 candidate reservoir")
def build_reservoir(request: pytest.FixtureRequest) -> None:
    scenario: _Scenario = request.node._v3_scenario  # type: ignore[attr-defined]
    scenario.result = build_v3_candidate_pool(
        scenario.settings,
        quotas=scenario.quotas,
        adapters=scenario.adapters,
        cell_for_location=lambda latitude, longitude: "unused",
        center_of_cell=scenario.centers.__getitem__,
    )


@then("every V3 source has two fresh cells in the preflight")
def sources_have_expected_cells(request: pytest.FixtureRequest) -> None:
    scenario: _Scenario = request.node._v3_scenario  # type: ignore[attr-defined]
    assert scenario.result is not None
    assert scenario.result.preflight.candidate_cells_by_source == {source: 2 for source in V3_SOURCES}


@then("no selected cell is occupied by V2")
def selected_cells_are_fresh(request: pytest.FixtureRequest) -> None:
    scenario: _Scenario = request.node._v3_scenario  # type: ignore[attr-defined]
    assert scenario.result is not None
    assert "v2-cell" not in scenario.result.pool.cells


@when("I rerun the V3 candidate reservoir")
def rerun_reservoir(request: pytest.FixtureRequest) -> None:
    scenario: _Scenario = request.node._v3_scenario  # type: ignore[attr-defined]

    @dataclass(frozen=True, slots=True)
    class _FailingSource:
        def iter_candidates(self) -> Iterator[Candidate]:
            raise AssertionError("cached V3 reservoir reopened an upstream stream")

    cached = build_v3_candidate_pool(
        scenario.settings,
        quotas=scenario.quotas,
        adapters=V3SourceAdapters(
            description=_FailingSource(),
            wikipedia=_FailingSource(),
            website=_FailingSource(),
        ),
        cell_for_location=lambda latitude, longitude: "unused",
        center_of_cell=scenario.centers.__getitem__,
    )
    scenario.result = cached


@then("the saved reservoir is reused without reading source streams")
def cached_reservoir_is_available(request: pytest.FixtureRequest) -> None:
    scenario: _Scenario = request.node._v3_scenario  # type: ignore[attr-defined]
    assert scenario.result is not None
    assert len(scenario.result.pool.candidates) == 6
