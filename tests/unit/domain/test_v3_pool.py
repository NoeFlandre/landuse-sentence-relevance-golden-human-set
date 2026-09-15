from __future__ import annotations

from dataclasses import replace

import pytest
from tests.unit.test_models import make_candidate

import landuse_sentence_relevance.domain.v3_pool as v3_pool
from landuse_sentence_relevance.domain.models import Annotation, Candidate, Label, Source
from landuse_sentence_relevance.domain.profile import SourceLabelQuotas
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.domain.v3_pool import (
    V3CandidateReservoir,
    V3PreflightError,
    preflight_v3_candidate_pool,
)


def candidate(identifier: str, source: Source, cell: str, latitude: float) -> Candidate:
    return replace(
        make_candidate(identifier),
        source=source,
        source_record_id=identifier,
        h3_cell=cell,
        latitude=latitude,
        longitude=0.0,
    )


def test_v3_reservoir_skips_reserved_cells_and_keeps_one_candidate_per_cell() -> None:
    reservoir = V3CandidateReservoir(
        capacity_per_source=4,
        seed="v3-test",
        reserved_cells={"reserved"},
    )

    assert reservoir.add(candidate("reserved", Source.DESCRIPTION, "reserved", 0.0)) is False
    assert reservoir.add(candidate("description-a", Source.DESCRIPTION, "a", 1.0)) is True
    reservoir.add(candidate("description-a-duplicate", Source.DESCRIPTION, "a", 1.0))

    snapshot = reservoir.snapshot()

    assert all(row.h3_cell != "reserved" for row in snapshot)
    assert {row.h3_cell for row in snapshot} == {"a"}


def test_v3_reservoir_is_order_independent_for_a_fixed_seed() -> None:
    rows = [
        candidate("description-a", Source.DESCRIPTION, "a", 1.0),
        candidate("description-b", Source.DESCRIPTION, "b", 2.0),
        candidate("description-c", Source.DESCRIPTION, "c", 3.0),
    ]

    first = V3CandidateReservoir(capacity_per_source=2, seed="v3-test")
    second = V3CandidateReservoir(capacity_per_source=2, seed="v3-test")
    for row in rows:
        first.add(row)
    for row in reversed(rows):
        second.add(row)

    assert first.snapshot() == second.snapshot()


def test_v3_candidate_rank_depends_on_seed_and_candidate_id() -> None:
    first = candidate("description-a", Source.DESCRIPTION, "a", 1.0)
    second = candidate("description-b", Source.DESCRIPTION, "b", 1.0)

    assert v3_pool._candidate_key("v3-test", first)[0] != v3_pool._candidate_key("other", first)[0]
    assert v3_pool._candidate_key("v3-test", first)[0] != v3_pool._candidate_key("v3-test", second)[0]


def test_v3_reservoir_seed_changes_duplicate_selection() -> None:
    rows = (
        candidate("description-0", Source.DESCRIPTION, "same", 0.0),
        candidate("description-2", Source.DESCRIPTION, "same", 1.0),
    )
    first = V3CandidateReservoir(capacity_per_source=1, seed="v3-test")
    second = V3CandidateReservoir(capacity_per_source=1, seed="other")

    for row in rows:
        first.add(row)
        second.add(row)

    assert first.snapshot()[0].candidate_id != second.snapshot()[0].candidate_id


def test_v3_reservoir_keeps_the_existing_candidate_on_an_equal_rank(monkeypatch) -> None:
    bucket: dict[str, Candidate] = {}
    first = candidate("description-first", Source.DESCRIPTION, "same", 0.0)
    second = candidate("description-second", Source.DESCRIPTION, "same", 1.0)

    v3_pool._retain_candidate(bucket, first, "v3-test", capacity=1)
    monkeypatch.setattr(v3_pool, "_candidate_key", lambda _seed, _candidate: ("same", "", ""))
    v3_pool._retain_candidate(bucket, second, "v3-test", capacity=1)

    assert bucket["same"] is first


def test_v3_reservoir_discards_the_highest_ranked_cell_when_full() -> None:
    reservoir = V3CandidateReservoir(capacity_per_source=2, seed="v3-test")
    for index, cell in enumerate(("cell-a", "cell-b", "cell-c")):
        reservoir.add(candidate(f"description-{index}", Source.DESCRIPTION, cell, float(index)))

    assert {row.candidate_id for row in reservoir.snapshot()} == {
        "description-0",
        "description-2",
    }


def test_v3_reservoir_validates_capacity_and_source_configuration() -> None:
    with pytest.raises(ValueError, match=r"^capacity_per_source must be positive$"):
        V3CandidateReservoir(capacity_per_source=0, seed="v3-test")
    with pytest.raises(ValueError, match=r"^sources must not be empty$"):
        V3CandidateReservoir(capacity_per_source=1, seed="v3-test", sources=())
    with pytest.raises(ValueError, match=r"^sources must be unique$"):
        V3CandidateReservoir(
            capacity_per_source=1,
            seed="v3-test",
            sources=(Source.DESCRIPTION, Source.DESCRIPTION),
        )


def test_v3_reservoir_rejects_sources_outside_the_profile() -> None:
    reservoir = V3CandidateReservoir(
        capacity_per_source=1,
        seed="v3-test",
        sources=(Source.DESCRIPTION,),
    )
    foreign_candidate = candidate("wikipedia", Source.WIKIPEDIA, "wiki", 0.0)

    with pytest.raises(ValueError, match=r"^candidate source is outside the V3 profile: wikipedia$"):
        reservoir.add(foreign_candidate)
    with pytest.raises(ValueError, match=r"^candidate source is outside the V3 profile: wikipedia$"):
        reservoir.candidates_for_source(Source.WIKIPEDIA)


def test_v3_reservoir_passes_seed_and_distance_to_geographic_selection(monkeypatch) -> None:
    calls: list[tuple[str, float, tuple[Source, ...]]] = []

    def select(
        eligible_cells,
        *,
        target_count_per_source,
        center_of_cell,
        seed,
        minimum_distance_km,
        sources,
    ):
        del target_count_per_source, center_of_cell
        calls.append((seed, minimum_distance_km, sources))
        return {source: (sorted(eligible_cells[source])[0],) for source in sources}

    monkeypatch.setattr(v3_pool, "select_distinct_source_cells", select)
    reservoir = V3CandidateReservoir(
        capacity_per_source=2,
        seed="v3-finalize",
        sources=(Source.DESCRIPTION,),
    )
    reservoir.add(candidate("description-a", Source.DESCRIPTION, "a", 0.0))
    reservoir.add(candidate("description-b", Source.DESCRIPTION, "b", 1.0))

    pool = reservoir.finalize(
        target_cells_per_source=1,
        center_of_cell=lambda cell: (0.0, float(ord(cell))),
        minimum_distance_km=12.5,
    )

    assert calls == [("v3-finalize", 12.5, (Source.DESCRIPTION,))]
    assert pool.cells == ("a",)


def test_v3_reservoir_rejects_a_target_larger_than_its_capacity() -> None:
    reservoir = V3CandidateReservoir(capacity_per_source=1, seed="v3-test")

    with pytest.raises(ValueError, match=r"^target_cells_per_source must not exceed reservoir capacity$"):
        reservoir.finalize(target_cells_per_source=2, center_of_cell=lambda _cell: (0.0, 0.0))


@pytest.mark.parametrize(
    ("target", "capacity", "message"),
    (
        (0, None, "target_cells_per_source must be positive"),
        (2, 1, "target_cells_per_source must not exceed reservoir capacity"),
    ),
)
def test_v3_target_validation_messages_are_explicit(target, capacity, message) -> None:
    with pytest.raises(ValueError, match=rf"^{message}$"):
        v3_pool._validate_target_cells(target, capacity)


def test_v3_reservoir_finalizes_a_disjoint_geographically_spread_pool() -> None:
    centers = {
        "description-west": (0.0, -120.0),
        "description-east": (0.0, 120.0),
        "description-extra": (45.0, 0.0),
        "wikipedia-west": (-45.0, -60.0),
        "wikipedia-east": (45.0, 60.0),
        "wikipedia-extra": (0.0, 0.0),
        "website-west": (-30.0, -30.0),
        "website-east": (30.0, 30.0),
        "website-extra": (0.0, 180.0),
    }
    reservoir = V3CandidateReservoir(capacity_per_source=3, seed="v3-test")
    for source in (Source.WIKIPEDIA, Source.WEBSITE, Source.DESCRIPTION):
        for cell, (latitude, longitude) in centers.items():
            if cell.startswith(source.value):
                reservoir.add(
                    replace(candidate(f"{source.value}-{cell}", source, cell, latitude), longitude=longitude)
                )

    pool = reservoir.finalize(
        target_cells_per_source=2,
        center_of_cell=centers.__getitem__,
    )

    assert len(pool.candidates) == 6
    assert len({row.h3_cell for row in pool.candidates}) == 6
    assert {row.source for row in pool.candidates} == {
        Source.WIKIPEDIA,
        Source.WEBSITE,
        Source.DESCRIPTION,
    }


def test_v3_preflight_rejects_a_candidate_in_a_reserved_v2_cell() -> None:
    v2_candidate = candidate("v2", Source.WIKIPEDIA, "reserved", 0.0)
    fresh_candidate = candidate("fresh", Source.DESCRIPTION, "fresh", 1.0)
    pool = FinalizedCandidatePool(
        candidates=(v2_candidate, fresh_candidate),
        cells=("reserved", "fresh"),
    )
    quotas = SourceLabelQuotas(
        {
            (Source.WIKIPEDIA, Label.YES): 1,
            (Source.DESCRIPTION, Label.YES): 1,
        }
    )
    existing = (Annotation(candidate=v2_candidate, label=Label.YES),)

    with pytest.raises(V3PreflightError, match="reserved V2 H3 cell"):
        preflight_v3_candidate_pool(
            pool,
            existing_v2=existing,
            quotas=quotas,
            target_cells_per_source=1,
            seed="v3-test",
        )


def test_v3_preflight_checks_the_exact_remaining_source_quota() -> None:
    rows = tuple(
        candidate(f"{source.value}-{index}", source, f"{source.value}-{index}", float(index))
        for source in (Source.WIKIPEDIA, Source.WEBSITE, Source.DESCRIPTION)
        for index in range(1)
    )
    pool = FinalizedCandidatePool(rows, tuple(row.h3_cell for row in rows))
    quotas = SourceLabelQuotas(
        {
            (source, label): 2
            for source in (Source.WIKIPEDIA, Source.WEBSITE, Source.DESCRIPTION)
            for label in (Label.YES, Label.NO)
        }
    )

    with pytest.raises(V3PreflightError, match="need 4 new"):
        preflight_v3_candidate_pool(
            pool,
            existing_v2=(),
            quotas=quotas,
            target_cells_per_source=1,
            seed="v3-test",
        )


def test_v3_preflight_reports_counts_cells_remaining_quotas_and_reserved_cells(monkeypatch) -> None:
    sources = (Source.WIKIPEDIA, Source.WEBSITE, Source.DESCRIPTION)
    rows = tuple(
        candidate(f"{source.value}-{index}", source, f"{source.value}-{index}", float(index))
        for source in sources
        for index in (1, 2)
    )
    pool = FinalizedCandidatePool(rows, tuple(row.h3_cell for row in rows))
    quotas = SourceLabelQuotas({(source, label): 1 for source in sources for label in Label})
    existing_row = Annotation(candidate("v2", Source.WIKIPEDIA, "reserved", 0.0), label=Label.YES)
    original_plan_seed = v3_pool.plan_seed
    seen_seeds: list[str] = []

    def plan_seed_spy(existing, requested_quotas, requested_seed):
        seen_seeds.append(requested_seed)
        return original_plan_seed(existing, requested_quotas, requested_seed)

    monkeypatch.setattr(v3_pool, "plan_seed", plan_seed_spy)
    report = preflight_v3_candidate_pool(
        pool,
        existing_v2=(existing_row,),
        quotas=quotas,
        target_cells_per_source=2,
        seed="preflight-seed",
    )

    assert seen_seeds == ["preflight-seed"]
    assert report.candidate_count_by_source == {source: 2 for source in sources}
    assert report.candidate_cells_by_source == {
        source: frozenset({f"{source.value}-1", f"{source.value}-2"}) for source in sources
    }
    assert report.required_new_by_source == {
        Source.WIKIPEDIA: 1,
        Source.WEBSITE: 2,
        Source.DESCRIPTION: 2,
    }
    assert report.remaining_quotas == {
        (Source.WIKIPEDIA, Label.NO): 1,
        (Source.WEBSITE, Label.YES): 1,
        (Source.WEBSITE, Label.NO): 1,
        (Source.DESCRIPTION, Label.YES): 1,
        (Source.DESCRIPTION, Label.NO): 1,
    }
    assert report.reserved_cells == frozenset({"reserved"})
    assert report.total_candidates == 6


def test_v3_preflight_rejects_reserved_cell_previews_exactly() -> None:
    for cells, expected in (
        (("c", "a", "b"), "candidate pool contains a reserved V2 H3 cell: a, b, c"),
        (("d", "a", "c", "b"), "candidate pool contains a reserved V2 H3 cell: a, b, c..."),
    ):
        rows = tuple(
            candidate(f"description-{cell}", Source.DESCRIPTION, cell, float(index))
            for index, cell in enumerate(cells)
        )
        pool = FinalizedCandidatePool(rows, tuple(cell for cell in cells))

        with pytest.raises(V3PreflightError) as error:
            v3_pool._reject_reserved_cells(pool, frozenset(cells))
        assert str(error.value) == expected


def test_v3_preflight_rejects_unexpected_sources_with_names() -> None:
    rows = (
        candidate("wikipedia", Source.WIKIPEDIA, "wiki", 0.0),
        candidate("website", Source.WEBSITE, "web", 1.0),
    )
    pool = FinalizedCandidatePool(rows, tuple(row.h3_cell for row in rows))

    with pytest.raises(V3PreflightError) as error:
        v3_pool._reject_unexpected_sources(pool, (Source.DESCRIPTION,))
    assert str(error.value) == (
        "candidate pool contains sources outside the V3 quota matrix: website, wikipedia"
    )


def test_v3_preflight_accepts_exact_source_boundaries() -> None:
    v3_pool._reject_source_shortfalls(
        {Source.DESCRIPTION: 1},
        {Source.DESCRIPTION: 1},
        (Source.DESCRIPTION,),
        target_cells_per_source=1,
    )


def test_v3_preflight_reports_an_exact_quota_shortfall() -> None:
    with pytest.raises(V3PreflightError) as error:
        v3_pool._reject_source_shortfalls(
            {Source.DESCRIPTION: 1},
            {Source.DESCRIPTION: 2},
            (Source.DESCRIPTION,),
            target_cells_per_source=1,
        )
    assert str(error.value) == (
        "V3 description source has 1 candidates; need 2 new candidates for its exact quota"
    )


def test_v3_preflight_reports_a_fresh_candidate_target_shortfall() -> None:
    with pytest.raises(V3PreflightError) as error:
        v3_pool._reject_source_shortfalls(
            {Source.DESCRIPTION: 1},
            {Source.DESCRIPTION: 1},
            (Source.DESCRIPTION,),
            target_cells_per_source=2,
        )
    assert str(error.value) == ("V3 description source has 1 candidates; need at least 2 fresh candidates")
