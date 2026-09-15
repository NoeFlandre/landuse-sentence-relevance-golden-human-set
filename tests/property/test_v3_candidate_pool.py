from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.v3_pool import V3_SOURCES, V3CandidateReservoir
from tests.unit.test_models import make_candidate

pytestmark = pytest.mark.property


def candidate(identifier: str, source: Source, cell: str, latitude: float, longitude: float) -> Candidate:
    return replace(
        make_candidate(identifier),
        source=source,
        source_record_id=identifier,
        h3_cell=cell,
        latitude=latitude,
        longitude=longitude,
    )


@given(st.lists(st.integers(min_value=0, max_value=40), min_size=1, max_size=20, unique=True))
def test_v3_reservoir_is_bounded_and_permutation_invariant(indexes: list[int]) -> None:
    rows = [
        candidate(f"description-{index}", Source.DESCRIPTION, f"cell-{index}", float(index), 0.0)
        for index in indexes
    ]
    first = V3CandidateReservoir(capacity_per_source=5, seed="property")
    second = V3CandidateReservoir(capacity_per_source=5, seed="property")
    for row in rows:
        first.add(row)
    for row in reversed(rows):
        second.add(row)

    assert first.snapshot() == second.snapshot()
    assert len(first.snapshot()) <= 5
    assert len({row.h3_cell for row in first.snapshot()}) == len(first.snapshot())


@given(st.integers(min_value=1, max_value=4))
def test_v3_finalization_keeps_one_global_cell_per_source(index_count: int) -> None:
    rows_by_source = {
        source: tuple(
            candidate(
                f"{source.value}-{index}",
                source,
                f"{source.value}-{index}",
                float(index),
                float(index),
            )
            for index in range(index_count)
        )
        for source in V3_SOURCES
    }
    centers = {row.h3_cell: (row.latitude, row.longitude) for rows in rows_by_source.values() for row in rows}
    reservoir = V3CandidateReservoir(
        capacity_per_source=index_count,
        seed="property",
    )
    for rows in rows_by_source.values():
        for row in rows:
            reservoir.add(row)

    pool = reservoir.finalize(
        target_cells_per_source=index_count,
        center_of_cell=centers.__getitem__,
    )

    assert len(pool.candidates) == index_count * len(V3_SOURCES)
    assert len({row.h3_cell for row in pool.candidates}) == len(pool.candidates)
