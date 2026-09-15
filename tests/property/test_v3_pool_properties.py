from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.unit.test_models import make_candidate

from landuse_sentence_relevance.domain.models import Candidate
from landuse_sentence_relevance.domain.profile import V3_SOURCES
from landuse_sentence_relevance.domain.sampling import BoundedCandidatePool

pytestmark = pytest.mark.property

_PROPERTY_SETTINGS = settings(max_examples=40, derandomize=True, database=None, deadline=None)


@st.composite
def source_candidate_sets(draw: st.DrawFn) -> tuple[int, tuple[Candidate, ...]]:
    target = draw(st.integers(min_value=1, max_value=5))
    extra = draw(st.integers(min_value=0, max_value=4))
    candidates = tuple(
        replace(
            make_candidate(f"{source.value}-{index}"),
            source=source,
            h3_cell=f"{source.value}-cell-{index}",
        )
        for source in V3_SOURCES
        for index in range(target + extra)
    )
    return target, candidates


def _finalize(target: int, candidates: tuple[Candidate, ...]):
    pool = BoundedCandidatePool(capacity_per_stratum=1, seed="property", sources=V3_SOURCES)
    for candidate in candidates:
        pool.add(candidate)
    centers = {
        candidate.h3_cell: (0.0, float(candidate.h3_cell.rsplit("-", maxsplit=1)[-1]))
        for candidate in candidates
    }
    return pool.finalize(target, centers.__getitem__)


@_PROPERTY_SETTINGS
@given(source_candidate_sets())
def test_v3_pool_is_order_independent_and_globally_unique(
    generated: tuple[int, tuple[Candidate, ...]],
) -> None:
    target, candidates = generated
    first = _finalize(target, candidates)
    second = _finalize(target, tuple(reversed(candidates)))

    assert first == second
    assert len(first.candidates) == target * len(V3_SOURCES)
    assert len(first.cells) == len(set(first.cells))
    assert Counter(candidate.source for candidate in first.candidates) == {
        source: target for source in V3_SOURCES
    }
    assert {candidate.h3_cell for candidate in first.candidates} == set(first.cells)
