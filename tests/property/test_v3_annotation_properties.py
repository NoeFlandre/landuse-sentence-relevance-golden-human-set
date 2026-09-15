from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.unit.test_models import make_candidate

from landuse_sentence_relevance.domain.models import Label
from landuse_sentence_relevance.domain.profile import V3_SOURCES, balanced_quotas
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.domain.seeding import plan_seed
from landuse_sentence_relevance.domain.v3_annotation import select_v3_annotation_seed

pytestmark = pytest.mark.property

_PROPERTY_SETTINGS = settings(max_examples=30, derandomize=True, database=None, deadline=None)


def _candidates() -> tuple:
    return tuple(
        replace(
            make_candidate(f"property-{source.value}-{index:02d}"),
            source=source,
            h3_cell=f"property-{source.value}-cell-{index:02d}",
        )
        for source in V3_SOURCES
        for index in range(4)
    )


def _select(candidates: tuple):
    quotas = balanced_quotas(V3_SOURCES, rows_per_source_label=1)
    return select_v3_annotation_seed(
        plan_seed((), quotas, seed="property-seed"),
        FinalizedCandidatePool(candidates, tuple(candidate.h3_cell for candidate in candidates)),
        quotas=quotas,
        seed="property-seed",
        benchmark_sha256="b" * 64,
    )


@_PROPERTY_SETTINGS
@given(st.permutations(tuple(range(12))))
def test_v3_source_by_label_selection_is_order_independent_and_globally_unique(
    permutation: list[int],
) -> None:
    candidates = _candidates()
    first = _select(candidates)
    second = _select(tuple(candidates[index] for index in permutation))

    assert second == first
    assert first.total_rows == 6
    assert first.quota_counts == {
        (source, label): 1 for source in V3_SOURCES for label in (Label.YES, Label.NO)
    }
    assert Counter(row.candidate.source for row in first.pending_rows) == {source: 2 for source in V3_SOURCES}
    assert all(row.annotation is None for row in first.pending_rows)
    assert len({row.candidate.h3_cell for row in first.rows}) == first.total_rows
