import pytest

from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.sampling import (
    BoundedCandidatePool,
    FinalizedCandidatePool,
    _sorted_bucket,
)


def make_candidate(candidate_id: str, source: Source, cell: str) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        sentence=f"Sentence {candidate_id}.",
        source=source,
        source_record_id=candidate_id,
        source_field="text",
        h3_cell=cell,
        h3_resolution=3,
        latitude=0.0,
        longitude=0.0,
    )


def test_pool_caps_each_source_cell_deterministically() -> None:
    pool = BoundedCandidatePool(capacity_per_stratum=2, seed="test")

    for candidate_id in ("a", "b", "c"):
        pool.add(make_candidate(candidate_id, Source.WIKIPEDIA, "cell"))

    first = pool.snapshot()
    second = BoundedCandidatePool(capacity_per_stratum=2, seed="test")
    for candidate_id in ("c", "a", "b"):
        second.add(make_candidate(candidate_id, Source.WIKIPEDIA, "cell"))

    assert first == second.snapshot()
    assert len(first) == 2
    assert {candidate.candidate_id for candidate in first} == {"b", "c"}


def test_pool_seed_changes_the_bounded_reservoir() -> None:
    first = BoundedCandidatePool(capacity_per_stratum=2, seed="test")
    second = BoundedCandidatePool(capacity_per_stratum=2, seed="seed")
    for candidate_id in ("a", "b", "c"):
        candidate = make_candidate(candidate_id, Source.WIKIPEDIA, "cell")
        first.add(candidate)
        second.add(candidate)

    assert {candidate.candidate_id for candidate in first.snapshot()} == {"b", "c"}
    assert {candidate.candidate_id for candidate in second.snapshot()} == {"a", "c"}


def test_pool_requires_positive_capacity() -> None:
    with pytest.raises(ValueError, match=r"^capacity_per_stratum must be positive$"):
        BoundedCandidatePool(capacity_per_stratum=0, seed="test")


def test_capacity_one_is_a_valid_bounded_pool() -> None:
    pool = BoundedCandidatePool(capacity_per_stratum=1, seed="test")

    pool.add(make_candidate("candidate", Source.WIKIPEDIA, "cell"))

    assert [candidate.candidate_id for candidate in pool.snapshot()] == ["candidate"]


def test_pool_finalization_keeps_one_candidate_per_disjoint_source_cell() -> None:
    centers = {
        "wiki-west": (0.0, 0.0),
        "wiki-east": (0.0, 30.0),
        "wiki-extra": (0.0, 60.0),
        "web-west": (0.0, 90.0),
        "web-east": (0.0, 120.0),
        "web-extra": (0.0, 150.0),
    }
    pool = BoundedCandidatePool(capacity_per_stratum=2, seed="test")
    for source, cells in (
        (Source.WIKIPEDIA, ("wiki-west", "wiki-east", "wiki-extra")),
        (Source.WEBSITE, ("web-west", "web-east", "web-extra")),
    ):
        for cell in cells:
            pool.add(make_candidate(f"{source.value}-{cell}", source, cell))

    finalized = pool.finalize(
        target_cells_per_source=2,
        center_of_cell=centers.__getitem__,
        minimum_distance_km=1_000,
    )

    assert len(finalized.candidates) == 4
    assert len({candidate.h3_cell for candidate in finalized.candidates}) == 4
    assert len(finalized.cells) == 4
    assert {candidate.source for candidate in finalized.candidates} == set(Source)


def test_finalized_pool_rejects_more_than_one_sentence_per_h3_cell() -> None:
    candidate = make_candidate("wikipedia-a", Source.WIKIPEDIA, "shared-cell")

    with pytest.raises(
        ValueError,
        match=r"^candidate pool must contain one candidate per H3 cell$",
    ):
        FinalizedCandidatePool(
            (candidate, make_candidate("website-a", Source.WEBSITE, "shared-cell")),
            ("shared-cell",),
        )


def test_finalized_pool_rejects_duplicate_candidate_ids() -> None:
    candidate = make_candidate("duplicate", Source.WIKIPEDIA, "cell-a")

    with pytest.raises(ValueError, match=r"^candidate pool must contain unique candidate IDs$"):
        FinalizedCandidatePool((candidate, candidate), ("cell-a", "cell-a"))


def test_finalized_pool_rejects_cells_in_a_different_order() -> None:
    candidates = (
        make_candidate("wikipedia-a", Source.WIKIPEDIA, "cell-a"),
        make_candidate("website-b", Source.WEBSITE, "cell-b"),
    )

    with pytest.raises(ValueError, match=r"^candidate pool cells must match candidate order$"):
        FinalizedCandidatePool(candidates, ("cell-b", "cell-a"))


def test_pool_finalization_uses_center_function_and_seed() -> None:
    centers = {
        "a": (0.0, 0.0),
        "b": (0.0, 10.0),
        "c": (0.0, 20.0),
        "d": (0.0, 30.0),
    }
    first = BoundedCandidatePool(capacity_per_stratum=1, seed="test")
    second = BoundedCandidatePool(capacity_per_stratum=1, seed="other")
    for cell in centers:
        for source in Source:
            candidate = make_candidate(f"{source.value}-{cell}", source, cell)
            first.add(candidate)
            second.add(candidate)

    first_finalized = first.finalize(2, centers.__getitem__)
    second_finalized = second.finalize(2, centers.__getitem__)

    assert len(first_finalized.cells) == 4
    assert len(set(first_finalized.cells)) == 4
    assert len(second_finalized.cells) == 4
    assert len(set(second_finalized.cells)) == 4
    assert first_finalized.cells != second_finalized.cells


def test_snapshot_orders_cells_before_sources() -> None:
    pool = BoundedCandidatePool(capacity_per_stratum=1, seed="test")
    pool.add(make_candidate("wiki-a", Source.WIKIPEDIA, "a"))
    pool.add(make_candidate("web-b", Source.WEBSITE, "b"))

    assert [candidate.candidate_id for candidate in pool.snapshot()] == ["wiki-a", "web-b"]


def test_sorted_bucket_orders_candidates_by_id() -> None:
    candidates = {
        "b": make_candidate("b", Source.WIKIPEDIA, "cell"),
        "a": make_candidate("a", Source.WIKIPEDIA, "cell"),
    }

    assert [candidate.candidate_id for candidate in _sorted_bucket(candidates)] == ["a", "b"]


def test_pool_requires_enough_disjoint_source_cells() -> None:
    pool = BoundedCandidatePool(capacity_per_stratum=2, seed="test")
    pool.add(make_candidate("wiki", Source.WIKIPEDIA, "shared"))
    pool.add(make_candidate("web", Source.WEBSITE, "shared"))

    with pytest.raises(ValueError) as error:
        pool.finalize(target_cells_per_source=1, center_of_cell=lambda cell: (0.0, 0.0))

    assert str(error.value) == "need enough disjoint source cells"


def test_pool_finalization_rejects_unreachable_minimum_distance() -> None:
    centers = {
        "wiki-a": (0.0, 0.0),
        "wiki-b": (0.0, 0.001),
        "web-a": (0.0, 10.0),
        "web-b": (0.0, 10.001),
    }
    pool = BoundedCandidatePool(capacity_per_stratum=1, seed="test")
    for source, cells in (
        (Source.WIKIPEDIA, ("wiki-a", "wiki-b")),
        (Source.WEBSITE, ("web-a", "web-b")),
    ):
        for cell in cells:
            pool.add(make_candidate(f"{source.value}-{cell}", source, cell))

    with pytest.raises(ValueError) as error:
        pool.finalize(
            target_cells_per_source=2,
            center_of_cell=centers.__getitem__,
            minimum_distance_km=1_000,
        )

    assert str(error.value) == "cannot satisfy the minimum distance between H3 cells"


def test_next_unannotated_is_stable_and_skips_seen_ids() -> None:
    pool = BoundedCandidatePool(capacity_per_stratum=2, seed="test")
    pool.add(make_candidate("wiki", Source.WIKIPEDIA, "wiki-cell"))
    pool.add(make_candidate("web", Source.WEBSITE, "web-cell"))
    finalized = pool.finalize(target_cells_per_source=1, center_of_cell=lambda cell: (0.0, 0.0))

    first = finalized.next_unannotated(set())
    second = finalized.next_unannotated({"wiki"})
    assert first is not None
    assert second is not None
    assert first.candidate_id == "wiki"
    assert second.candidate_id == "web"
    assert finalized.next_unannotated({"web", "wiki"}) is None
