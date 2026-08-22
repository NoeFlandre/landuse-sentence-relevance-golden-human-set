import pytest

from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.sampling import BoundedCandidatePool, _sorted_bucket


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


def test_pool_finalization_keeps_only_shared_cells() -> None:
    pool = BoundedCandidatePool(capacity_per_stratum=3, seed="test")
    pool.add(make_candidate("wiki-shared", Source.WIKIPEDIA, "shared"))
    pool.add(make_candidate("web-shared", Source.WEBSITE, "shared"))
    pool.add(make_candidate("wiki-only", Source.WIKIPEDIA, "wiki-only"))
    pool.add(make_candidate("web-only", Source.WEBSITE, "web-only"))

    finalized = pool.finalize(
        target_cell_count=1,
        center_of_cell=lambda cell: {"shared": (0.0, 0.0)}[cell],
    )

    assert {candidate.candidate_id for candidate in finalized.candidates} == {"wiki-shared", "web-shared"}
    assert finalized.cells == ("shared",)


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

    assert first_finalized.cells == ("b", "d")
    assert second_finalized.cells == ("d", "a")


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


def test_pool_requires_enough_common_cells() -> None:
    pool = BoundedCandidatePool(capacity_per_stratum=2, seed="test")
    pool.add(make_candidate("wiki", Source.WIKIPEDIA, "wiki-only"))
    pool.add(make_candidate("web", Source.WEBSITE, "web-only"))

    with pytest.raises(ValueError, match="shared cells"):
        pool.finalize(target_cell_count=1, center_of_cell=lambda cell: (0.0, 0.0))


def test_next_unannotated_is_stable_and_skips_seen_ids() -> None:
    pool = BoundedCandidatePool(capacity_per_stratum=2, seed="test")
    pool.add(make_candidate("wiki", Source.WIKIPEDIA, "cell"))
    pool.add(make_candidate("web", Source.WEBSITE, "cell"))
    finalized = pool.finalize(target_cell_count=1, center_of_cell=lambda cell: (0.0, 0.0))

    first = finalized.next_unannotated(set())
    second = finalized.next_unannotated({"wiki"})
    assert first is not None
    assert second is not None
    assert first.candidate_id == "wiki"
    assert second.candidate_id == "web"
    assert finalized.next_unannotated({"web", "wiki"}) is None
