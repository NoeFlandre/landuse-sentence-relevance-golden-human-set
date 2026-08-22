import pytest

from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.domain.stratification import (
    _distance_to_selection,
    _haversine_km,
    _stable_key,
    select_common_cells,
)


def test_common_cell_selection_is_deterministic_and_spread_out() -> None:
    centers = {
        "a": (0.0, 0.0),
        "b": (0.0, 10.0),
        "c": (0.0, 20.0),
        "d": (0.0, 30.0),
        "e": (0.0, 40.0),
    }
    eligible = {source: centers for source in Source}

    first = select_common_cells(eligible, target_count=3, center_of_cell=centers.__getitem__, seed="test")
    second = select_common_cells(eligible, target_count=3, center_of_cell=centers.__getitem__, seed="test")

    assert first == second
    assert len(first) == 3
    assert max(centers[cell][1] for cell in first) - min(centers[cell][1] for cell in first) >= 20


def test_common_cell_selection_uses_only_the_intersection() -> None:
    centers = {"shared": (0.0, 0.0), "wikipedia-only": (0.0, 10.0), "website-only": (0.0, 20.0)}

    selected = select_common_cells(
        {
            Source.WIKIPEDIA: {"shared", "wikipedia-only"},
            Source.WEBSITE: {"shared", "website-only"},
        },
        target_count=1,
        center_of_cell=centers.__getitem__,
        seed="test",
    )

    assert selected == ("shared",)


def test_common_cell_selection_uses_the_seed_for_the_first_cell() -> None:
    centers = {"a": (0.0, 0.0), "b": (0.0, 10.0), "c": (0.0, 20.0)}
    eligible = {source: centers for source in Source}

    assert select_common_cells(eligible, 1, centers.__getitem__, "one") == ("a",)
    assert select_common_cells(eligible, 1, centers.__getitem__, "three") == ("b",)


def test_common_cell_selection_uses_the_seed_for_equal_distance_ties() -> None:
    centers = {"a": (0.0, 0.0), "b": (0.0, 1.0), "c": (0.0, -1.0)}
    eligible = {source: centers for source in Source}

    assert select_common_cells(eligible, 2, centers.__getitem__, "one") == ("a", "c")
    assert select_common_cells(eligible, 2, centers.__getitem__, "two") == ("a", "b")


def test_haversine_distance_uses_coordinate_differences_and_earth_radius() -> None:
    assert _haversine_km((0.0, 0.0), (0.0, 1.0)) == pytest.approx(111.1950802335329)
    assert _haversine_km((10.0, 20.0), (30.0, 40.0)) == pytest.approx(3040.607017927688)
    assert _haversine_km((0.0, 0.0), (180.0, 0.0)) == pytest.approx(20015.114442035923)


def test_distance_to_selection_returns_distance_and_seed_tiebreaker() -> None:
    centers = {"selected": (0.0, 0.0), "candidate": (0.0, 1.0)}

    distance, tie_breaker = _distance_to_selection("candidate", ["selected"], centers.__getitem__, "seed-a")

    assert distance == pytest.approx(111.1950802335329)
    assert tie_breaker == _stable_key("seed-a", "candidate")
    assert tie_breaker != _distance_to_selection("candidate", ["selected"], centers.__getitem__, "seed-b")[1]


def test_common_cell_selection_requires_a_positive_target() -> None:
    with pytest.raises(ValueError, match=r"^target_count must be positive$"):
        select_common_cells({}, target_count=0, center_of_cell=lambda cell: (0.0, 0.0), seed="test")
