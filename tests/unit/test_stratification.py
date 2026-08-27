from collections import Counter

import pytest

import landuse_sentence_relevance.domain.stratification as stratification
from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.domain.stratification import (
    _append_next_source_cell,
    _distance_to_selection,
    _distant_cells,
    _farthest_or_stable_cell,
    _feasible_source_cells,
    _haversine_km,
    _stable_key,
    select_common_cells,
    select_distinct_source_cells,
    select_spread_cells,
)


def test_distinct_source_cell_selection_is_disjoint_and_globally_spread() -> None:
    centers = {
        "wiki-west": (0.0, 0.0),
        "wiki-east": (0.0, 30.0),
        "website-west": (0.0, 60.0),
        "website-east": (0.0, 90.0),
        "shared": (0.0, 120.0),
    }
    selected = select_distinct_source_cells(
        {
            Source.WIKIPEDIA: {"wiki-west", "wiki-east", "shared"},
            Source.WEBSITE: {"website-west", "website-east", "shared"},
        },
        target_count_per_source=2,
        center_of_cell=centers.__getitem__,
        seed="test",
        minimum_distance_km=1_000,
    )

    wikipedia_cells = set(selected[Source.WIKIPEDIA])
    website_cells = set(selected[Source.WEBSITE])
    assert len(wikipedia_cells) == 2
    assert len(website_cells) == 2
    assert wikipedia_cells.isdisjoint(website_cells)
    assert set(selected[Source.WIKIPEDIA] + selected[Source.WEBSITE]) != {"shared"}
    assert (
        min(
            _haversine_km(centers[first], centers[second])
            for first in wikipedia_cells | website_cells
            for second in wikipedia_cells | website_cells
            if first != second
        )
        >= 1_000
    )


def test_distinct_source_cell_selection_uses_zero_distance_by_default() -> None:
    centers = {
        "wiki-a": (0.0, 0.0),
        "wiki-b": (0.0, 0.001),
        "web-a": (0.0, 10.0),
        "web-b": (0.0, 10.001),
    }

    selected = select_distinct_source_cells(
        {
            Source.WIKIPEDIA: {"wiki-a", "wiki-b"},
            Source.WEBSITE: {"web-a", "web-b"},
        },
        target_count_per_source=2,
        center_of_cell=centers.__getitem__,
        seed="test",
    )

    assert sum(len(cells) for cells in selected.values()) == 4


def test_feasible_source_cells_reserves_cells_for_the_other_source() -> None:
    available = {
        Source.WIKIPEDIA: {"wiki-a", "wiki-b", "shared"},
        Source.WEBSITE: {"shared"},
    }

    assert _feasible_source_cells(
        Source.WIKIPEDIA,
        available,
        {source: [] for source in Source},
        [],
        target_count=1,
    ) == ["wiki-a", "wiki-b"]


def test_feasible_source_cells_rejects_an_insufficient_current_source() -> None:
    available = {Source.WIKIPEDIA: {"wiki-a"}, Source.WEBSITE: {"web-a", "web-b"}}

    assert (
        _feasible_source_cells(
            Source.WIKIPEDIA,
            available,
            {source: [] for source in Source},
            [],
            target_count=2,
        )
        == []
    )


def test_feasible_source_cells_rejects_an_insufficient_other_source() -> None:
    available = {Source.WIKIPEDIA: {"wiki-a", "wiki-b"}, Source.WEBSITE: {"web-a"}}

    assert (
        _feasible_source_cells(
            Source.WIKIPEDIA,
            available,
            {source: [] for source in Source},
            [],
            target_count=2,
        )
        == []
    )


def test_distinct_source_cell_selection_rejects_insufficient_disjoint_cells() -> None:
    with pytest.raises(ValueError, match="disjoint source cells"):
        select_distinct_source_cells(
            {
                Source.WIKIPEDIA: {"shared", "wiki-only"},
                Source.WEBSITE: {"shared"},
            },
            target_count_per_source=2,
            center_of_cell=lambda cell: (0.0, 0.0),
            seed="test",
        )


def test_distinct_source_cell_selection_rejects_an_unreachable_minimum_distance() -> None:
    centers = {
        "wiki-west": (0.0, 0.0),
        "wiki-east": (0.0, 10.0),
        "web-west": (0.0, 20.0),
        "web-east": (0.0, 30.0),
    }

    with pytest.raises(ValueError, match="minimum distance"):
        select_distinct_source_cells(
            {
                Source.WIKIPEDIA: {"wiki-west", "wiki-east"},
                Source.WEBSITE: {"web-west", "web-east"},
            },
            target_count_per_source=2,
            center_of_cell=centers.__getitem__,
            seed="test",
            minimum_distance_km=20_000,
        )


def test_distinct_source_cell_selection_rejects_negative_minimum_distance() -> None:
    with pytest.raises(ValueError) as error:
        select_distinct_source_cells(
            {source: {"cell"} for source in Source},
            target_count_per_source=1,
            center_of_cell=lambda cell: (0.0, 0.0),
            seed="test",
            minimum_distance_km=-1,
        )

    assert str(error.value) == "minimum_distance_km must not be negative"


def test_spread_cell_selection_returns_all_cells_when_target_is_larger() -> None:
    centers = {"a": (0.0, 0.0), "b": (0.0, 10.0)}

    assert select_spread_cells(centers, 3, centers.__getitem__, "test") == ("a", "b")


def test_spread_cell_selection_does_not_call_centers_at_an_exact_target() -> None:
    centers = {"a": (0.0, 0.0), "b": (0.0, 10.0)}

    def unexpected_center(cell: str) -> tuple[float, float]:
        raise AssertionError(f"center lookup was unexpected for {cell}")

    assert select_spread_cells(centers, 2, unexpected_center, "test") == ("a", "b")


def test_spread_cell_selection_uses_the_seed_for_its_first_cell() -> None:
    centers = {"a": (0.0, 0.0), "b": (0.0, 10.0), "c": (0.0, 20.0)}

    assert select_spread_cells(centers, 1, centers.__getitem__, "one") != select_spread_cells(
        centers, 1, centers.__getitem__, "three"
    )


def test_common_cell_selection_rejects_an_insufficient_intersection() -> None:
    with pytest.raises(ValueError, match="shared cells"):
        select_common_cells(
            {source: {"shared"} for source in Source},
            target_count=2,
            center_of_cell=lambda cell: (0.0, 0.0),
            seed="test",
        )


def test_distant_cells_keep_a_candidate_at_the_exact_distance_boundary() -> None:
    centers = {"selected": (0.0, 0.0), "candidate": (0.0, 1.0)}
    distance = _haversine_km(centers["selected"], centers["candidate"])

    assert _distant_cells(["candidate"], ["selected"], centers, distance) == ["candidate"]


def test_distant_cells_use_zero_distance_without_filtering() -> None:
    class UnexpectedCenters(dict[str, tuple[float, float]]):
        def __getitem__(self, cell: str) -> tuple[float, float]:
            raise AssertionError(f"center lookup was unexpected for {cell}")

    assert _distant_cells(["candidate"], ["selected"], UnexpectedCenters(), 0.0) == ["candidate"]


def test_append_next_source_cell_uses_the_farthest_candidate() -> None:
    available = {
        Source.WIKIPEDIA: {"anchor", "near", "far"},
        Source.WEBSITE: {"web-a", "web-b"},
    }
    selected_by_source = {Source.WIKIPEDIA: ["anchor"], Source.WEBSITE: []}
    selected_cells = ["anchor"]
    centers = {
        "anchor": (0.0, 0.0),
        "near": (0.0, 1.0),
        "far": (0.0, 10.0),
        "web-a": (0.0, 20.0),
        "web-b": (0.0, 30.0),
    }

    _append_next_source_cell(
        Source.WIKIPEDIA,
        available,
        selected_by_source,
        selected_cells,
        centers,
        target_count=2,
        minimum_distance_km=0.0,
        seed="a",
    )

    assert selected_by_source[Source.WIKIPEDIA] == ["anchor", "far"]
    assert selected_cells == ["anchor", "far"]


def test_farthest_cell_selection_uses_distance_and_seed_tiebreaking() -> None:
    centers = {"selected": (0.0, 0.0), "near": (0.0, 1.0), "far": (0.0, 10.0)}

    assert _farthest_or_stable_cell(["near", "far"], ["selected"], centers, "test") == "far"

    tie_centers = {"selected": (0.0, 0.0), "left": (0.0, -1.0), "right": (0.0, 1.0)}
    expected = max(
        ("left", "right"),
        key=lambda cell: (
            _haversine_km(tie_centers[cell], tie_centers["selected"]),
            _stable_key("one", cell),
        ),
    )
    assert _farthest_or_stable_cell(["left", "right"], ["selected"], tie_centers, "one") == expected


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


def test_common_cell_selection_resolves_each_cell_center_once() -> None:
    centers = {f"cell-{index}": (0.0, float(index)) for index in range(6)}
    calls: Counter[str] = Counter()

    def center_of_cell(cell: str) -> tuple[float, float]:
        calls[cell] += 1
        return centers[cell]

    select_common_cells(
        {source: centers for source in Source},
        target_count=4,
        center_of_cell=center_of_cell,
        seed="test",
    )

    assert calls == Counter({cell: 1 for cell in centers})


def test_common_cell_selection_updates_nearest_distances_incrementally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    centers = {f"cell-{index}": (0.0, float(index)) for index in range(6)}
    calls = 0
    original_distance = stratification._haversine_km

    def distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        nonlocal calls
        calls += 1
        return original_distance(first, second)

    monkeypatch.setattr(stratification, "_haversine_km", distance)
    selected = select_common_cells(
        {source: centers for source in Source},
        target_count=4,
        center_of_cell=centers.__getitem__,
        seed="test",
    )

    assert calls == 14
    assert selected == ("cell-5", "cell-0", "cell-2", "cell-3")

    tie_centers = {"a": (0.0, 0.0), "b": (0.0, 1.0), "c": (0.0, -1.0)}
    assert select_common_cells(
        {source: tie_centers for source in Source},
        target_count=2,
        center_of_cell=tie_centers.__getitem__,
        seed="two",
    ) == ("a", "b")
    equal_distances = {cell: 1.0 for cell in ("a", "b", "c")}
    assert stratification._farthest_cell(["a", "b", "c"], equal_distances, "one") == "c"
    assert stratification._farthest_cell(["a", "b", "c"], equal_distances, "test") == "a"


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
