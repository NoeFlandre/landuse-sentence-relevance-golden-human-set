import math
from collections import Counter
from collections.abc import Mapping

import pytest

import landuse_sentence_relevance.domain.stratification as stratification
from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.domain.stratification import (
    DEFAULT_SOURCES,
    _append_next_source_cell,
    _cached_distant_cells,
    _conflict_counts,
    _count_bucket_conflicts,
    _cube_bin,
    _distance_to_selection,
    _distant_candidates,
    _distant_cells,
    _farthest_cached_cell,
    _farthest_or_stable_cell,
    _feasible_source_cells,
    _haversine_km,
    _next_source_cell,
    _required_distant_candidates,
    _stable_key,
    _unit_vector,
    _update_selected_distance_cache,
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


def test_distinct_source_cell_selection_avoids_a_greedy_dead_end() -> None:
    centers = {
        **{f"wikipedia-{index}": (0.0, 5.0 * longitude) for index, longitude in enumerate((0, 1, 2, 6))},
        **{f"website-{index}": (0.0, 5.0 * longitude) for index, longitude in enumerate((3, 4, 5, 7))},
    }

    selected = select_distinct_source_cells(
        {
            Source.WIKIPEDIA: {f"wikipedia-{index}" for index in range(4)},
            Source.WEBSITE: {f"website-{index}" for index in range(4)},
        },
        target_count_per_source=2,
        center_of_cell=centers.__getitem__,
        seed="test",
        minimum_distance_km=1_000,
    )

    cells = selected[Source.WIKIPEDIA] + selected[Source.WEBSITE]
    assert len(cells) == 4
    assert len(set(cells)) == 4
    assert (
        min(
            _haversine_km(centers[first], centers[second])
            for first in cells
            for second in cells
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


def test_distinct_source_cell_selection_updates_minimum_distances_incrementally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    centers = {
        **{f"wiki-{index}": (0.0, -150.0 + index * 50.0) for index in range(4)},
        **{f"web-{index}": (0.0, -125.0 + index * 50.0) for index in range(4)},
    }
    calls = 0
    original_haversine = stratification._haversine_km

    def counted_haversine(first: tuple[float, float], second: tuple[float, float]) -> float:
        nonlocal calls
        calls += 1
        return original_haversine(first, second)

    monkeypatch.setattr(stratification, "_haversine_km", counted_haversine)

    selected = select_distinct_source_cells(
        {
            Source.WIKIPEDIA: {f"wiki-{index}" for index in range(4)},
            Source.WEBSITE: {f"web-{index}" for index in range(4)},
        },
        target_count_per_source=2,
        center_of_cell=centers.__getitem__,
        seed="test",
        minimum_distance_km=1_000,
    )

    assert sum(len(cells) for cells in selected.values()) == 4
    assert calls == 22


def test_distinct_source_cell_selection_keeps_the_distance_cache_at_zero_distance() -> None:
    """The cache is not only for distance filtering: the farthest-point choice needs it.

    This previously asserted the opposite, on the reading that a zero minimum
    distance has no distances to check. It does: every round still picks the cell
    farthest from the current selection. Without the cache that distance is
    recomputed against the whole selection each round, which is quadratic in the
    target count and made a real 400-cell build take tens of minutes. Maintaining
    it changes the cost, not the chosen cells.
    """

    centers = {
        "wiki-a": (0.0, 0.0),
        "wiki-b": (0.0, 1.0),
        "web-a": (0.0, 10.0),
        "web-b": (0.0, 11.0),
    }
    updates: list[str] = []
    real_update = stratification._update_selected_distance_cache

    def recording_update(
        cell_centers: Mapping[str, tuple[float, float]],
        nearest_distances: dict[str, float],
        selected_cells: list[str],
        selected: str,
    ) -> None:
        updates.append(selected)
        real_update(cell_centers, nearest_distances, selected_cells, selected)

    monkeypatch_attr(stratification, "_update_selected_distance_cache", recording_update)
    try:
        selected = select_distinct_source_cells(
            {
                Source.WIKIPEDIA: {"wiki-a", "wiki-b"},
                Source.WEBSITE: {"web-a", "web-b"},
            },
            target_count_per_source=2,
            center_of_cell=centers.__getitem__,
            seed="test",
            minimum_distance_km=0.0,
        )
    finally:
        monkeypatch_attr(stratification, "_update_selected_distance_cache", real_update)

    assert updates, "the cache must be maintained even when no distance filter applies"
    assert sorted(selected[Source.WIKIPEDIA]) == ["wiki-a", "wiki-b"]
    assert sorted(selected[Source.WEBSITE]) == ["web-a", "web-b"]


def test_conflict_counts_return_zero_for_zero_distance() -> None:
    centers = {"a": (0.0, 0.0), "b": (0.0, 0.001)}

    assert _conflict_counts(centers, 0.0) == {"a": 0, "b": 0}


def test_conflict_counts_count_nearby_cells_for_any_positive_distance() -> None:
    centers = {"a": (0.0, 0.0), "b": (0.0, 0.001)}

    assert _conflict_counts(centers, 1.0) == {"a": 1, "b": 1}


def test_conflict_counts_use_a_conservative_result_at_the_maximum_distance() -> None:
    centers = {"a": (0.0, 0.0), "b": (0.0, 180.0), "c": (30.0, 30.0)}
    maximum_distance = math.pi * stratification._EARTH_RADIUS_KM

    assert _conflict_counts(centers, maximum_distance) == {"a": 2, "b": 2, "c": 2}


def test_spatial_conflict_counts_use_the_chord_radius_for_bucketing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    centers = {"a": (0.0, 0.0)}
    received: list[float] = []
    original_buckets = stratification._spatial_buckets

    def capture_buckets(
        cells: tuple[str, ...],
        vectors: dict[str, tuple[float, float, float]],
        width: float,
    ) -> dict[tuple[int, int, int], list[str]]:
        received.append(width)
        return original_buckets(cells, vectors, width)

    monkeypatch.setattr(stratification, "_spatial_buckets", capture_buckets)

    _conflict_counts(centers, 1_000.0)

    assert received == [2 * math.sin(1_000.0 / stratification._EARTH_RADIUS_KM / 2)]


def test_count_bucket_conflicts_skip_only_cells_before_the_current_cell() -> None:
    centers = {"a": (0.0, 0.0), "b": (0.0, 0.001), "c": (0.0, 0.002)}
    counts = dict.fromkeys(centers, 0)

    _count_bucket_conflicts("b", centers, ["a", "c"], 1.0, counts)

    assert counts == {"a": 0, "b": 1, "c": 1}


def test_count_bucket_conflicts_keep_the_strict_distance_boundary() -> None:
    centers = {"a": (0.0, 0.0), "b": (0.0, 1.0)}
    counts = {"a": 0, "b": 0}
    boundary = _haversine_km(centers["a"], centers["b"])

    _count_bucket_conflicts("a", centers, ["b"], boundary, counts)

    assert counts == {"a": 0, "b": 0}


def test_count_bucket_conflicts_increment_each_endpoint_once() -> None:
    centers = {"a": (0.0, 0.0), "b": (0.0, 0.001)}
    counts = {"a": 4, "b": 7}

    _count_bucket_conflicts("a", centers, ["b"], 1.0, counts)

    assert counts == {"a": 5, "b": 8}


def test_unit_vector_preserves_all_three_spherical_components() -> None:
    latitude = math.radians(45.0)
    longitude = math.radians(60.0)

    assert _unit_vector((45.0, 60.0)) == pytest.approx(
        (
            math.cos(latitude) * math.cos(longitude),
            math.cos(latitude) * math.sin(longitude),
            math.sin(latitude),
        )
    )


def test_cube_bin_offsets_each_vector_component_by_one_before_scaling() -> None:
    assert _cube_bin((-0.4, 0.1, 0.6), 0.2) == (2, 5, 8)


def test_distinct_source_cell_selection_does_not_build_conflicts_at_zero_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_conflict_counts(
        centers: dict[str, tuple[float, float]], minimum_distance_km: float
    ) -> dict[str, int]:
        raise AssertionError("zero minimum distance must not build conflict counts")

    monkeypatch.setattr(stratification, "_conflict_counts", unexpected_conflict_counts)

    select_distinct_source_cells(
        {
            Source.WIKIPEDIA: {"wiki-a", "wiki-b"},
            Source.WEBSITE: {"web-a", "web-b"},
        },
        target_count_per_source=2,
        center_of_cell=lambda cell: (0.0, float(len(cell))),
        seed="test",
        minimum_distance_km=0.0,
    )


def test_distinct_source_cell_selection_builds_conflicts_for_one_kilometer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[float] = []
    original_conflict_counts = stratification._conflict_counts

    def capture_conflict_counts(
        centers: dict[str, tuple[float, float]], minimum_distance_km: float
    ) -> dict[str, int]:
        received.append(minimum_distance_km)
        return original_conflict_counts(centers, minimum_distance_km)

    monkeypatch.setattr(stratification, "_conflict_counts", capture_conflict_counts)

    select_distinct_source_cells(
        {
            Source.WIKIPEDIA: {"wiki-a", "wiki-b"},
            Source.WEBSITE: {"web-a", "web-b"},
        },
        target_count_per_source=1,
        center_of_cell=lambda cell: (0.0, float(len(cell))),
        seed="test",
        minimum_distance_km=1.0,
    )

    assert received == [1.0]


def test_append_next_source_cell_passes_distance_cache_to_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = {Source.WIKIPEDIA: {"wiki"}, Source.WEBSITE: {"web"}}
    selected_by_source = {source: [] for source in DEFAULT_SOURCES}
    selected_cells: list[str] = []
    centers = {"wiki": (0.0, 0.0), "web": (0.0, 10.0)}
    nearest_distances = {"wiki": 100.0}
    received: list[dict[str, float] | None] = []

    def capture_next_cell(
        candidates: list[str],
        selected: list[str],
        candidate_centers: dict[str, tuple[float, float]],
        seed: str,
        distances: dict[str, float] | None,
        conflict_counts: dict[str, int] | None,
    ) -> str:
        received.append(distances)
        return candidates[0]

    monkeypatch.setattr(stratification, "_next_source_cell", capture_next_cell)

    _append_next_source_cell(
        Source.WIKIPEDIA,
        available,
        selected_by_source,
        selected_cells,
        centers,
        target_count=1,
        minimum_distance_km=0.0,
        seed="test",
        nearest_distances=nearest_distances,
    )

    assert received == [nearest_distances]


def test_append_next_source_cell_passes_all_selection_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = {Source.WIKIPEDIA: {"wiki"}, Source.WEBSITE: {"web"}}
    selected_by_source = {source: [] for source in DEFAULT_SOURCES}
    selected_cells: list[str] = []
    centers = {"wiki": (0.0, 0.0), "web": (0.0, 10.0)}
    nearest_distances = {"wiki": 100.0}
    conflict_counts = {"wiki": 0}
    received: tuple[object, ...] | None = None

    def capture_next_cell(
        candidates: list[str],
        selected: list[str],
        candidate_centers: dict[str, tuple[float, float]],
        received_seed: str,
        distances: dict[str, float] | None,
        received_conflicts: dict[str, int] | None,
    ) -> str:
        nonlocal received
        received = (
            candidates,
            list(selected),
            candidate_centers,
            received_seed,
            distances,
            received_conflicts,
        )
        return candidates[0]

    monkeypatch.setattr(stratification, "_next_source_cell", capture_next_cell)

    _append_next_source_cell(
        Source.WIKIPEDIA,
        available,
        selected_by_source,
        selected_cells,
        centers,
        target_count=1,
        minimum_distance_km=0.0,
        seed="distinct-seed",
        nearest_distances=nearest_distances,
        conflict_counts=conflict_counts,
    )

    assert received == (
        ["wiki"],
        [],
        centers,
        "distinct-seed",
        nearest_distances,
        conflict_counts,
    )
    assert selected_by_source[Source.WIKIPEDIA] == ["wiki"]
    assert selected_cells == ["wiki"]


def test_conflict_ties_use_the_requested_seed() -> None:
    centers = {"a": (0.0, 0.0), "b": (0.0, 1.0)}
    conflicts = {"a": 0, "b": 0}

    assert _next_source_cell(["b", "a"], [], centers, "seed", None, conflicts) == "a"


def test_distinct_source_cell_selection_caches_any_positive_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    centers = {
        "wiki-a": (0.0, 0.0),
        "wiki-b": (0.0, 1.0),
        "web-a": (0.0, 10.0),
        "web-b": (0.0, 11.0),
    }
    cache_updates = 0
    original_update = stratification._update_selected_distance_cache

    def counted_cache_update(
        cache_centers: dict[str, tuple[float, float]],
        nearest_distances: dict[str, float],
        selected_cells: list[str],
        selected: str,
    ) -> None:
        nonlocal cache_updates
        cache_updates += 1
        original_update(cache_centers, nearest_distances, selected_cells, selected)

    monkeypatch.setattr(stratification, "_update_selected_distance_cache", counted_cache_update)

    select_distinct_source_cells(
        {
            Source.WIKIPEDIA: {"wiki-a", "wiki-b"},
            Source.WEBSITE: {"web-a", "web-b"},
        },
        target_count_per_source=2,
        center_of_cell=centers.__getitem__,
        seed="test",
        minimum_distance_km=1.0,
    )

    assert cache_updates == 4


def test_append_next_source_cell_passes_centers_to_distance_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = {Source.WIKIPEDIA: {"wiki"}, Source.WEBSITE: {"web"}}
    selected_by_source = {source: [] for source in DEFAULT_SOURCES}
    selected_cells: list[str] = []
    centers = {"wiki": (0.0, 0.0), "web": (0.0, 10.0)}
    received: dict[str, object] = {}

    def capture_distance_filter(
        candidates: list[str],
        selected: list[str],
        filter_centers: dict[str, tuple[float, float]],
        minimum_distance: float,
        nearest_distances: dict[str, float] | None,
    ) -> list[str]:
        received["centers"] = filter_centers
        return candidates

    monkeypatch.setattr(stratification, "_required_distant_candidates", capture_distance_filter)

    _append_next_source_cell(
        Source.WIKIPEDIA,
        available,
        selected_by_source,
        selected_cells,
        centers,
        target_count=1,
        minimum_distance_km=0.0,
        seed="test",
    )

    assert received["centers"] is centers


def test_required_distant_candidates_passes_centers_to_distance_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    centers = {"candidate": (0.0, 0.0)}
    received: dict[str, object] = {}

    def capture_distance_filter(
        candidates: list[str],
        selected: list[str],
        filter_centers: dict[str, tuple[float, float]],
        minimum_distance: float,
        nearest_distances: dict[str, float] | None,
    ) -> list[str]:
        received["centers"] = filter_centers
        return candidates

    monkeypatch.setattr(stratification, "_distant_candidates", capture_distance_filter)

    assert _required_distant_candidates(["candidate"], [], centers, 1.0, None) == ["candidate"]
    assert received["centers"] is centers


def test_distant_candidates_use_centers_when_the_cache_is_disabled() -> None:
    centers = {"selected": (0.0, 0.0), "candidate": (0.0, 0.001)}

    assert (
        _distant_candidates(
            ["candidate"],
            ["selected"],
            centers,
            1.0,
            None,
        )
        == []
    )


def test_next_source_cell_uses_seed_for_cached_ties() -> None:
    assert (
        _next_source_cell(
            ["a", "b"],
            ["selected"],
            {"a": (0.0, 0.0), "b": (0.0, 1.0)},
            "seed",
            {"a": 1.0, "b": 1.0},
        )
        == "b"
    )


def test_cached_distant_cells_keep_a_candidate_at_the_boundary() -> None:
    assert _cached_distant_cells(["candidate"], {"candidate": 100.0}, 100.0) == ["candidate"]


def test_farthest_cached_cell_prioritizes_distance() -> None:
    assert _farthest_cached_cell(["a", "z"], {"a": 100.0, "z": 1.0}, "seed") == "a"


def test_farthest_cached_cell_uses_seed_for_ties() -> None:
    assert _farthest_cached_cell(["a", "b"], {"a": 1.0, "b": 1.0}, "seed") == "b"


def test_farthest_cached_cell_uses_each_cell_in_the_tiebreaker() -> None:
    assert _farthest_cached_cell(["b", "a"], {"a": 1.0, "b": 1.0}, "seed") == "b"


def test_update_selected_distance_cache_marks_the_selected_cell_as_zero() -> None:
    nearest_distances = {"candidate": math.inf}
    centers = {"selected": (0.0, 0.0), "candidate": (0.0, 1.0)}

    _update_selected_distance_cache(centers, nearest_distances, [], "selected")

    assert nearest_distances["selected"] == 0.0


def test_feasible_source_cells_reserves_cells_for_the_other_source() -> None:
    available = {
        Source.WIKIPEDIA: {"wiki-a", "wiki-b", "shared"},
        Source.WEBSITE: {"shared"},
    }

    assert _feasible_source_cells(
        Source.WIKIPEDIA,
        available,
        {source: [] for source in DEFAULT_SOURCES},
        [],
        target_count=1,
    ) == ["wiki-a", "wiki-b"]


def test_feasible_source_cells_rejects_an_insufficient_current_source() -> None:
    available = {Source.WIKIPEDIA: {"wiki-a"}, Source.WEBSITE: {"web-a", "web-b"}}

    assert (
        _feasible_source_cells(
            Source.WIKIPEDIA,
            available,
            {source: [] for source in DEFAULT_SOURCES},
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
            {source: [] for source in DEFAULT_SOURCES},
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


V3_SOURCE_TRIPLE = (Source.WIKIPEDIA, Source.WEBSITE, Source.DESCRIPTION)


def test_distinct_source_cell_selection_serves_a_three_source_profile() -> None:
    eligible = {
        Source.WIKIPEDIA: {"a", "b", "c", "d"},
        Source.WEBSITE: {"a", "b", "c", "d"},
        Source.DESCRIPTION: {"a", "b", "c", "d"},
    }

    selected = select_distinct_source_cells(
        eligible,
        target_count_per_source=1,
        center_of_cell=lambda cell: (0.0, float(ord(cell[0]))),
        seed="seed",
        sources=V3_SOURCE_TRIPLE,
    )

    assert set(selected) == set(V3_SOURCE_TRIPLE)
    chosen = [cell for cells in selected.values() for cell in cells]
    assert len(chosen) == 3
    assert len(set(chosen)) == 3


def test_distinct_source_cell_selection_reserves_cells_for_every_declared_source() -> None:
    eligible = {
        Source.WIKIPEDIA: {"a", "b", "c"},
        Source.WEBSITE: {"a"},
        Source.DESCRIPTION: {"b"},
    }

    selected = select_distinct_source_cells(
        eligible,
        target_count_per_source=1,
        center_of_cell=lambda cell: (0.0, float(ord(cell[0]))),
        seed="seed",
        sources=V3_SOURCE_TRIPLE,
    )

    assert selected[Source.WEBSITE] == ("a",)
    assert selected[Source.DESCRIPTION] == ("b",)
    assert selected[Source.WIKIPEDIA] == ("c",)


def test_distinct_source_cell_selection_ignores_sources_it_was_not_given() -> None:
    eligible = {
        Source.WIKIPEDIA: {"a", "b"},
        Source.WEBSITE: {"a", "b"},
        Source.DESCRIPTION: {"a", "b"},
    }

    selected = select_distinct_source_cells(
        eligible,
        target_count_per_source=1,
        center_of_cell=lambda cell: (0.0, float(ord(cell[0]))),
        seed="seed",
        sources=(Source.DESCRIPTION,),
    )

    assert set(selected) == {Source.DESCRIPTION}
    assert len(selected[Source.DESCRIPTION]) == 1


def test_source_cell_selection_does_not_rescan_the_whole_selection_each_round() -> None:
    """Distance work must grow with the selection, not with its square.

    The farthest-point choice needs each cell's distance to its nearest selected
    cell. Recomputing that against every already-selected cell each round turns a
    400-cell pool into hundreds of millions of haversine calls; keeping a nearest
    distance per cell and folding in only the newest selection gives the same
    answer for a fraction of the work.
    """

    from landuse_sentence_relevance.domain import stratification

    calls = {"n": 0}
    real = stratification._haversine_km

    def counting(first: tuple[float, float], second: tuple[float, float]) -> float:
        calls["n"] += 1
        return real(first, second)

    cells = {
        source: tuple(f"{source.value}-{index:04d}" for index in range(120)) for source in DEFAULT_SOURCES
    }
    centers = {
        cell: (float(index % 90), float(index % 180))
        for source_cells in cells.values()
        for index, cell in enumerate(source_cells)
    }
    target = 40

    monkeypatch_attr(stratification, "_haversine_km", counting)
    try:
        selected = stratification.select_distinct_source_cells(
            cells,
            target_count_per_source=target,
            center_of_cell=centers.__getitem__,
            seed="seed",
        )
    finally:
        monkeypatch_attr(stratification, "_haversine_km", real)

    assert all(len(chosen) == target for chosen in selected.values())
    total_cells = len(centers)
    rounds = target * len(DEFAULT_SOURCES)
    naive = total_cells * rounds * rounds
    assert calls["n"] < naive // 10, f"{calls['n']} haversine calls suggests a per-round rescan"


def monkeypatch_attr(module: object, name: str, value: object) -> None:
    """Swap a module attribute without tripping the type checker on a function slot."""

    setattr(module, name, value)


def _brute_force_feasible(
    source: Source,
    available: Mapping[Source, set[str]],
    selected_by_source: Mapping[Source, list[str]],
    selected_cells: list[str],
    target_count: int,
    sources: tuple[Source, ...],
) -> list[str]:
    """The feasibility rule stated directly, as the reference to optimise against."""

    candidates = sorted(available[source] - set(selected_cells))
    feasible: list[str] = []
    for candidate in candidates:
        used = set(selected_cells) | {candidate}
        if all(
            len(available[other] - used)
            >= target_count - len(selected_by_source[other]) - (1 if other is source else 0)
            for other in sources
        ):
            feasible.append(candidate)
    return feasible


def test_feasible_source_cells_matches_the_rule_it_optimises() -> None:
    """Reducing the per-candidate set work must not change which cells are feasible."""

    import random

    rng = random.Random(20260918)
    for _ in range(40):
        pool = [f"c{index:03d}" for index in range(30)]
        available = {source: set(rng.sample(pool, rng.randint(8, 25))) for source in DEFAULT_SOURCES}
        selected_cells = rng.sample(pool, rng.randint(0, 6))
        selected_by_source = {
            source: rng.sample(sorted(available[source]), rng.randint(0, 3)) for source in DEFAULT_SOURCES
        }
        target = rng.randint(1, 8)
        for source in DEFAULT_SOURCES:
            assert stratification._feasible_source_cells(
                source, available, selected_by_source, selected_cells, target, DEFAULT_SOURCES
            ) == _brute_force_feasible(
                source, available, selected_by_source, selected_cells, target, DEFAULT_SOURCES
            )


def test_feasible_source_cells_does_not_rebuild_the_selection_per_candidate() -> None:
    """Per-candidate set differences make selection quadratic in the pool size."""

    class CountingSet(set):  # type: ignore[type-arg]
        differences = 0

        def __sub__(self, other):  # type: ignore[no-untyped-def]
            type(self).differences += 1
            return CountingSet(set(self) - set(other))

    available = {source: CountingSet(f"c{index:04d}" for index in range(400)) for source in DEFAULT_SOURCES}
    selected_cells = [f"c{index:04d}" for index in range(50)]
    selected_by_source = {source: [] for source in DEFAULT_SOURCES}

    CountingSet.differences = 0
    stratification._feasible_source_cells(
        Source.WIKIPEDIA, available, selected_by_source, selected_cells, 100, DEFAULT_SOURCES
    )

    assert CountingSet.differences <= len(DEFAULT_SOURCES) + 1, (
        f"{CountingSet.differences} set differences for one round scales with the candidate count"
    )
