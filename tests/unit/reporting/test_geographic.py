from __future__ import annotations

from pathlib import Path

import pytest

from landuse_sentence_relevance.reporting.geographic import (
    load_benchmark_points,
    load_boundary_rings,
    render_world_map,
)

ROOT = Path(__file__).parents[3]
FIXTURE_BENCHMARK = ROOT / "tests/fixtures/geographic_map/benchmark.csv"
FIXTURE_BOUNDARIES = ROOT / "tests/fixtures/geographic_map/boundaries.geojson"
SMALL_SOURCE_COUNTS = {"description": 1, "website": 1, "wikipedia": 1}


def test_load_points_preserves_rows_and_validates_small_contract() -> None:
    points = load_benchmark_points(FIXTURE_BENCHMARK, expected_counts=SMALL_SOURCE_COUNTS)

    assert [(point.latitude, point.longitude, point.source) for point in points] == [
        (10.0, 20.0, "description"),
        (-10.0, -20.0, "website"),
        (30.0, 40.0, "wikipedia"),
    ]


def test_load_points_rejects_wrong_source_counts() -> None:
    with pytest.raises(ValueError, match="source counts"):
        load_benchmark_points(FIXTURE_BENCHMARK)


def test_load_points_rejects_non_finite_coordinates(tmp_path: Path) -> None:
    malformed = tmp_path / "benchmark.csv"
    malformed.write_text(
        FIXTURE_BENCHMARK.read_text(encoding="utf-8").replace(",10,20,description,", ",nan,20,description,"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="latitude"):
        load_benchmark_points(malformed, expected_counts=SMALL_SOURCE_COUNTS)


def test_load_boundary_rings_reads_polygon_and_multipolygon_exteriors() -> None:
    rings = load_boundary_rings(FIXTURE_BOUNDARIES)

    assert len(rings) == 2
    assert all(len(ring) == 5 for ring in rings)
    assert rings[0][0] == (0.0, 0.0)
    assert rings[1][0] == (2.0, 2.0)


def test_render_map_is_byte_identical_for_identical_inputs(tmp_path: Path) -> None:
    points = load_benchmark_points(FIXTURE_BENCHMARK, expected_counts=SMALL_SOURCE_COUNTS)
    rings = load_boundary_rings(FIXTURE_BOUNDARIES)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    render_world_map(points, rings, first)
    render_world_map(points, rings, second)

    assert first.read_bytes() == second.read_bytes()
