from __future__ import annotations

import json
from pathlib import Path

import pytest

from landuse_sentence_relevance.reporting.geographic import (
    load_benchmark_points,
    load_land_basemap,
    render_world_map,
)

ROOT = Path(__file__).parents[3]
FIXTURE_BENCHMARK = ROOT / "tests/fixtures/geographic_map/benchmark.csv"
SMALL_SOURCE_COUNTS = {"description": 1, "website": 1, "wikipedia": 1}

LAND_DOCUMENT = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 0]]]},
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [[[[-20, -20], [-10, -20], [-10, -10], [-20, -20]]]],
            },
        },
    ],
}


def write_land(path: Path, document: object = LAND_DOCUMENT) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


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


def test_load_land_basemap_returns_polygon_and_multipolygon_rings(tmp_path: Path) -> None:
    rings = load_land_basemap(write_land(tmp_path / "land.geojson"))

    assert rings == (
        ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 0.0)),
        ((-20.0, -20.0), (-10.0, -20.0), (-10.0, -10.0), (-20.0, -20.0)),
    )


def test_load_land_basemap_rejects_a_non_collection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="FeatureCollection"):
        load_land_basemap(write_land(tmp_path / "land.geojson", {"type": "Feature"}))


def test_load_land_basemap_rejects_a_checksum_mismatch(tmp_path: Path) -> None:
    path = write_land(tmp_path / "land.geojson")

    with pytest.raises(ValueError, match="checksum"):
        load_land_basemap(path, expected_sha256="0" * 64)


def test_committed_land_basemap_matches_its_manifest_checksum() -> None:
    assets = ROOT / "data/benchmark/v3/assets"
    manifest = json.loads((assets / "ne-110m-land-manifest.json").read_text(encoding="utf-8"))

    rings = load_land_basemap(assets / "ne-110m-land.geojson", expected_sha256=manifest["sha256"])

    assert len(rings) >= 100


def test_render_map_is_byte_identical_for_identical_inputs(tmp_path: Path) -> None:
    points = load_benchmark_points(FIXTURE_BENCHMARK, expected_counts=SMALL_SOURCE_COUNTS)
    land = load_land_basemap(write_land(tmp_path / "land.geojson"))
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    render_world_map(points, land, first)
    render_world_map(points, land, second)

    assert first.read_bytes() == second.read_bytes()
