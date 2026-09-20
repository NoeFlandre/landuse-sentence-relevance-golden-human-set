from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from landuse_sentence_relevance.reporting.geographic import (
    load_benchmark_points,
    load_osm_basemap,
    render_world_map,
    web_mercator_y,
)

ROOT = Path(__file__).parents[3]
FIXTURE_BENCHMARK = ROOT / "tests/fixtures/geographic_map/benchmark.csv"
SMALL_SOURCE_COUNTS = {"description": 1, "website": 1, "wikipedia": 1}


def write_basemap(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    Image.new("RGBA", size, (190, 215, 230, 255)).save(path)


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


def test_load_osm_basemap_validates_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "basemap.png"
    write_basemap(path)

    image = load_osm_basemap(path, expected_size=(8, 8))

    assert image.shape == (8, 8, 4)


def test_load_osm_basemap_rejects_wrong_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "basemap.png"
    write_basemap(path, size=(8, 4))

    with pytest.raises(ValueError, match="1024"):
        load_osm_basemap(path)


def test_web_mercator_y_is_symmetric_and_clips_poles() -> None:
    assert web_mercator_y(0.0) == pytest.approx(0.0)
    assert web_mercator_y(60.0) == pytest.approx(-web_mercator_y(-60.0))
    assert web_mercator_y(90.0) == pytest.approx(web_mercator_y(85.05112878))


def test_render_map_is_byte_identical_for_identical_inputs(tmp_path: Path) -> None:
    points = load_benchmark_points(FIXTURE_BENCHMARK, expected_counts=SMALL_SOURCE_COUNTS)
    basemap_path = tmp_path / "basemap.png"
    write_basemap(basemap_path)
    basemap = load_osm_basemap(basemap_path, expected_size=(8, 8))
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    render_world_map(points, basemap, first)
    render_world_map(points, basemap, second)

    assert first.read_bytes() == second.read_bytes()
