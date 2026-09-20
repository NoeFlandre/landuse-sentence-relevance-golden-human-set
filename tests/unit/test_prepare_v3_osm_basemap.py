from __future__ import annotations

import pytest
from PIL import Image
from scripts.prepare_v3_osm_basemap import (
    stitch_tiles,
    tile_coordinates,
    verify_tile_hash,
)


def solid_tile(color: tuple[int, int, int, int]) -> Image.Image:
    return Image.new("RGBA", (2, 2), color)


def test_tile_coordinates_for_zoom_two_are_four_by_four() -> None:
    assert list(tile_coordinates(2)) == [(x, y) for y in range(4) for x in range(4)]


def test_stitch_tiles_places_tiles_in_row_major_order() -> None:
    tiles = {
        (0, 0): solid_tile((255, 0, 0, 255)),
        (1, 0): solid_tile((0, 255, 0, 255)),
        (0, 1): solid_tile((0, 0, 255, 255)),
        (1, 1): solid_tile((255, 255, 0, 255)),
    }

    mosaic = stitch_tiles(tiles, columns=2, rows=2, tile_size=2)

    assert mosaic.getpixel((0, 0)) == (255, 0, 0, 255)
    assert mosaic.getpixel((2, 0)) == (0, 255, 0, 255)
    assert mosaic.getpixel((0, 2)) == (0, 0, 255, 255)
    assert mosaic.getpixel((2, 2)) == (255, 255, 0, 255)


def test_verify_tile_hash_rejects_changed_bytes() -> None:
    with pytest.raises(ValueError, match="checksum"):
        verify_tile_hash(b"changed", "0" * 64, "2/0/0")
