from __future__ import annotations

import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from scripts import prepare_v3_osm_basemap as basemap
from scripts.prepare_v3_osm_basemap import (
    prepare_basemap,
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


def test_prepare_basemap_rebuilds_from_local_snapshot_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles: dict[tuple[int, int], Image.Image] = {}
    entries: list[dict[str, int | str]] = []
    for x, y in tile_coordinates(2):
        tile_path = tiles_dir / str(x) / f"{y}.png"
        tile_path.parent.mkdir(parents=True, exist_ok=True)
        tile = solid_tile((x * 40, y * 40, 100, 255)).resize((256, 256))
        tile.save(tile_path)
        tiles[(x, y)] = Image.open(BytesIO(tile_path.read_bytes())).convert("RGBA")
        entries.append({"x": x, "y": y, "sha256": sha256(tile_path.read_bytes()).hexdigest()})

    expected = stitch_tiles(tiles, columns=4, rows=4, tile_size=256)
    expected_path = tmp_path / "expected.png"
    expected.save(expected_path, format="PNG", optimize=False, compress_level=9)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_url": basemap.TILE_URL,
                "zoom": 2,
                "tile_size": 256,
                "retrieved_at": "2026-09-20",
                "attribution": "© OpenStreetMap contributors",
                "license": "Open Database License (ODbL) 1.0",
                "tiles": entries,
                "mosaic_sha256": sha256(expected_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    def reject_network(_: str) -> bytes:
        raise AssertionError("network access")

    monkeypatch.setattr(basemap, "_download_tile", reject_network)

    output_path = tmp_path / "output.png"
    prepare_basemap(manifest_path, output_path, tiles_dir=tiles_dir)

    assert output_path.read_bytes() == expected_path.read_bytes()


def test_prepare_basemap_requires_recording_for_network_refresh(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --record-checksums"):
        prepare_basemap(
            tmp_path / "manifest.json",
            tmp_path / "output.png",
            tiles_dir=tmp_path / "tiles",
            download=True,
        )
