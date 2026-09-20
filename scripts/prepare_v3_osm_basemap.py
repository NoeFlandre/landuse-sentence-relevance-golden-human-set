"""Prepare a pinned, offline OpenStreetMap tile mosaic for the V3 map."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import date
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from PIL import Image

TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
ZOOM = 2
TILE_SIZE = 256
USER_AGENT = "landuse-sentence-relevance-golden-human-set/2.0 (OSM basemap maintenance)"


def tile_coordinates(zoom: int) -> tuple[tuple[int, int], ...]:
    """Return all tile coordinates in deterministic row-major order."""

    width = 2**zoom
    return tuple((x, y) for y in range(width) for x in range(width))


def verify_tile_hash(data: bytes, expected: str, tile_name: str) -> None:
    """Reject tile bytes that differ from the pinned manifest checksum."""

    actual = sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"{tile_name} checksum {actual} does not match {expected}")


def stitch_tiles(
    tiles: Mapping[tuple[int, int], Image.Image],
    *,
    columns: int,
    rows: int,
    tile_size: int,
) -> Image.Image:
    """Stitch same-sized RGBA tiles into a deterministic mosaic."""

    expected_coordinates = {(x, y) for y in range(rows) for x in range(columns)}
    if set(tiles) != expected_coordinates:
        raise ValueError("tile set does not match the requested mosaic dimensions")
    mosaic = Image.new("RGBA", (columns * tile_size, rows * tile_size))
    for (x, y), tile in sorted(tiles.items(), key=lambda item: (item[0][1], item[0][0])):
        if tile.size != (tile_size, tile_size):
            raise ValueError(f"tile {(x, y)} has size {tile.size}, expected {(tile_size, tile_size)}")
        mosaic.paste(tile.convert("RGBA"), (x * tile_size, y * tile_size))
    return mosaic


def _download_tile(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _manifest_for_recording(retrieved_at: str | None) -> dict[str, Any]:
    if retrieved_at is None:
        raise ValueError("--retrieved-at is required with --record-checksums")
    try:
        date.fromisoformat(retrieved_at)
    except ValueError as error:
        raise ValueError("--retrieved-at must be YYYY-MM-DD") from error
    return {
        "source_url": TILE_URL,
        "zoom": ZOOM,
        "tile_size": TILE_SIZE,
        "retrieved_at": retrieved_at,
        "attribution": "© OpenStreetMap contributors",
        "license": "Open Database License (ODbL) 1.0",
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return manifest


def _tile_hashes(
    manifest: Mapping[str, Any], coordinates: Sequence[tuple[int, int]]
) -> dict[tuple[int, int], str]:
    entries = manifest.get("tiles")
    if not isinstance(entries, list):
        raise ValueError("manifest tiles must be a list")
    hashes: dict[tuple[int, int], str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("manifest tile entry must be an object")
        coordinate = (entry.get("x"), entry.get("y"))
        checksum = entry.get("sha256")
        if coordinate in hashes or coordinate not in coordinates or not isinstance(checksum, str):
            raise ValueError("manifest contains an invalid or duplicate tile entry")
        hashes[coordinate] = checksum
    if set(hashes) != set(coordinates):
        raise ValueError("manifest does not contain exactly the expected tiles")
    return hashes


def _manifest_entries(hashes: Mapping[tuple[int, int], str]) -> list[dict[str, Any]]:
    return [
        {"x": x, "y": y, "sha256": hashes[(x, y)]}
        for x, y in sorted(hashes, key=lambda item: (item[1], item[0]))
    ]


def prepare_basemap(
    manifest_path: Path,
    output_path: Path,
    *,
    record_checksums: bool = False,
    retrieved_at: str | None = None,
) -> None:
    """Download, verify, stitch, and persist the pinned OSM tile mosaic."""

    manifest = _manifest_for_recording(retrieved_at) if record_checksums else _read_manifest(manifest_path)
    zoom = manifest.get("zoom")
    tile_size = manifest.get("tile_size")
    if zoom != ZOOM or tile_size != TILE_SIZE or manifest.get("source_url") != TILE_URL:
        raise ValueError("manifest does not match the pinned OSM z2 tile contract")

    coordinates = tile_coordinates(ZOOM)
    expected_hashes = {} if record_checksums else _tile_hashes(manifest, coordinates)
    tiles: dict[tuple[int, int], Image.Image] = {}
    actual_hashes: dict[tuple[int, int], str] = {}
    for x, y in coordinates:
        tile_name = f"{ZOOM}/{x}/{y}"
        data = _download_tile(TILE_URL.format(z=ZOOM, x=x, y=y))
        actual_hashes[(x, y)] = sha256(data).hexdigest()
        if not record_checksums:
            verify_tile_hash(data, expected_hashes[(x, y)], tile_name)
        with Image.open(BytesIO(data)) as tile:
            tiles[(x, y)] = tile.convert("RGBA")

    mosaic = stitch_tiles(tiles, columns=2**ZOOM, rows=2**ZOOM, tile_size=TILE_SIZE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(output_path, format="PNG", optimize=False, compress_level=9)
    mosaic_hash = sha256(output_path.read_bytes()).hexdigest()
    if not record_checksums and mosaic_hash != manifest.get("mosaic_sha256"):
        raise ValueError(f"mosaic checksum {mosaic_hash} does not match manifest")
    if record_checksums:
        manifest["tiles"] = _manifest_entries(actual_hashes)
        manifest["mosaic_sha256"] = mosaic_hash
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-checksums", action="store_true")
    parser.add_argument("--retrieved-at")
    args = parser.parse_args(argv)
    prepare_basemap(
        args.manifest,
        args.output,
        record_checksums=args.record_checksums,
        retrieved_at=args.retrieved_at,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
