"""Prepare and verify the pinned, offline Natural Earth land basemap."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

SOURCE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/{ref}/geojson/ne_110m_land.geojson"
)
SOURCE_REF = "v5.1.2"
USER_AGENT = "landuse-sentence-relevance-golden-human-set/2.0 (Natural Earth basemap maintenance)"


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return manifest


def _manifest_for_recording(retrieved_at: str | None) -> dict[str, Any]:
    if retrieved_at is None:
        raise ValueError("--retrieved-at is required with --record-checksums")
    try:
        date.fromisoformat(retrieved_at)
    except ValueError as error:
        raise ValueError("--retrieved-at must be YYYY-MM-DD") from error
    return {
        "source_url": SOURCE_URL,
        "source_ref": SOURCE_REF,
        "dataset": "Natural Earth 1:110m physical land",
        "retrieved_at": retrieved_at,
        "attribution": "Made with Natural Earth",
        "license": "Public domain",
    }


def _validate_geojson(data: bytes, path: Path) -> None:
    document = json.loads(data.decode("utf-8"))
    if not isinstance(document, dict) or document.get("type") != "FeatureCollection":
        raise ValueError(f"{path} must contain a GeoJSON FeatureCollection")
    features = document.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"{path} must contain at least one land feature")


def prepare_basemap(
    manifest_path: Path,
    output_path: Path,
    *,
    download: bool = False,
    record_checksums: bool = False,
    retrieved_at: str | None = None,
) -> None:
    """Verify or refresh the pinned Natural Earth land snapshot."""

    if download and not record_checksums:
        raise ValueError("--download requires --record-checksums to create a new snapshot")

    manifest = _manifest_for_recording(retrieved_at) if record_checksums else _read_manifest(manifest_path)
    if manifest.get("source_url") != SOURCE_URL or manifest.get("source_ref") != SOURCE_REF:
        raise ValueError("manifest does not match the pinned Natural Earth land contract")

    if download:
        data = _download(SOURCE_URL.format(ref=SOURCE_REF))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
    else:
        try:
            data = output_path.read_bytes()
        except FileNotFoundError as error:
            message = f"missing vendored basemap {output_path}; use --download to create a snapshot"
            raise ValueError(message) from error

    _validate_geojson(data, output_path)
    checksum = sha256(data).hexdigest()
    if record_checksums:
        manifest["sha256"] = checksum
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    if checksum != manifest.get("sha256"):
        raise ValueError(f"basemap checksum {checksum} does not match manifest")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Fetch and save a new snapshot")
    parser.add_argument("--record-checksums", action="store_true")
    parser.add_argument("--retrieved-at")
    args = parser.parse_args(argv)
    prepare_basemap(
        args.manifest,
        args.output,
        download=args.download,
        record_checksums=args.record_checksums,
        retrieved_at=args.retrieved_at,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
