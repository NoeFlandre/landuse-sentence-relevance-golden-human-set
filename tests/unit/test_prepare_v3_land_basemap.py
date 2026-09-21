from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.prepare_v3_land_basemap import SOURCE_REF, SOURCE_URL, main, prepare_basemap

LAND_DOCUMENT = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        }
    ],
}


def write_land(path: Path, document: object = LAND_DOCUMENT) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_record_checksums_writes_a_pinned_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    output = write_land(tmp_path / "land.geojson")

    prepare_basemap(manifest_path, output, record_checksums=True, retrieved_at="2026-09-21")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_url"] == SOURCE_URL
    assert manifest["source_ref"] == SOURCE_REF
    assert manifest["retrieved_at"] == "2026-09-21"
    assert len(manifest["sha256"]) == 64


def test_verification_accepts_the_recorded_snapshot(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    output = write_land(tmp_path / "land.geojson")
    prepare_basemap(manifest_path, output, record_checksums=True, retrieved_at="2026-09-21")

    assert main(["--manifest", str(manifest_path), "--output", str(output)]) == 0


def test_verification_rejects_modified_bytes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    output = write_land(tmp_path / "land.geojson")
    prepare_basemap(manifest_path, output, record_checksums=True, retrieved_at="2026-09-21")
    write_land(output, {"type": "FeatureCollection", "features": LAND_DOCUMENT["features"] * 2})

    with pytest.raises(ValueError, match="checksum"):
        prepare_basemap(manifest_path, output)


def test_download_requires_recording_checksums(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--record-checksums"):
        prepare_basemap(tmp_path / "manifest.json", tmp_path / "land.geojson", download=True)


def test_record_checksums_requires_a_retrieval_date(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--retrieved-at"):
        prepare_basemap(
            tmp_path / "manifest.json",
            write_land(tmp_path / "land.geojson"),
            record_checksums=True,
        )


def test_record_checksums_rejects_a_malformed_retrieval_date(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        prepare_basemap(
            tmp_path / "manifest.json",
            write_land(tmp_path / "land.geojson"),
            record_checksums=True,
            retrieved_at="21-09-2026",
        )


def test_verification_rejects_a_manifest_off_the_pinned_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    output = write_land(tmp_path / "land.geojson")
    prepare_basemap(manifest_path, output, record_checksums=True, retrieved_at="2026-09-21")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_ref"] = "v0.0.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="pinned Natural Earth land contract"):
        prepare_basemap(manifest_path, output)


def test_verification_reports_a_missing_snapshot(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    output = write_land(tmp_path / "land.geojson")
    prepare_basemap(manifest_path, output, record_checksums=True, retrieved_at="2026-09-21")
    output.unlink()

    with pytest.raises(ValueError, match="missing vendored basemap"):
        prepare_basemap(manifest_path, output)


def test_verification_rejects_a_non_collection_snapshot(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    output = write_land(tmp_path / "land.geojson")
    prepare_basemap(manifest_path, output, record_checksums=True, retrieved_at="2026-09-21")
    write_land(output, {"type": "Feature"})

    with pytest.raises(ValueError, match="FeatureCollection"):
        prepare_basemap(manifest_path, output)
