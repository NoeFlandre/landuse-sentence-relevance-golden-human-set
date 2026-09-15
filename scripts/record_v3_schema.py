"""Re-record the pinned upstream Parquet schemas the V3 projections are checked against.

Run this only when a V3 revision pin changes:

    uv run python scripts/record_v3_schema.py > tests/fixtures/v3_upstream_schema.json

It reads Parquet footers and one recorded shard over the network; it never
downloads or materializes a dataset.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from landuse_sentence_relevance.bootstrap import v3_stream_specs
from landuse_sentence_relevance.config import V3Settings

_RECORDED_SHARD = "afghanistan-latest.parquet"


def _filesystem() -> Any:
    import fsspec

    return fsspec.filesystem("http")


def _parquet_files(dataset_id: str, revision: str, directory: str) -> tuple[str, ...]:
    from huggingface_hub import HfApi

    entries = HfApi().list_repo_tree(
        dataset_id, revision=revision, repo_type="dataset", path_in_repo=directory
    )
    return tuple(
        sorted(
            entry.path
            for entry in entries
            if isinstance(getattr(entry, "path", None), str) and entry.path.endswith(".parquet")
        )
    )


def _url(dataset_id: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{path}"


def _columns(url: str) -> dict[str, str]:
    import pyarrow.parquet as pq

    with _filesystem().open(url) as handle:
        schema = pq.ParquetFile(handle).schema_arrow
    return {field.name: str(field.type) for field in schema}


def _read(url: str, columns: list[str]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    with _filesystem().open(url) as handle:
        return pq.ParquetFile(handle).read(columns=columns).to_pylist()


def _wikipedia_samples(dataset_id: str, revision: str) -> dict[str, Any]:
    path = f"wikipedia/sentences/{_RECORDED_SHARD}"
    rows = _read(
        _url(dataset_id, revision, path),
        [
            "project",
            "language",
            "section_index",
            "heading",
            "level",
            "section_path",
            "sentence_index",
            "text",
        ],
    )
    english = [row for row in rows if row["language"] == "en" and row["project"] == "wikipedia"]
    lead = [row for row in english if row["section_index"] == 0]
    body = [row for row in english if row["section_index"] > 0]
    first = [row for row in body if row["sentence_index"] == 0]
    return {
        "recorded_from": path,
        "english_rows": len(english),
        "lead_rows": len(lead),
        "body_rows": len(body),
        "lead_headings_are_all_empty": all(row["heading"] == "" for row in lead),
        "lead_levels_are_all_zero": all(row["level"] == 0 for row in lead),
        "lead_section_paths": sorted({row["section_path"] for row in lead}),
        "body_rows_with_an_empty_heading": sum(1 for row in body if row["heading"] == ""),
        "body_first_sentences": len(first),
        "body_first_sentences_starting_with_the_edit_marker": sum(
            1 for row in first if row["text"].strip().startswith("[")
        ),
        "edit_markers_outside_a_first_sentence": sum(
            1 for row in english if row["sentence_index"] > 0 and row["text"].strip().startswith("[ edit")
        ),
    }


def record(settings: V3Settings) -> dict[str, Any]:
    """Return the recorded schema document for the pinned V3 revisions."""

    streams: dict[str, Any] = {}
    for spec in v3_stream_specs(settings):
        files = _parquet_files(spec.dataset_id, spec.revision, spec.directory)
        streams[spec.name] = {
            "dataset_id": spec.dataset_id,
            "revision": spec.revision,
            "config": spec.config,
            "split": spec.split,
            "directory": spec.directory,
            "parquet_file_count": len(files),
            "recorded_from": files[0],
            "columns": _columns(_url(spec.dataset_id, spec.revision, files[0])),
        }
        print(f"recorded {spec.name}: {len(streams[spec.name]['columns'])} columns", file=sys.stderr)

    samples: dict[str, Any] = {
        "wikipedia_sentences": _wikipedia_samples(
            settings.wikipedia_dataset_id, settings.wikipedia_dataset_revision
        )
    }
    for name in ("wikipedia_polygons", "website_polygons"):
        stream = streams[name]
        rows = _read(
            _url(stream["dataset_id"], stream["revision"], f"{stream['directory']}/{_RECORDED_SHARD}"),
            ["region", "source_pbf"],
        )
        samples[name] = {
            "recorded_from": f"{stream['directory']}/{_RECORDED_SHARD}",
            "regions": sorted({row["region"] for row in rows})[:3],
            "source_pbfs": sorted({row["source_pbf"] for row in rows})[:3],
        }
    geometry = streams["description_geometry"]
    rows = _read(
        _url(geometry["dataset_id"], geometry["revision"], f"{geometry['directory']}/{_RECORDED_SHARD}"),
        ["source_pbf", "osm_url"],
    )
    samples["description_geometry"] = {
        "recorded_from": f"{geometry['directory']}/{_RECORDED_SHARD}",
        "source_pbfs": sorted({row["source_pbf"] for row in rows})[:3],
        "osm_url_example": rows[0]["osm_url"],
    }
    return {
        "description": (
            "Recorded Parquet schemas of the pinned V3 upstream revisions. The V3 column "
            "projections are asserted against this file so a projection can never name a "
            "column the pinned revision does not publish. Re-record with "
            "`uv run python scripts/record_v3_schema.py` when a pin changes."
        ),
        "streams": streams,
        "samples": samples,
    }


def main() -> int:
    print(json.dumps(record(V3Settings.from_env()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
