from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from landuse_sentence_relevance.bootstrap import v3_stream_specs
from landuse_sentence_relevance.config import V3Settings

RECORDED_SCHEMA_PATH = Path(__file__).parents[1] / "fixtures/v3_upstream_schema.json"


@pytest.fixture(scope="module")
def recorded_schema() -> dict[str, Any]:
    return json.loads(RECORDED_SCHEMA_PATH.read_text(encoding="utf-8"))


def test_the_recorded_schema_covers_exactly_the_five_v3_streams(recorded_schema: dict[str, Any]) -> None:
    assert set(recorded_schema["streams"]) == {
        "description_sentences",
        "description_geometry",
        "wikipedia_sentences",
        "wikipedia_polygons",
        "website_polygons",
    }
    assert {spec.name for spec in v3_stream_specs(V3Settings())} == set(recorded_schema["streams"])


def test_the_recorded_schema_was_taken_from_the_pinned_revisions(recorded_schema: dict[str, Any]) -> None:
    settings = V3Settings()
    revisions = {name: stream["revision"] for name, stream in recorded_schema["streams"].items()}

    assert revisions == {
        "description_sentences": settings.description_dataset_revision,
        "description_geometry": settings.description_dataset_revision,
        "wikipedia_sentences": settings.wikipedia_dataset_revision,
        "wikipedia_polygons": settings.wikipedia_dataset_revision,
        "website_polygons": settings.website_dataset_revision,
    }
    assert all(stream["columns"] for stream in recorded_schema["streams"].values())


def test_every_projected_column_exists_in_the_recorded_upstream_schema(
    recorded_schema: dict[str, Any],
) -> None:
    missing: dict[str, tuple[str, ...]] = {}
    for spec in v3_stream_specs(V3Settings()):
        published = set(recorded_schema["streams"][spec.name]["columns"])
        absent = tuple(column for column in spec.columns if column not in published)
        if absent:
            missing[spec.name] = absent

    assert missing == {}


def test_every_stream_spec_matches_the_recorded_dataset_coordinates(
    recorded_schema: dict[str, Any],
) -> None:
    for spec in v3_stream_specs(V3Settings()):
        recorded = recorded_schema["streams"][spec.name]
        assert (spec.dataset_id, spec.revision, spec.config, spec.split, spec.directory) == (
            recorded["dataset_id"],
            recorded["revision"],
            recorded["config"],
            recorded["split"],
            recorded["directory"],
        )


def test_the_projections_still_carry_the_fields_the_adapters_read(
    recorded_schema: dict[str, Any],
) -> None:
    columns = {spec.name: set(spec.columns) for spec in v3_stream_specs(V3Settings())}

    assert {"language_code", "top_score", "sentences", "source_pbf"} <= columns["description_sentences"]
    assert {"osm_url", "name", "bbox_min_x", "bbox_max_y", "source_pbf"} <= columns["description_geometry"]
    assert {"project", "language", "section_index", "heading", "sentence_index", "page_id", "text"} <= (
        columns["wikipedia_sentences"]
    )
    assert {"region", "name", "lat", "lon", "has_english_wikipedia"} <= columns["wikipedia_polygons"]
    assert {"region", "website", "website_sentences", "website_language"} <= columns["website_polygons"]


def test_the_wikipedia_sentences_stream_publishes_no_lead_or_title_flag(
    recorded_schema: dict[str, Any],
) -> None:
    published = set(recorded_schema["streams"]["wikipedia_sentences"]["columns"])

    assert {"is_lead", "is_title", "source_url"} & published == set()
    assert {"section_index", "heading", "level", "section_path", "page_id"} <= published


def test_the_description_geometry_stream_publishes_no_region_column(
    recorded_schema: dict[str, Any],
) -> None:
    published = set(recorded_schema["streams"]["description_geometry"]["columns"])

    assert "region" not in published
    assert "source_pbf" in published


def test_the_recorded_samples_justify_the_contextual_wikipedia_rule(
    recorded_schema: dict[str, Any],
) -> None:
    sample = recorded_schema["samples"]["wikipedia_sentences"]

    assert sample["lead_headings_are_all_empty"] is True
    assert sample["lead_levels_are_all_zero"] is True
    assert sample["lead_section_paths"] == ["[]"]
    assert sample["body_rows_with_an_empty_heading"] == 0
    assert sample["body_first_sentences"] == sample["body_first_sentences_starting_with_the_edit_marker"]
    assert sample["edit_markers_outside_a_first_sentence"] == 0


def test_the_recorded_samples_justify_deriving_a_region_from_the_source_pbf(
    recorded_schema: dict[str, Any],
) -> None:
    from landuse_sentence_relevance.sources.v3 import region_from_source_pbf

    samples = recorded_schema["samples"]
    for name in ("wikipedia_polygons", "website_polygons"):
        assert [region_from_source_pbf(pbf) for pbf in samples[name]["source_pbfs"]] == samples[name][
            "regions"
        ]
    assert [region_from_source_pbf(pbf) for pbf in samples["description_geometry"]["source_pbfs"]] == (
        samples["wikipedia_polygons"]["regions"]
    )
