"""The five immutable V3 upstream streams and how production opens them.

Every projection here is checked against the recorded upstream schema, so this
module is the single place a column name has to be right.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.sources.huggingface import (
    HuggingFaceDatasetLoader,
    HuggingFaceDatasetRows,
    HuggingFaceRowConfig,
)
from landuse_sentence_relevance.sources.remote_files import pinned_remote_file_urls

logger = logging.getLogger(__name__)

_V3_DESCRIPTION_SENTENCE_COLUMNS = (
    "description_identity",
    "source_pbf",
    "osm_type",
    "osm_id",
    "language_code",
    "top_score",
    "sentences",
)
_V3_DESCRIPTION_GEOMETRY_COLUMNS = (
    "source_pbf",
    "osm_type",
    "osm_id",
    "osm_url",
    "name",
    "bbox_min_x",
    "bbox_min_y",
    "bbox_max_x",
    "bbox_max_y",
)
_V3_WIKIPEDIA_SENTENCE_COLUMNS = (
    "sentence_id",
    "document_id",
    "section_id",
    "wikidata",
    "project",
    "language",
    "page_id",
    "section_index",
    "heading",
    "sentence_index",
    "text",
)
_V3_WIKIPEDIA_POLYGON_COLUMNS = (
    "polygon_id",
    "wikidata",
    "has_english_wikipedia",
    "lat",
    "lon",
    "name",
    "region",
)
_V3_WEBSITE_COLUMNS = (
    "polygon_id",
    "lat",
    "lon",
    "name",
    "region",
    "website",
    "contact_website",
    "website_language",
    "website_language_probability",
    "website_sentences",
    "website_sentence_status",
    "contact_website_language",
    "contact_website_language_probability",
    "contact_website_sentences",
    "contact_website_sentence_status",
)


@dataclass(frozen=True, slots=True)
class V3StreamSpec:
    """One immutable upstream V3 stream, with the exact columns production reads.

    Every projection here is asserted against the recorded upstream schema in
    ``tests/fixtures/v3_upstream_schema.json``, so a column the pinned revision
    does not publish cannot reach a stream and fail with ``ArrowInvalid``.
    """

    name: str
    dataset_id: str
    revision: str
    config: str
    split: str
    directory: str
    columns: tuple[str, ...]


def v3_stream_specs(settings: V3Settings) -> tuple[V3StreamSpec, ...]:
    """Return the five V3 streams a run opens, in a deterministic order."""

    return (
        V3StreamSpec(
            name="description_sentences",
            dataset_id=settings.description_dataset_id,
            revision=settings.description_dataset_revision,
            config=settings.description_sentences_config,
            split=settings.description_sentences_split,
            directory="language-v1/data",
            columns=_V3_DESCRIPTION_SENTENCE_COLUMNS,
        ),
        V3StreamSpec(
            name="description_geometry",
            dataset_id=settings.description_dataset_id,
            revision=settings.description_dataset_revision,
            config=settings.description_geometry_config,
            split=settings.description_geometry_split,
            directory="data",
            columns=_V3_DESCRIPTION_GEOMETRY_COLUMNS,
        ),
        V3StreamSpec(
            name="wikipedia_sentences",
            dataset_id=settings.wikipedia_dataset_id,
            revision=settings.wikipedia_dataset_revision,
            config=settings.wikipedia_sentences_config,
            split=settings.wikipedia_sentences_split,
            directory="wikipedia/sentences",
            columns=_V3_WIKIPEDIA_SENTENCE_COLUMNS,
        ),
        V3StreamSpec(
            name="wikipedia_polygons",
            dataset_id=settings.wikipedia_dataset_id,
            revision=settings.wikipedia_dataset_revision,
            config=settings.wikipedia_polygons_config,
            split=settings.wikipedia_polygons_split,
            directory="polygons",
            columns=_V3_WIKIPEDIA_POLYGON_COLUMNS,
        ),
        V3StreamSpec(
            name="website_polygons",
            dataset_id=settings.website_dataset_id,
            revision=settings.website_dataset_revision,
            config=settings.website_config,
            split=settings.website_split,
            directory="polygons",
            columns=_V3_WEBSITE_COLUMNS,
        ),
    )


def v3_streams_by_dataset(
    settings: V3Settings,
) -> tuple[tuple[str, str, tuple[V3StreamSpec, ...]], ...]:
    """Group the V3 streams by dataset so each catalog is listed once."""

    grouped: dict[tuple[str, str], list[V3StreamSpec]] = {}
    for spec in v3_stream_specs(settings):
        grouped.setdefault((spec.dataset_id, spec.revision), []).append(spec)
    return tuple((dataset_id, revision, tuple(specs)) for (dataset_id, revision), specs in grouped.items())


def v3_remote_files(settings: V3Settings) -> dict[str, tuple[str, ...]]:
    """Select the aligned remote Parquet shards of every V3 stream, by stream name."""

    logger.info(
        "Selecting %d aligned remote V3 Parquet shards per source dataset",
        settings.remote_file_sample_count,
    )
    remote_files: dict[str, tuple[str, ...]] = {}
    for dataset_id, revision, specs in v3_streams_by_dataset(settings):
        remote_files.update(
            pinned_remote_file_urls(
                dataset_id,
                revision,
                {spec.name: spec.directory for spec in specs},
                settings.remote_file_sample_count,
                token=settings.hf_token,
            )
        )
    return remote_files


def v3_row_config(spec: V3StreamSpec, remote_files: tuple[str, ...]) -> HuggingFaceRowConfig:
    """Return the exact streaming configuration production opens for one stream."""

    return HuggingFaceRowConfig(
        dataset_id=spec.dataset_id,
        revision=spec.revision,
        split=spec.split,
        config=spec.config,
        columns=spec.columns,
        remote_files=remote_files,
    )


def _v3_rows(
    spec: V3StreamSpec,
    remote_files: Mapping[str, tuple[str, ...]],
    loader: HuggingFaceDatasetLoader | None,
) -> HuggingFaceDatasetRows:
    return HuggingFaceDatasetRows(v3_row_config(spec, remote_files[spec.name]), loader=loader)
