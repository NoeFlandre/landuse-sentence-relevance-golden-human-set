from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.sources.huggingface import HuggingFaceRowConfig
from landuse_sentence_relevance.sources.v3 import (
    DescriptionSentenceSource,
    WebsiteSentenceSource,
    WikipediaSentenceSource,
)


class CapturingRows:
    instances: ClassVar[list[CapturingRows]] = []

    def __init__(self, config: HuggingFaceRowConfig, loader: object = None) -> None:
        self.config = config
        self.loader = loader
        self.__class__.instances.append(self)

    def __call__(self) -> Iterable[Mapping[str, Any]]:
        return iter(())

    def shards(self) -> tuple[Iterable[Mapping[str, Any]], ...]:
        return ()


def test_build_v3_source_adapters_uses_aligned_pinned_sentence_and_geometry_shards(monkeypatch) -> None:
    import landuse_sentence_relevance.bootstrap as bootstrap

    settings = V3Settings()
    catalog_calls: list[tuple[str, str, Mapping[str, str], int, str | None]] = []

    def pinned_files(
        dataset_id: str,
        revision: str,
        directories: Mapping[str, str],
        target_count: int,
        token: str | None = None,
    ) -> dict[str, tuple[str, ...]]:
        catalog_calls.append((dataset_id, revision, directories, target_count, token))
        return {key: (f"https://example.test/{key}.parquet",) for key in directories}

    CapturingRows.instances.clear()
    monkeypatch.setattr(bootstrap, "pinned_remote_file_urls", pinned_files)
    monkeypatch.setattr(bootstrap, "HuggingFaceDatasetRows", CapturingRows)

    adapters = bootstrap.build_v3_source_adapters(
        settings,
        cell_for_location=lambda latitude, longitude: "8928308280fffff",
    )

    assert isinstance(adapters.description, DescriptionSentenceSource)
    assert isinstance(adapters.wikipedia, WikipediaSentenceSource)
    assert isinstance(adapters.website, WebsiteSentenceSource)
    assert adapters.description.max_rows_per_shard == settings.max_rows_per_shard
    assert adapters.description.min_language_score == settings.description_min_language_score
    assert adapters.description.max_text_characters == settings.max_text_characters
    assert adapters.description.max_join_entries == settings.max_join_entries
    assert adapters.wikipedia.max_join_entries == settings.max_join_entries
    assert adapters.website.min_language_probability == settings.website_min_language_probability
    assert [call[0] for call in catalog_calls] == [
        "NoeFlandre/osm-polygon-description-tag",
        "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
        "NoeFlandre/osm-polygon-website-tag",
    ]
    assert [call[1] for call in catalog_calls] == [
        settings.description_dataset_revision,
        settings.wikipedia_dataset_revision,
        settings.website_dataset_revision,
    ]
    assert [call[2] for call in catalog_calls] == [
        {"sentences": "language-v1/data", "geometry": "data"},
        {"sentences": "wikipedia/sentences", "polygons": "polygons"},
        {"polygons": "polygons"},
    ]
    assert all(call[3] == settings.remote_file_sample_count for call in catalog_calls)

    configs = [instance.config for instance in CapturingRows.instances]
    website_columns = configs[-1].columns
    geometry_columns = configs[1].columns
    wikipedia_sentence_columns = configs[2].columns
    assert website_columns is not None
    assert geometry_columns is not None
    assert wikipedia_sentence_columns is not None
    assert "website_sentence_status" in website_columns
    assert "contact_website_sentence_status" in website_columns
    assert "bbox_min_x" in geometry_columns
    assert "bbox_max_y" in geometry_columns
    assert "region" in geometry_columns
    assert {"is_lead", "is_title", "source_url"}.issubset(wikipedia_sentence_columns)
    assert [
        (config.dataset_id, config.revision, config.config, config.split, config.remote_files)
        for config in configs
    ] == [
        (
            settings.description_dataset_id,
            settings.description_dataset_revision,
            settings.description_sentences_config,
            settings.description_sentences_split,
            ("https://example.test/sentences.parquet",),
        ),
        (
            settings.description_dataset_id,
            settings.description_dataset_revision,
            settings.description_geometry_config,
            settings.description_geometry_split,
            ("https://example.test/geometry.parquet",),
        ),
        (
            settings.wikipedia_dataset_id,
            settings.wikipedia_dataset_revision,
            settings.wikipedia_sentences_config,
            settings.wikipedia_sentences_split,
            ("https://example.test/sentences.parquet",),
        ),
        (
            settings.wikipedia_dataset_id,
            settings.wikipedia_dataset_revision,
            settings.wikipedia_polygons_config,
            settings.wikipedia_polygons_split,
            ("https://example.test/polygons.parquet",),
        ),
        (
            settings.website_dataset_id,
            settings.website_dataset_revision,
            settings.website_config,
            settings.website_split,
            ("https://example.test/polygons.parquet",),
        ),
    ]
