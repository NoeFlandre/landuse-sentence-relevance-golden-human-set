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


def _build(monkeypatch, settings: V3Settings) -> tuple[Any, list[tuple[Any, ...]]]:
    import landuse_sentence_relevance.bootstrap as bootstrap

    catalog_calls: list[tuple[Any, ...]] = []

    def pinned_files(
        dataset_id: str,
        revision: str,
        directories: Mapping[str, str],
        target_count: int,
        token: str | None = None,
    ) -> dict[str, tuple[str, ...]]:
        catalog_calls.append((dataset_id, revision, dict(directories), target_count, token))
        return {key: (f"https://example.test/{key}.parquet",) for key in directories}

    CapturingRows.instances.clear()
    monkeypatch.setattr(bootstrap, "pinned_remote_file_urls", pinned_files)
    monkeypatch.setattr(bootstrap, "HuggingFaceDatasetRows", CapturingRows)
    adapters = bootstrap.build_v3_source_adapters(
        settings, cell_for_location=lambda latitude, longitude: "8928308280fffff"
    )
    return adapters, catalog_calls


def test_build_v3_source_adapters_opens_every_stream_with_its_pinned_specification(
    monkeypatch,
) -> None:
    from landuse_sentence_relevance.bootstrap import v3_stream_specs

    settings = V3Settings()
    adapters, _ = _build(monkeypatch, settings)
    specs = v3_stream_specs(settings)

    assert isinstance(adapters.description, DescriptionSentenceSource)
    assert isinstance(adapters.wikipedia, WikipediaSentenceSource)
    assert isinstance(adapters.website, WebsiteSentenceSource)
    assert [
        (config.dataset_id, config.revision, config.config, config.split, config.columns)
        for config in (instance.config for instance in CapturingRows.instances)
    ] == [(spec.dataset_id, spec.revision, spec.config, spec.split, spec.columns) for spec in specs]
    assert [config.remote_files for config in (instance.config for instance in CapturingRows.instances)] == [
        (f"https://example.test/{spec.name}.parquet",) for spec in specs
    ]


def test_build_v3_source_adapters_lists_each_pinned_catalog_once(monkeypatch) -> None:
    settings = V3Settings()
    _, catalog_calls = _build(monkeypatch, settings)

    assert [call[0] for call in catalog_calls] == [
        settings.description_dataset_id,
        settings.wikipedia_dataset_id,
        settings.website_dataset_id,
    ]
    assert [call[1] for call in catalog_calls] == [
        settings.description_dataset_revision,
        settings.wikipedia_dataset_revision,
        settings.website_dataset_revision,
    ]
    assert [call[2] for call in catalog_calls] == [
        {"description_sentences": "language-v1/data", "description_geometry": "data"},
        {"wikipedia_sentences": "wikipedia/sentences", "wikipedia_polygons": "polygons"},
        {"website_polygons": "polygons"},
    ]
    assert all(call[3] == settings.remote_file_sample_count for call in catalog_calls)
    assert all(call[4] == settings.hf_token for call in catalog_calls)


def test_build_v3_source_adapters_propagates_every_configured_bound(monkeypatch) -> None:
    settings = V3Settings()
    adapters, _ = _build(monkeypatch, settings)

    assert adapters.description.max_rows_per_shard == settings.max_rows_per_shard
    assert adapters.description.min_language_score == settings.description_min_language_score
    assert adapters.description.max_text_characters == settings.max_text_characters
    assert adapters.description.max_join_entries == settings.max_join_entries
    assert adapters.wikipedia.max_join_entries == settings.max_join_entries
    assert adapters.wikipedia.max_text_characters == settings.max_text_characters
    assert adapters.website.min_language_probability == settings.website_min_language_probability
    assert adapters.website.max_rows_per_shard == settings.max_rows_per_shard
