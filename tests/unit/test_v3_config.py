from __future__ import annotations

from pathlib import Path

import pytest

from landuse_sentence_relevance.config import (
    DEFAULT_DATA_ROOT,
    PROJECT_NAME,
    Settings,
    V3Settings,
)
from landuse_sentence_relevance.domain.models import Label, Source
from landuse_sentence_relevance.domain.profile import V3_QUOTAS


def test_v3_pins_every_source_dataset_to_an_immutable_revision() -> None:
    settings = V3Settings()

    assert settings.description_dataset_id == "NoeFlandre/osm-polygon-description-tag"
    assert settings.description_dataset_revision == "fec858b679f5ee7e87f0ecfaaa6b7223b2a7f5e2"
    assert settings.wikipedia_dataset_id == "NoeFlandre/osm-polygon-wikidata-and-wikipedia"
    assert settings.wikipedia_dataset_revision == "f48c5aaaec6aecd63ecf5c195565cc2787597f1e"
    assert settings.website_dataset_id == "NoeFlandre/osm-polygon-website-tag"
    assert settings.website_dataset_revision == "5c8e56a50b5679118a28aef057af002209f80a5e"
    assert all(
        revision != "main" and len(revision) == 40
        for revision in (
            settings.description_dataset_revision,
            settings.wikipedia_dataset_revision,
            settings.website_dataset_revision,
        )
    )


def test_v3_names_the_upstream_sentence_level_configurations() -> None:
    settings = V3Settings()

    assert (settings.description_sentences_config, settings.description_sentences_split) == (
        "language-v1",
        "train",
    )
    assert (settings.description_geometry_config, settings.description_geometry_split) == (
        "default",
        "train",
    )
    assert (settings.wikipedia_sentences_config, settings.wikipedia_sentences_split) == (
        "wikipedia_sentences",
        "wikipedia_sentences",
    )
    assert (settings.wikipedia_polygons_config, settings.wikipedia_polygons_split) == (
        "polygons",
        "polygons",
    )
    assert (settings.website_config, settings.website_split) == ("default", "polygons")


def test_v3_uses_paths_that_cannot_collide_with_v2() -> None:
    settings = V3Settings()
    v2 = Settings()

    assert settings.candidate_pool_path == DEFAULT_DATA_ROOT / "results/candidates/v3/pool.json"
    assert settings.candidate_progress_path == DEFAULT_DATA_ROOT / "results/candidates/v3/progress.json"
    assert settings.session_path == DEFAULT_DATA_ROOT / "results/annotations/sessions/v3.jsonl"
    assert settings.output_dataset_split == "v3"
    assert settings.candidate_pool_path != v2.candidate_pool_path
    assert settings.candidate_progress_path != v2.candidate_progress_path
    assert settings.session_path != v2.session_path
    assert settings.output_dataset_split != v2.output_dataset_split


def test_v3_keeps_every_generated_path_under_the_project_data_root(tmp_path: Path) -> None:
    settings = V3Settings.from_env({"PROJECT_DATA_ROOT": str(tmp_path)})

    for path in (settings.candidate_pool_path, settings.candidate_progress_path, settings.session_path):
        assert path.is_relative_to(tmp_path)


def test_v3_targets_three_hundred_rows_from_an_oversized_pool() -> None:
    settings = V3Settings()

    assert settings.h3_resolution == 3
    assert settings.rows_per_source == 100
    assert settings.rows_per_source_label == 50
    assert settings.candidate_cells_per_source >= 400


def test_v3_bounds_the_streaming_joins_it_resolves_geolocation_with() -> None:
    settings = V3Settings()

    assert settings.max_rows_per_shard == 20_000
    assert settings.max_join_entries > 0
    assert settings.max_join_entries <= 1_000_000


def test_v3_does_not_configure_a_local_splitter_or_language_model() -> None:
    fields = set(V3Settings.__dataclass_fields__)

    assert not {field for field in fields if "sat" in field or "language_model" in field}


def test_v3_environment_overrides_stay_explicit() -> None:
    settings = V3Settings.from_env(
        {"PROJECT_DATA_ROOT": "/tmp/example", "V3_CANDIDATE_CELLS_PER_SOURCE": "512"}
    )

    assert settings.candidate_cells_per_source == 512
    assert settings.data_root == Path("/tmp/example")


def test_v3_paths_stay_inside_an_explicitly_chosen_data_root(tmp_path: Path) -> None:
    settings = V3Settings(data_root=tmp_path)

    for path in (
        settings.benchmark_path,
        settings.candidate_pool_path,
        settings.candidate_progress_path,
        settings.session_path,
        settings.model_cache_dir,
        settings.hf_auth_dir,
    ):
        assert path.is_relative_to(settings.data_root)


def test_v3_rejects_a_generated_path_outside_its_data_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must stay under the project data root"):
        V3Settings(data_root=tmp_path, session_path=Path("/elsewhere/v3.jsonl"))


def test_v3_row_targets_agree_with_the_domain_quota_matrix() -> None:
    settings = V3Settings()

    assert settings.rows_per_source == V3_QUOTAS.for_source(Source.WIKIPEDIA)
    assert settings.rows_per_source_label == V3_QUOTAS.required(Source.WIKIPEDIA, Label.YES)
    assert settings.rows_per_source * len(V3_QUOTAS.sources) == V3_QUOTAS.total
    assert settings.h3_resolution == V3_QUOTAS.h3_resolution


def test_v3_oversizes_the_candidate_pool_against_its_own_row_target() -> None:
    settings = V3Settings()

    assert settings.candidate_cells_per_source >= 4 * settings.rows_per_source
    assert settings.candidate_reservoir_cells_per_source >= settings.candidate_cells_per_source


def test_v3_reservoir_capacity_can_be_overridden_without_touching_v2() -> None:
    settings = V3Settings.from_env({"V3_CANDIDATE_RESERVOIR_CELLS_PER_SOURCE": "800"})

    assert settings.candidate_reservoir_cells_per_source == 800
    assert settings.candidate_cells_per_source == V3Settings().candidate_cells_per_source


def test_v3_seed_differs_from_the_v2_seed() -> None:
    assert V3Settings().seed != Settings().seed


def test_v3_source_revisions_are_not_environment_overridable() -> None:
    pinned = V3Settings()
    overridden = V3Settings.from_env(
        {
            "V3_DESCRIPTION_DATASET_REVISION": "main",
            "V3_WIKIPEDIA_DATASET_REVISION": "main",
            "V3_WEBSITE_DATASET_REVISION": "main",
        }
    )

    assert overridden.description_dataset_revision == pinned.description_dataset_revision
    assert overridden.wikipedia_dataset_revision == pinned.wikipedia_dataset_revision
    assert overridden.website_dataset_revision == pinned.website_dataset_revision


def test_v2_settings_defaults_are_untouched_by_the_v3_profile() -> None:
    v2 = Settings()

    assert v2.output_dataset_split == "v2"
    assert v2.candidate_pool_path == DEFAULT_DATA_ROOT / "results/candidates/v2/pool.json"
    assert v2.candidate_progress_path == DEFAULT_DATA_ROOT / "results/candidates/v2/progress.json"
    assert v2.session_path == DEFAULT_DATA_ROOT / "results/annotations/sessions/v2-wikipedia.jsonl"
    assert v2.wikipedia_dataset_revision == "7c2a123ba2d4b27db415af6153deab9b98b75ec1"
    assert v2.website_dataset_revision == "2c68154460f0bee314b887217ba54b23e3a2e181"
    assert v2.seed == PROJECT_NAME
    assert v2.h3_resolution == 3
