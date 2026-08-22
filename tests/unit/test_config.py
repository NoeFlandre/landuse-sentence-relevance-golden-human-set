from pathlib import Path

from landuse_sentence_relevance.config import Settings


def test_defaults_pin_the_approved_data_and_model_revisions() -> None:
    settings = Settings.from_env({})

    assert settings.wikipedia_dataset_revision == "7c2a123ba2d4b27db415af6153deab9b98b75ec1"
    assert settings.website_dataset_revision == "2c68154460f0bee314b887217ba54b23e3a2e181"
    assert settings.sat_model_revision == "d70c72a9331b2d5a9e82baad00c64964a23a09bb"
    assert settings.language_model_revision == "43fe88d75e94b11283b66daccbbe4a73e7bc1361"
    assert settings.sat_tokenizer_revision == "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
    assert settings.target_cell_count == 25


def test_environment_can_change_only_runtime_paths_and_thresholds() -> None:
    settings = Settings.from_env(
        {
            "SESSION_PATH": "/tmp/session.jsonl",
            "MODEL_CACHE_DIR": "/tmp/models",
            "WEBSITE_LANGUAGE_MIN_CONFIDENCE": "0.95",
        }
    )

    assert settings.session_path == Path("/tmp/session.jsonl")
    assert settings.model_cache_dir == Path("/tmp/models")
    assert settings.website_language_min_confidence == 0.95
    assert settings.wikipedia_dataset_revision == "7c2a123ba2d4b27db415af6153deab9b98b75ec1"
