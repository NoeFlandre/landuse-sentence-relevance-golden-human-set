from pathlib import Path

from landuse_sentence_relevance.config import Settings

SEAGATE_DATA_ROOT = Path("/Volumes/Seagate M3/projects/landuse-sentence-relevance-golden-human-set")


def test_defaults_pin_the_approved_data_and_model_revisions() -> None:
    settings = Settings.from_env({})

    assert settings.wikipedia_dataset_revision == "7c2a123ba2d4b27db415af6153deab9b98b75ec1"
    assert settings.website_dataset_revision == "2c68154460f0bee314b887217ba54b23e3a2e181"
    assert settings.sat_model_revision == "d70c72a9331b2d5a9e82baad00c64964a23a09bb"
    assert settings.language_model_revision == "43fe88d75e94b11283b66daccbbe4a73e7bc1361"
    assert settings.sat_tokenizer_revision == "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
    assert settings.candidate_cell_count == 256
    assert settings.minimum_candidate_cells == 256
    assert settings.candidate_pool_cells_per_source == 128
    assert settings.candidate_capacity_per_stratum == 1
    assert settings.minimum_website_candidates_per_cell == 1
    assert settings.minimum_cell_distance_km == 500.0


def test_defaults_keep_all_persistent_runtime_data_on_the_seagate_drive() -> None:
    settings = Settings.from_env({})

    assert settings.data_root == SEAGATE_DATA_ROOT
    assert settings.candidate_pool_path == SEAGATE_DATA_ROOT / "candidate-pool.json"
    assert settings.session_path == SEAGATE_DATA_ROOT / "annotations.jsonl"
    assert settings.model_cache_dir == SEAGATE_DATA_ROOT / "runtime-cache"
    assert settings.hf_auth_dir == SEAGATE_DATA_ROOT / "huggingface-auth"


def test_environment_can_change_only_runtime_paths_and_thresholds() -> None:
    settings = Settings.from_env(
        {
            "SESSION_PATH": "/tmp/session.jsonl",
            "CANDIDATE_POOL_PATH": "/tmp/candidate-pool.json",
            "MODEL_CACHE_DIR": "/tmp/models",
            "HF_AUTH_DIR": "/tmp/huggingface-auth",
            "WEBSITE_LANGUAGE_MIN_CONFIDENCE": "0.95",
        }
    )

    assert settings.session_path == Path("/tmp/session.jsonl")
    assert settings.candidate_pool_path == Path("/tmp/candidate-pool.json")
    assert settings.model_cache_dir == Path("/tmp/models")
    assert settings.hf_auth_dir == Path("/tmp/huggingface-auth")
    assert settings.website_language_min_confidence == 0.95
    assert settings.wikipedia_dataset_revision == "7c2a123ba2d4b27db415af6153deab9b98b75ec1"
