from pathlib import Path

from landuse_sentence_relevance.config import Settings

SEAGATE_DATA_ROOT = Path("/Volumes/Seagate M3/projects/landuse-sentence-relevance-golden-human-set")
RESULTS_ROOT = SEAGATE_DATA_ROOT / "results"
STATE_ROOT = SEAGATE_DATA_ROOT / "state"


def test_defaults_pin_the_approved_data_and_model_revisions() -> None:
    settings = Settings.from_env({})

    assert settings.wikipedia_dataset_revision == "7c2a123ba2d4b27db415af6153deab9b98b75ec1"
    assert settings.website_dataset_revision == "2c68154460f0bee314b887217ba54b23e3a2e181"
    assert settings.sat_model_revision == "d70c72a9331b2d5a9e82baad00c64964a23a09bb"
    assert settings.language_model_revision == "43fe88d75e94b11283b66daccbbe4a73e7bc1361"
    assert settings.language_model_device == "auto"
    assert settings.sat_tokenizer_revision == "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
    assert settings.candidate_cell_count == 512
    assert settings.minimum_candidate_cells == 320
    assert settings.candidate_pool_cells_per_source == 256
    assert settings.website_max_text_characters == 400
    assert settings.wikipedia_max_text_characters == 800
    assert settings.candidate_capacity_per_stratum == 1
    assert settings.minimum_website_candidates_per_cell == 1
    assert settings.website_rows_per_cell == 16
    assert settings.minimum_cell_distance_km == 500.0
    assert settings.remote_file_sample_count == 32
    assert settings.max_section_rows_per_shard == 10_000
    assert settings.sat_batch_size == 4
    assert settings.sat_workers == 1
    assert settings.stream_workers == 32
    assert settings.max_polygon_rows_per_shard == 500
    assert settings.max_website_discovery_rows_per_shard == 4_000
    assert settings.output_dataset_split == "v2"


def test_defaults_keep_all_persistent_runtime_data_on_the_seagate_drive() -> None:
    settings = Settings.from_env({})

    assert settings.data_root == SEAGATE_DATA_ROOT
    assert settings.candidate_pool_path == RESULTS_ROOT / "candidates/v2/pool.json"
    assert settings.candidate_progress_path == RESULTS_ROOT / "candidates/v2/progress.json"
    assert settings.session_path == RESULTS_ROOT / "annotations/sessions/v2-wikipedia.jsonl"
    assert settings.model_cache_dir == STATE_ROOT / "runtime-cache"
    assert settings.hf_auth_dir == STATE_ROOT / "huggingface-auth"


def test_environment_can_change_only_runtime_paths_and_thresholds() -> None:
    settings = Settings.from_env(
        {
            "SESSION_PATH": "/tmp/session.jsonl",
            "CANDIDATE_POOL_PATH": "/tmp/candidate-pool.json",
            "CANDIDATE_PROGRESS_PATH": "/tmp/candidate-progress.json",
            "MODEL_CACHE_DIR": "/tmp/models",
            "HF_AUTH_DIR": "/tmp/huggingface-auth",
            "OUTPUT_DATASET_SPLIT": "v2-review",
            "WEBSITE_LANGUAGE_MIN_CONFIDENCE": "0.95",
            "LANGUAGE_MODEL_DEVICE": "cpu",
            "REMOTE_FILE_SAMPLE_COUNT": "64",
            "MAX_SECTION_ROWS_PER_SHARD": "5000",
            "SAT_BATCH_SIZE": "2",
            "SAT_WORKERS": "3",
            "STREAM_WORKERS": "5",
            "MAX_POLYGON_ROWS_PER_SHARD": "250",
            "MAX_WEBSITE_DISCOVERY_ROWS_PER_SHARD": "1500",
            "WEBSITE_MAX_TEXT_CHARACTERS": "1500",
            "WIKIPEDIA_MAX_TEXT_CHARACTERS": "1200",
            "WEBSITE_ROWS_PER_CELL": "7",
        }
    )

    assert settings.session_path == Path("/tmp/session.jsonl")
    assert settings.candidate_pool_path == Path("/tmp/candidate-pool.json")
    assert settings.candidate_progress_path == Path("/tmp/candidate-progress.json")
    assert settings.model_cache_dir == Path("/tmp/models")
    assert settings.hf_auth_dir == Path("/tmp/huggingface-auth")
    assert settings.output_dataset_split == "v2-review"
    assert settings.website_language_min_confidence == 0.95
    assert settings.language_model_device == "cpu"
    assert settings.remote_file_sample_count == 64
    assert settings.max_section_rows_per_shard == 5000
    assert settings.sat_batch_size == 2
    assert settings.sat_workers == 3
    assert settings.stream_workers == 5
    assert settings.max_polygon_rows_per_shard == 250
    assert settings.max_website_discovery_rows_per_shard == 1500
    assert settings.website_max_text_characters == 1500
    assert settings.wikipedia_max_text_characters == 1200
    assert settings.website_rows_per_cell == 7
    assert settings.wikipedia_dataset_revision == "7c2a123ba2d4b27db415af6153deab9b98b75ec1"
