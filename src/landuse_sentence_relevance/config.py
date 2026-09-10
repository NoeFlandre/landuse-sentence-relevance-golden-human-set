from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PROJECT_NAME = "landuse-sentence-relevance-golden-human-set"
DEFAULT_DATA_ROOT = Path("/Volumes/Seagate M3/projects/landuse-sentence-relevance-golden-human-set")
DEFAULT_RESULTS_ROOT = DEFAULT_DATA_ROOT / "results"
DEFAULT_STATE_ROOT = DEFAULT_DATA_ROOT / "state"


@dataclass(frozen=True, slots=True)
class Settings:
    data_root: Path = DEFAULT_DATA_ROOT
    wikipedia_dataset_id: str = "NoeFlandre/osm-polygon-wikidata-and-wikipedia"
    wikipedia_dataset_revision: str = "7c2a123ba2d4b27db415af6153deab9b98b75ec1"
    website_dataset_id: str = "NoeFlandre/osm-polygon-website-tag"
    website_dataset_revision: str = "2c68154460f0bee314b887217ba54b23e3a2e181"
    output_dataset_id: str = "NoeFlandre/landuse-sentence-relevance-golden-human-set"
    sat_model_id: str = "segment-any-text/sat-12l-sm"
    sat_model_revision: str = "d70c72a9331b2d5a9e82baad00c64964a23a09bb"
    sat_tokenizer_id: str = "FacebookAI/xlm-roberta-base"
    sat_tokenizer_revision: str = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
    sat_batch_size: int = 4
    sat_workers: int = 1
    language_model_id: str = "PleIAs/CommonLingua"
    language_model_revision: str = "43fe88d75e94b11283b66daccbbe4a73e7bc1361"
    language_model_device: str = "auto"
    website_config: str = "default"
    website_split: str = "polygons"
    candidate_pool_path: Path = DEFAULT_RESULTS_ROOT / "candidates/v2/pool.json"
    candidate_progress_path: Path = DEFAULT_RESULTS_ROOT / "candidates/v2/progress.json"
    session_path: Path = DEFAULT_RESULTS_ROOT / "annotations/sessions/v2-wikipedia.jsonl"
    model_cache_dir: Path = DEFAULT_STATE_ROOT / "runtime-cache"
    hf_auth_dir: Path = DEFAULT_STATE_ROOT / "huggingface-auth"
    output_dataset_split: str = "v2"
    h3_resolution: int = 3
    candidate_cell_count: int = 512
    minimum_candidate_cells: int = 320
    candidate_pool_cells_per_source: int = 256
    candidate_capacity_per_stratum: int = 1
    minimum_website_candidates_per_cell: int = 1
    website_rows_per_cell: int = 16
    max_polygons_per_cell: int = 8
    website_language_min_confidence: float = 0.90
    website_max_text_characters: int = 400
    wikipedia_max_text_characters: int = 800
    minimum_cell_distance_km: float = 500.0
    remote_file_sample_count: int = 32
    stream_workers: int = 32
    max_polygon_rows_per_shard: int = 500
    max_section_rows_per_shard: int = 10_000
    max_website_discovery_rows_per_shard: int = 4_000
    seed: str = PROJECT_NAME
    hf_token: str | None = None

    @classmethod
    def from_env(cls, values: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if values is None else values
        data_root = Path(env.get("PROJECT_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
        results_root = data_root / "results"
        state_root = data_root / "state"
        return cls(
            data_root=data_root,
            candidate_pool_path=Path(
                env.get("CANDIDATE_POOL_PATH", str(results_root / "candidates/v2/pool.json"))
            ),
            candidate_progress_path=Path(
                env.get("CANDIDATE_PROGRESS_PATH", str(results_root / "candidates/v2/progress.json"))
            ),
            session_path=Path(
                env.get("SESSION_PATH", str(results_root / "annotations/sessions/v2-wikipedia.jsonl"))
            ),
            model_cache_dir=Path(env.get("MODEL_CACHE_DIR", str(state_root / "runtime-cache"))),
            hf_auth_dir=Path(env.get("HF_AUTH_DIR", str(state_root / "huggingface-auth"))),
            output_dataset_split=env.get("OUTPUT_DATASET_SPLIT", "v2"),
            sat_batch_size=int(env.get("SAT_BATCH_SIZE", "4")),
            sat_workers=int(env.get("SAT_WORKERS", "1")),
            remote_file_sample_count=int(env.get("REMOTE_FILE_SAMPLE_COUNT", "32")),
            stream_workers=int(env.get("STREAM_WORKERS", "32")),
            max_polygon_rows_per_shard=int(env.get("MAX_POLYGON_ROWS_PER_SHARD", "500")),
            max_section_rows_per_shard=int(env.get("MAX_SECTION_ROWS_PER_SHARD", "10000")),
            max_website_discovery_rows_per_shard=int(env.get("MAX_WEBSITE_DISCOVERY_ROWS_PER_SHARD", "4000")),
            website_max_text_characters=int(env.get("WEBSITE_MAX_TEXT_CHARACTERS", "400")),
            wikipedia_max_text_characters=int(env.get("WIKIPEDIA_MAX_TEXT_CHARACTERS", "800")),
            website_rows_per_cell=int(env.get("WEBSITE_ROWS_PER_CELL", "16")),
            website_language_min_confidence=float(env.get("WEBSITE_LANGUAGE_MIN_CONFIDENCE", "0.90")),
            language_model_device=env.get("LANGUAGE_MODEL_DEVICE", "auto"),
            hf_token=env.get("HF_TOKEN"),
        )
