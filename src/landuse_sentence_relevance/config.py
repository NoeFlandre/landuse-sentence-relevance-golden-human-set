from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PROJECT_NAME = "landuse-sentence-relevance-golden-human-set"
DEFAULT_DATA_ROOT = Path("/Volumes/Seagate M3/projects/landuse-sentence-relevance-golden-human-set")


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
    language_model_id: str = "PleIAs/CommonLingua"
    language_model_revision: str = "43fe88d75e94b11283b66daccbbe4a73e7bc1361"
    website_config: str = "default"
    website_split: str = "polygons"
    candidate_pool_path: Path = DEFAULT_DATA_ROOT / "candidate-pool.json"
    session_path: Path = DEFAULT_DATA_ROOT / "annotations.jsonl"
    model_cache_dir: Path = DEFAULT_DATA_ROOT / "runtime-cache"
    hf_auth_dir: Path = DEFAULT_DATA_ROOT / "huggingface-auth"
    h3_resolution: int = 3
    candidate_cell_count: int = 256
    minimum_candidate_cells: int = 256
    candidate_pool_cells_per_source: int = 128
    candidate_capacity_per_stratum: int = 1
    minimum_website_candidates_per_cell: int = 1
    website_rows_per_cell: int = 4
    max_polygons_per_cell: int = 8
    website_language_min_confidence: float = 0.90
    minimum_cell_distance_km: float = 500.0
    seed: str = PROJECT_NAME
    hf_token: str | None = None

    @classmethod
    def from_env(cls, values: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if values is None else values
        data_root = Path(env.get("PROJECT_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
        return cls(
            data_root=data_root,
            candidate_pool_path=Path(env.get("CANDIDATE_POOL_PATH", str(data_root / "candidate-pool.json"))),
            session_path=Path(env.get("SESSION_PATH", str(data_root / "annotations.jsonl"))),
            model_cache_dir=Path(env.get("MODEL_CACHE_DIR", str(data_root / "runtime-cache"))),
            hf_auth_dir=Path(env.get("HF_AUTH_DIR", str(data_root / "huggingface-auth"))),
            website_language_min_confidence=float(env.get("WEBSITE_LANGUAGE_MIN_CONFIDENCE", "0.90")),
            hf_token=env.get("HF_TOKEN"),
        )
