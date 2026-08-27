from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from landuse_sentence_relevance.config import Settings
from landuse_sentence_relevance.domain.sampling import BoundedCandidatePool
from landuse_sentence_relevance.models.language_identifier import CommonLinguaIdentifier
from landuse_sentence_relevance.models.sentence_splitter import SaTSentenceSplitter
from landuse_sentence_relevance.sources.huggingface import HuggingFaceDatasetRows, HuggingFaceRowConfig
from landuse_sentence_relevance.sources.website import WebsiteCandidateSource
from landuse_sentence_relevance.sources.wikipedia import WikipediaCandidateSource
from landuse_sentence_relevance.storage.auth import HuggingFaceAuth
from landuse_sentence_relevance.storage.cache import ManagedCache
from landuse_sentence_relevance.storage.candidate_pool import CandidatePoolStore
from landuse_sentence_relevance.storage.publisher import DatasetPublisher
from landuse_sentence_relevance.storage.session import AnnotationStore
from landuse_sentence_relevance.workflow import AnnotationWorkflow

logger = logging.getLogger(__name__)


def build_workflow(  # pragma: no cover - full startup needs remote datasets and model weights
    settings: Settings,
) -> AnnotationWorkflow:
    logger.info("Starting annotation workflow")
    cache = ManagedCache(settings.model_cache_dir)
    cache.prepare()
    logger.info("Disposable runtime cache prepared at %s", cache.root)
    authenticated = HuggingFaceAuth(settings.hf_auth_dir).prepare(
        legacy_home=cache.root / "huggingface",
        token=settings.hf_token,
    )
    if authenticated:
        logger.info("Hugging Face login available; reusing the saved credential")
    else:
        logger.info("No saved Hugging Face login found; authenticate once before the final upload")
    pool_store = CandidatePoolStore(settings.candidate_pool_path)
    metadata = _candidate_pool_metadata(settings)
    persisted_pool = pool_store.load(metadata)
    if persisted_pool is not None:
        logger.info(
            "Reusable candidate pool found at %s; skipping streamed rows and sentence splitting",
            settings.candidate_pool_path,
        )
        return _workflow(settings, cache, persisted_pool)
    try:
        from h3 import cell_to_latlng, latlng_to_cell
    except ImportError as error:  # pragma: no cover - dependency installation boundary
        raise RuntimeError("Install project dependencies with `uv sync`") from error

    def cell_for_location(latitude: float, longitude: float) -> str:
        return latlng_to_cell(latitude, longitude, settings.h3_resolution)

    def center_of_cell(cell: str) -> tuple[float, float]:
        return tuple(cell_to_latlng(cell))

    def wikipedia_rows(config: str) -> Iterable[Mapping[str, Any]]:
        return HuggingFaceDatasetRows(
            HuggingFaceRowConfig(
                dataset_id=settings.wikipedia_dataset_id,
                revision=settings.wikipedia_dataset_revision,
                split=config,
                config=config,
            )
        )()

    website_rows = HuggingFaceDatasetRows(
        HuggingFaceRowConfig(
            dataset_id=settings.website_dataset_id,
            revision=settings.website_dataset_revision,
            split=settings.website_split,
            config=settings.website_config,
        )
    )
    logger.info("Loading SaT sentence splitter model")
    splitter = SaTSentenceSplitter(
        model_id=settings.sat_model_id,
        revision=settings.sat_model_revision,
        tokenizer_id=settings.sat_tokenizer_id,
        tokenizer_revision=settings.sat_tokenizer_revision,
        cache_dir=settings.model_cache_dir,
    )
    logger.info("Sentence splitter ready (model cache is reusable between runs)")
    logger.info("Loading CommonLingua English detector model")
    language_identifier = CommonLinguaIdentifier(
        model_id=settings.language_model_id,
        revision=settings.language_model_revision,
        min_confidence=settings.website_language_min_confidence,
        cache_dir=settings.model_cache_dir,
    )
    logger.info("English detector ready (model cache is reusable between runs)")
    pool = BoundedCandidatePool(
        capacity_per_stratum=settings.candidate_capacity_per_stratum,
        seed=settings.seed,
    )

    wikipedia_source = WikipediaCandidateSource(
        row_loader=wikipedia_rows,
        splitter=splitter,
        cell_for_location=cell_for_location,
        max_polygons_per_cell=settings.max_polygons_per_cell,
        max_candidates_per_cell=settings.candidate_capacity_per_stratum,
        candidate_cell_count=settings.candidate_cell_count,
        center_of_cell=center_of_cell,
        minimum_candidate_cells=settings.minimum_candidate_cells,
        seed=settings.seed,
    )
    logger.info("Collecting Wikipedia candidates from the pinned streamed revisions")
    for candidate in wikipedia_source.iter_candidates():
        pool.add(candidate)
    logger.info("Wikipedia candidate collection complete")

    website_source = WebsiteCandidateSource(
        row_loader=website_rows,
        splitter=splitter,
        language_identifier=language_identifier,
        cell_for_location=cell_for_location,
        candidate_cell_count=settings.candidate_cell_count,
        center_of_cell=center_of_cell,
        seed=settings.seed,
        max_candidates_per_cell=settings.candidate_capacity_per_stratum,
        minimum_candidate_cells=settings.minimum_candidate_cells,
        minimum_candidates_per_cell=settings.minimum_website_candidates_per_cell,
        max_rows_per_cell=settings.website_rows_per_cell,
    )
    logger.info("Collecting website candidates from the pinned streamed revision")
    for candidate in website_source.iter_candidates():
        pool.add(candidate)
    logger.info("Website candidate collection complete")

    finalized_pool = pool.finalize(
        target_cells_per_source=settings.candidate_pool_cells_per_source,
        center_of_cell=center_of_cell,
        minimum_distance_km=settings.minimum_cell_distance_km,
    )
    logger.info(
        "Candidate pool ready: %d unique-cell candidates across %d globally spread H3 cells",
        len(finalized_pool.candidates),
        len(finalized_pool.cells),
    )
    pool_store.save(finalized_pool, metadata)
    logger.info("Saved reusable candidate pool to %s", settings.candidate_pool_path)
    return _workflow(settings, cache, finalized_pool)


def _workflow(
    settings: Settings,
    cache: ManagedCache,
    pool: Any,
) -> AnnotationWorkflow:
    return AnnotationWorkflow(
        pool=pool,
        store=AnnotationStore(settings.session_path),
        publisher=DatasetPublisher(
            settings.output_dataset_id,
            token=settings.hf_token,
            prepare=cache.prepare,
            cleanup=cache.cleanup,
        ),
        defer_publish=True,
    )


def _candidate_pool_metadata(settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "seed": settings.seed,
        "h3_resolution": settings.h3_resolution,
        "wikipedia_dataset": {
            "id": settings.wikipedia_dataset_id,
            "revision": settings.wikipedia_dataset_revision,
        },
        "website_dataset": {
            "id": settings.website_dataset_id,
            "revision": settings.website_dataset_revision,
            "config": settings.website_config,
            "split": settings.website_split,
        },
        "sentence_splitter": {
            "id": settings.sat_model_id,
            "revision": settings.sat_model_revision,
            "tokenizer_id": settings.sat_tokenizer_id,
            "tokenizer_revision": settings.sat_tokenizer_revision,
        },
        "language_identifier": {
            "id": settings.language_model_id,
            "revision": settings.language_model_revision,
            "minimum_confidence": settings.website_language_min_confidence,
        },
        "sampling": {
            "candidate_cell_count": settings.candidate_cell_count,
            "minimum_candidate_cells": settings.minimum_candidate_cells,
            "candidate_pool_cells_per_source": settings.candidate_pool_cells_per_source,
            "candidate_capacity_per_stratum": settings.candidate_capacity_per_stratum,
            "minimum_website_candidates_per_cell": settings.minimum_website_candidates_per_cell,
            "website_rows_per_cell": settings.website_rows_per_cell,
            "max_polygons_per_cell": settings.max_polygons_per_cell,
            "minimum_cell_distance_km": settings.minimum_cell_distance_km,
        },
    }
