"""Build the V2 candidate pool and annotation workflow.

The V2 path predates the V3 streams: it reads two datasets rather than five and
targets cells per source rather than a quota table. It is kept because the V2
benchmark is what reserves cells for V3, so its pool has to stay reproducible.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from landuse_sentence_relevance.bootstrap.runtime import (
    _h3_geometry,
    _prepare_runtime,
    _workflow,
)
from landuse_sentence_relevance.config import Settings
from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.sampling import BoundedCandidatePool, FinalizedCandidatePool
from landuse_sentence_relevance.domain.stratification import DEFAULT_SOURCES
from landuse_sentence_relevance.models.language_identifier import CommonLinguaIdentifier
from landuse_sentence_relevance.models.sentence_splitter import SaTSentenceSplitter
from landuse_sentence_relevance.sources.huggingface import HuggingFaceDatasetRows, HuggingFaceRowConfig
from landuse_sentence_relevance.sources.remote_files import pinned_remote_file_urls
from landuse_sentence_relevance.sources.website import WebsiteCandidateSource
from landuse_sentence_relevance.sources.wikipedia import WikipediaCandidateSource
from landuse_sentence_relevance.storage.candidate_pool import CandidatePoolStore
from landuse_sentence_relevance.storage.candidate_progress import CandidateProgressStore
from landuse_sentence_relevance.workflow import AnnotationWorkflow

logger = logging.getLogger(__name__)
_PROGRESS_CHECKPOINT_INTERVAL = 32


_WIKIPEDIA_COLUMNS = {
    "polygons": ("polygon_id", "has_english_wikipedia", "lat", "lon", "name", "region"),
    "polygon_document_links": ("polygon_id", "document_id", "project", "language"),
    "wikipedia_sections": ("document_id", "section_id", "section_index", "language", "text", "page_id"),
}
_WEBSITE_COLUMNS = (
    "polygon_id",
    "osm_id",
    "lat",
    "lon",
    "name",
    "region",
    "website",
    "contact_website",
    "website_text",
    "contact_website_text",
)
_WIKIPEDIA_DIRECTORIES = {
    "polygons": "polygons",
    "polygon_document_links": "polygon_document_links",
    "wikipedia_sections": "wikipedia/sections",
}
_WEBSITE_DIRECTORIES = {"polygons": "polygons"}


def _wikipedia_rows(
    settings: Settings,
    config: str,
    remote_files: Mapping[str, tuple[str, ...]],
) -> HuggingFaceDatasetRows:
    return HuggingFaceDatasetRows(
        HuggingFaceRowConfig(
            dataset_id=settings.wikipedia_dataset_id,
            revision=settings.wikipedia_dataset_revision,
            split=config,
            config=config,
            columns=_WIKIPEDIA_COLUMNS.get(config),
            remote_files=remote_files[config],
        )
    )


def _collect_candidates(
    candidates: Iterable[Candidate],
    pool: BoundedCandidatePool,
    progress_store: CandidateProgressStore,
    metadata: Mapping[str, Any],
) -> int:
    collected = 0
    try:
        for candidate in candidates:
            pool.add(candidate)
            collected += 1
            if collected % _PROGRESS_CHECKPOINT_INTERVAL == 0:
                progress_store.save(pool.snapshot(), metadata)
                logger.info("Saved candidate progress checkpoint after %d new candidates", collected)
    finally:
        progress_store.save(pool.snapshot(), metadata)
    return collected


def _candidate_cells(pool: BoundedCandidatePool, source: Source) -> frozenset[str]:
    return frozenset(candidate.h3_cell for candidate in pool.snapshot() if candidate.source is source)


def _remaining_candidate_cells(
    pool: BoundedCandidatePool,
    source: Source,
    target_cells: int,
) -> int:
    return max(0, target_cells - len(_candidate_cells(pool, source)))


def _candidate_cells_to_collect(
    pool: BoundedCandidatePool,
    source: Source,
    target_cells: int,
    retry_target: int,
) -> int:
    remaining = _remaining_candidate_cells(pool, source, target_cells)
    return remaining if remaining else retry_target


def _retry_target(settings: Settings) -> int:
    return max(1, settings.candidate_pool_cells_per_source)


def _source_has_buffer(pool: BoundedCandidatePool, source: Source, minimum_cells: int) -> bool:
    return len(_candidate_cells(pool, source)) >= minimum_cells


def _try_finalize(
    pool: BoundedCandidatePool,
    target_cells_per_source: int,
    center_of_cell: Callable[[str], tuple[float, float]],
    minimum_distance_km: float,
) -> FinalizedCandidatePool | None:
    if not all(_source_has_buffer(pool, source, target_cells_per_source) for source in DEFAULT_SOURCES):
        return None
    try:
        return pool.finalize(
            target_cells_per_source=target_cells_per_source,
            center_of_cell=center_of_cell,
            minimum_distance_km=minimum_distance_km,
        )
    except ValueError as error:
        logger.info("Candidate progress is not finalizable yet: %s", error)
        return None


def _load_candidate_pool(
    settings: Settings,
    progress_store: CandidateProgressStore,
    metadata: Mapping[str, Any],
) -> BoundedCandidatePool:
    progress_candidates = progress_store.load(metadata)
    pool = BoundedCandidatePool(
        capacity_per_stratum=settings.candidate_capacity_per_stratum,
        seed=settings.seed,
        sources=DEFAULT_SOURCES,
    )
    if progress_candidates is not None:
        for candidate in progress_candidates:
            pool.add(candidate)
        logger.info(
            "Resuming candidate progress from %s (%d compact candidates)",
            settings.candidate_progress_path,
            len(progress_candidates),
        )
    return pool


def _remote_files(
    settings: Settings,
) -> tuple[Mapping[str, tuple[str, ...]], Mapping[str, tuple[str, ...]]]:
    logger.info(
        "Selecting %d aligned remote Parquet shards per source dataset",
        settings.remote_file_sample_count,
    )
    wikipedia_remote_files = pinned_remote_file_urls(
        settings.wikipedia_dataset_id,
        settings.wikipedia_dataset_revision,
        _WIKIPEDIA_DIRECTORIES,
        settings.remote_file_sample_count,
        token=settings.hf_token,
    )
    website_remote_files = pinned_remote_file_urls(
        settings.website_dataset_id,
        settings.website_dataset_revision,
        _WEBSITE_DIRECTORIES,
        settings.remote_file_sample_count,
        token=settings.hf_token,
    )
    return wikipedia_remote_files, website_remote_files


def _wikipedia_row_loaders(
    settings: Settings,
    remote_files: Mapping[str, tuple[str, ...]],
) -> tuple[
    Callable[[str], Iterable[Mapping[str, Any]]],
    Callable[[str], Iterable[Iterable[Mapping[str, Any]]]],
]:
    def wikipedia_rows(config: str) -> Iterable[Mapping[str, Any]]:
        return _wikipedia_rows(settings, config, remote_files)()

    def wikipedia_row_shards(config: str) -> Iterable[Iterable[Mapping[str, Any]]]:
        return _wikipedia_rows(settings, config, remote_files).shards()

    return wikipedia_rows, wikipedia_row_shards


def _website_row_loaders(
    settings: Settings,
    remote_files: Mapping[str, tuple[str, ...]],
) -> tuple[HuggingFaceDatasetRows, Callable[[], Iterable[Iterable[Mapping[str, Any]]]]]:
    website_rows = HuggingFaceDatasetRows(
        HuggingFaceRowConfig(
            dataset_id=settings.website_dataset_id,
            revision=settings.website_dataset_revision,
            split=settings.website_split,
            config=settings.website_config,
            columns=_WEBSITE_COLUMNS,
            remote_files=remote_files["polygons"],
        )
    )

    def website_row_shards() -> Iterable[Iterable[Mapping[str, Any]]]:
        return website_rows.shards()

    return website_rows, website_row_shards


def _load_models(
    settings: Settings,
) -> tuple[SaTSentenceSplitter, CommonLinguaIdentifier]:
    logger.info("Loading SaT sentence splitter model")
    splitter = SaTSentenceSplitter(
        model_id=settings.sat_model_id,
        revision=settings.sat_model_revision,
        tokenizer_id=settings.sat_tokenizer_id,
        tokenizer_revision=settings.sat_tokenizer_revision,
        cache_dir=settings.model_cache_dir,
        batch_size=settings.sat_batch_size,
        max_workers=settings.sat_workers,
    )
    logger.info("Sentence splitter ready (model cache is reusable between runs)")
    logger.info("Loading CommonLingua English detector model")
    language_identifier = CommonLinguaIdentifier(
        model_id=settings.language_model_id,
        revision=settings.language_model_revision,
        min_confidence=settings.website_language_min_confidence,
        cache_dir=settings.model_cache_dir,
        device=settings.language_model_device,
    )
    logger.info("English detector ready (model cache is reusable between runs)")
    return splitter, language_identifier


def _excluded_cells(pool: BoundedCandidatePool) -> frozenset[str]:
    return _candidate_cells(pool, Source.WIKIPEDIA) | _candidate_cells(pool, Source.WEBSITE)


def _collection_targets(
    pool: BoundedCandidatePool,
    settings: Settings,
) -> tuple[int, int]:
    retry_target = _retry_target(settings)
    wikipedia_cells_to_collect = _candidate_cells_to_collect(
        pool,
        Source.WIKIPEDIA,
        target_cells=settings.minimum_candidate_cells,
        retry_target=retry_target,
    )
    remaining_website_cells = _remaining_candidate_cells(
        pool,
        Source.WEBSITE,
        target_cells=settings.minimum_candidate_cells,
    )
    website_cells_to_collect = _candidate_cells_to_collect(
        pool,
        Source.WEBSITE,
        target_cells=settings.minimum_candidate_cells,
        retry_target=retry_target,
    )
    if website_cells_to_collect != remaining_website_cells:
        logger.info(
            "Candidate progress is not finalizable; collecting %d additional website cells",
            website_cells_to_collect,
        )
    return wikipedia_cells_to_collect, website_cells_to_collect


def _collect_wikipedia_candidates(
    settings: Settings,
    pool: BoundedCandidatePool,
    progress_store: CandidateProgressStore,
    metadata: Mapping[str, Any],
    splitter: SaTSentenceSplitter,
    wikipedia_rows: Callable[[str], Iterable[Mapping[str, Any]]],
    wikipedia_row_shards: Callable[[str], Iterable[Iterable[Mapping[str, Any]]]],
    cell_for_location: Callable[[float, float], str],
    center_of_cell: Callable[[str], tuple[float, float]],
    cells_to_collect: int,
) -> None:
    if cells_to_collect == 0:
        logger.info("Wikipedia candidates already checkpointed; skipping that source scan")
        return
    source = WikipediaCandidateSource(
        row_loader=wikipedia_rows,
        row_shards_loader=wikipedia_row_shards,
        splitter=splitter,
        cell_for_location=cell_for_location,
        max_polygons_per_cell=settings.max_polygons_per_cell,
        max_polygon_rows_per_shard=settings.max_polygon_rows_per_shard,
        max_section_rows_per_shard=settings.max_section_rows_per_shard,
        max_text_characters=settings.wikipedia_max_text_characters,
        max_stream_workers=settings.stream_workers,
        max_candidates_per_cell=settings.candidate_capacity_per_stratum,
        candidate_cell_count=settings.candidate_cell_count,
        center_of_cell=center_of_cell,
        minimum_candidate_cells=cells_to_collect,
        excluded_cells=_excluded_cells(pool),
        seed=settings.seed,
    )
    logger.info("Collecting Wikipedia candidates from the pinned streamed revisions")
    _collect_candidates(source.iter_candidates(), pool, progress_store, metadata)
    logger.info("Wikipedia candidate collection complete")


def _collect_website_candidates(
    settings: Settings,
    pool: BoundedCandidatePool,
    progress_store: CandidateProgressStore,
    metadata: Mapping[str, Any],
    splitter: SaTSentenceSplitter,
    language_identifier: CommonLinguaIdentifier,
    website_rows: HuggingFaceDatasetRows,
    website_row_shards: Callable[[], Iterable[Iterable[Mapping[str, Any]]]],
    cell_for_location: Callable[[float, float], str],
    center_of_cell: Callable[[str], tuple[float, float]],
    cells_to_collect: int,
) -> None:
    source = WebsiteCandidateSource(
        row_loader=website_rows,
        splitter=splitter,
        language_identifier=language_identifier,
        cell_for_location=cell_for_location,
        candidate_cell_count=settings.candidate_cell_count,
        center_of_cell=center_of_cell,
        seed=settings.seed,
        max_candidates_per_cell=settings.candidate_capacity_per_stratum,
        minimum_candidate_cells=cells_to_collect,
        minimum_candidates_per_cell=settings.minimum_website_candidates_per_cell,
        max_rows_per_cell=settings.website_rows_per_cell,
        max_text_characters=settings.website_max_text_characters,
        excluded_cells=_excluded_cells(pool),
        row_shards_loader=website_row_shards,
        max_discovery_rows_per_shard=settings.max_website_discovery_rows_per_shard,
        max_stream_workers=settings.stream_workers,
        require_paragraph=True,
    )
    logger.info("Collecting website candidates from the pinned streamed revision")
    _collect_candidates(source.iter_candidates(), pool, progress_store, metadata)
    logger.info("Website candidate collection complete")


def build_workflow(settings: Settings) -> AnnotationWorkflow:
    logger.info("Starting annotation workflow")
    cache = _prepare_runtime(settings)
    pool_store = CandidatePoolStore(settings.candidate_pool_path)
    progress_store = CandidateProgressStore(settings.candidate_progress_path)
    metadata = _candidate_pool_metadata(settings)
    persisted_pool = pool_store.load(metadata)
    if persisted_pool is not None:
        logger.info(
            "Reusable candidate pool found at %s; skipping streamed rows and sentence splitting",
            settings.candidate_pool_path,
        )
        return _workflow(settings, cache, persisted_pool)

    pool = _load_candidate_pool(settings, progress_store, metadata)
    cell_for_location, center_of_cell = _h3_geometry(settings)
    resumed_pool = _try_finalize(
        pool,
        target_cells_per_source=settings.candidate_pool_cells_per_source,
        center_of_cell=center_of_cell,
        minimum_distance_km=settings.minimum_cell_distance_km,
    )
    if resumed_pool is not None:
        logger.info("Candidate progress already satisfies the final pool constraints")
        pool_store.save(resumed_pool, metadata)
        return _workflow(settings, cache, resumed_pool)

    wikipedia_remote_files, website_remote_files = _remote_files(settings)
    wikipedia_rows, wikipedia_row_shards = _wikipedia_row_loaders(settings, wikipedia_remote_files)
    website_rows, website_row_shards = _website_row_loaders(settings, website_remote_files)
    wikipedia_cells_to_collect, website_cells_to_collect = _collection_targets(pool, settings)
    splitter, language_identifier = _load_models(settings)
    _collect_wikipedia_candidates(
        settings,
        pool,
        progress_store,
        metadata,
        splitter,
        wikipedia_rows,
        wikipedia_row_shards,
        cell_for_location,
        center_of_cell,
        wikipedia_cells_to_collect,
    )
    _collect_website_candidates(
        settings,
        pool,
        progress_store,
        metadata,
        splitter,
        language_identifier,
        website_rows,
        website_row_shards,
        cell_for_location,
        center_of_cell,
        website_cells_to_collect,
    )
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


def _candidate_pool_metadata(settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "sentence_selection": "contextual-v2-paragraph",
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
            "website_max_text_characters": settings.website_max_text_characters,
            "wikipedia_max_text_characters": settings.wikipedia_max_text_characters,
            "max_polygons_per_cell": settings.max_polygons_per_cell,
            "max_polygon_rows_per_shard": settings.max_polygon_rows_per_shard,
            "max_section_rows_per_shard": settings.max_section_rows_per_shard,
            "max_website_discovery_rows_per_shard": settings.max_website_discovery_rows_per_shard,
            "minimum_cell_distance_km": settings.minimum_cell_distance_km,
            "remote_file_sample_count": settings.remote_file_sample_count,
        },
    }
