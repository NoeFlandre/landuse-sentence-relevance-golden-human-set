from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from landuse_sentence_relevance.config import Settings, V3Settings
from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.profile import V3_QUOTAS, V3_SOURCES, SourceLabelQuotas
from landuse_sentence_relevance.domain.sampling import BoundedCandidatePool, FinalizedCandidatePool
from landuse_sentence_relevance.domain.seeding import SeedPlan
from landuse_sentence_relevance.domain.stratification import DEFAULT_SOURCES
from landuse_sentence_relevance.domain.v3_annotation import V3AnnotationSeed, select_v3_annotation_seed
from landuse_sentence_relevance.domain.v3_preflight import (
    V3PreflightError,
    V3PreflightReport,
    preflight_v3_candidate_pool,
)
from landuse_sentence_relevance.models.language_identifier import CommonLinguaIdentifier
from landuse_sentence_relevance.models.sentence_splitter import SaTSentenceSplitter
from landuse_sentence_relevance.sources.huggingface import (
    HuggingFaceDatasetLoader,
    HuggingFaceDatasetRows,
    HuggingFaceRowConfig,
)
from landuse_sentence_relevance.sources.remote_files import pinned_remote_file_urls
from landuse_sentence_relevance.sources.v3 import (
    DescriptionSentenceSource,
    V3SourceAdapters,
    WebsiteSentenceSource,
    WikipediaSentenceSource,
)
from landuse_sentence_relevance.sources.website import WebsiteCandidateSource
from landuse_sentence_relevance.sources.wikipedia import WikipediaCandidateSource
from landuse_sentence_relevance.storage.auth import HuggingFaceAuth
from landuse_sentence_relevance.storage.cache import ManagedCache
from landuse_sentence_relevance.storage.candidate_pool import CandidatePoolStore
from landuse_sentence_relevance.storage.candidate_progress import CandidateProgressStore
from landuse_sentence_relevance.storage.publisher import DatasetPublisher
from landuse_sentence_relevance.storage.session import AnnotationStore
from landuse_sentence_relevance.storage.v3_annotation_seed import V3AnnotationSeedStore
from landuse_sentence_relevance.storage.v3_candidate_pool import load_v2_seed_plan
from landuse_sentence_relevance.workflow import AnnotationWorkflow, V3AnnotationWorkflow

logger = logging.getLogger(__name__)
_PROGRESS_CHECKPOINT_INTERVAL = 32
_V3_PROGRESS_CHECKPOINT_INTERVAL = 32

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
_V3_DESCRIPTION_SENTENCE_COLUMNS = (
    "description_identity",
    "source_pbf",
    "osm_type",
    "osm_id",
    "language_code",
    "top_score",
    "sentences",
)
_V3_DESCRIPTION_GEOMETRY_COLUMNS = (
    "source_pbf",
    "osm_type",
    "osm_id",
    "osm_url",
    "name",
    "bbox_min_x",
    "bbox_min_y",
    "bbox_max_x",
    "bbox_max_y",
)
_V3_WIKIPEDIA_SENTENCE_COLUMNS = (
    "sentence_id",
    "document_id",
    "section_id",
    "wikidata",
    "project",
    "language",
    "page_id",
    "section_index",
    "heading",
    "sentence_index",
    "text",
)
_V3_WIKIPEDIA_POLYGON_COLUMNS = (
    "polygon_id",
    "wikidata",
    "has_english_wikipedia",
    "lat",
    "lon",
    "name",
    "region",
)
_V3_WEBSITE_COLUMNS = (
    "polygon_id",
    "lat",
    "lon",
    "name",
    "region",
    "website",
    "contact_website",
    "website_language",
    "website_language_probability",
    "website_sentences",
    "website_sentence_status",
    "contact_website_language",
    "contact_website_language_probability",
    "contact_website_sentences",
    "contact_website_sentence_status",
)


@dataclass(frozen=True, slots=True)
class V3StreamSpec:
    """One immutable upstream V3 stream, with the exact columns production reads.

    Every projection here is asserted against the recorded upstream schema in
    ``tests/fixtures/v3_upstream_schema.json``, so a column the pinned revision
    does not publish cannot reach a stream and fail with ``ArrowInvalid``.
    """

    name: str
    dataset_id: str
    revision: str
    config: str
    split: str
    directory: str
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class V3CandidatePoolResult:
    """The finalized oversized pool and the evidence that it is annotation-ready."""

    pool: FinalizedCandidatePool
    preflight: V3PreflightReport
    seed_plan: SeedPlan


def v3_stream_specs(settings: V3Settings) -> tuple[V3StreamSpec, ...]:
    """Return the five V3 streams a run opens, in a deterministic order."""

    return (
        V3StreamSpec(
            name="description_sentences",
            dataset_id=settings.description_dataset_id,
            revision=settings.description_dataset_revision,
            config=settings.description_sentences_config,
            split=settings.description_sentences_split,
            directory="language-v1/data",
            columns=_V3_DESCRIPTION_SENTENCE_COLUMNS,
        ),
        V3StreamSpec(
            name="description_geometry",
            dataset_id=settings.description_dataset_id,
            revision=settings.description_dataset_revision,
            config=settings.description_geometry_config,
            split=settings.description_geometry_split,
            directory="data",
            columns=_V3_DESCRIPTION_GEOMETRY_COLUMNS,
        ),
        V3StreamSpec(
            name="wikipedia_sentences",
            dataset_id=settings.wikipedia_dataset_id,
            revision=settings.wikipedia_dataset_revision,
            config=settings.wikipedia_sentences_config,
            split=settings.wikipedia_sentences_split,
            directory="wikipedia/sentences",
            columns=_V3_WIKIPEDIA_SENTENCE_COLUMNS,
        ),
        V3StreamSpec(
            name="wikipedia_polygons",
            dataset_id=settings.wikipedia_dataset_id,
            revision=settings.wikipedia_dataset_revision,
            config=settings.wikipedia_polygons_config,
            split=settings.wikipedia_polygons_split,
            directory="polygons",
            columns=_V3_WIKIPEDIA_POLYGON_COLUMNS,
        ),
        V3StreamSpec(
            name="website_polygons",
            dataset_id=settings.website_dataset_id,
            revision=settings.website_dataset_revision,
            config=settings.website_config,
            split=settings.website_split,
            directory="polygons",
            columns=_V3_WEBSITE_COLUMNS,
        ),
    )


def v3_streams_by_dataset(
    settings: V3Settings,
) -> tuple[tuple[str, str, tuple[V3StreamSpec, ...]], ...]:
    """Group the V3 streams by dataset so each catalog is listed once."""

    grouped: dict[tuple[str, str], list[V3StreamSpec]] = {}
    for spec in v3_stream_specs(settings):
        grouped.setdefault((spec.dataset_id, spec.revision), []).append(spec)
    return tuple((dataset_id, revision, tuple(specs)) for (dataset_id, revision), specs in grouped.items())


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


def _prepare_runtime(settings: Settings) -> ManagedCache:
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
    return cache


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


def _h3_geometry(
    settings: Settings | V3Settings,
) -> tuple[Callable[[float, float], str], Callable[[str], tuple[float, float]]]:
    try:
        from h3 import cell_to_latlng, latlng_to_cell
    except ImportError as error:  # pragma: no cover - dependency installation boundary
        raise RuntimeError("Install project dependencies with `uv sync`") from error

    def cell_for_location(latitude: float, longitude: float) -> str:
        return latlng_to_cell(latitude, longitude, settings.h3_resolution)

    def center_of_cell(cell: str) -> tuple[float, float]:
        return tuple(cell_to_latlng(cell))

    return cell_for_location, center_of_cell


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


def v3_remote_files(settings: V3Settings) -> dict[str, tuple[str, ...]]:
    """Select the aligned remote Parquet shards of every V3 stream, by stream name."""

    logger.info(
        "Selecting %d aligned remote V3 Parquet shards per source dataset",
        settings.remote_file_sample_count,
    )
    remote_files: dict[str, tuple[str, ...]] = {}
    for dataset_id, revision, specs in v3_streams_by_dataset(settings):
        remote_files.update(
            pinned_remote_file_urls(
                dataset_id,
                revision,
                {spec.name: spec.directory for spec in specs},
                settings.remote_file_sample_count,
                token=settings.hf_token,
            )
        )
    return remote_files


def v3_row_config(spec: V3StreamSpec, remote_files: tuple[str, ...]) -> HuggingFaceRowConfig:
    """Return the exact streaming configuration production opens for one stream."""

    return HuggingFaceRowConfig(
        dataset_id=spec.dataset_id,
        revision=spec.revision,
        split=spec.split,
        config=spec.config,
        columns=spec.columns,
        remote_files=remote_files,
    )


def _v3_rows(
    spec: V3StreamSpec,
    remote_files: Mapping[str, tuple[str, ...]],
    loader: HuggingFaceDatasetLoader | None,
) -> HuggingFaceDatasetRows:
    return HuggingFaceDatasetRows(v3_row_config(spec, remote_files[spec.name]), loader=loader)


def build_v3_source_adapters(
    settings: V3Settings,
    *,
    cell_for_location: Callable[[float, float], str] | None = None,
    loader: HuggingFaceDatasetLoader | None = None,
) -> V3SourceAdapters:
    """Compose V3 adapters over immutable, streaming-only source revisions."""
    if cell_for_location is None:
        cell_for_location, _ = _h3_geometry(settings)
    remote_files = v3_remote_files(settings)
    rows = {spec.name: _v3_rows(spec, remote_files, loader) for spec in v3_stream_specs(settings)}

    return V3SourceAdapters(
        description=DescriptionSentenceSource(
            sentence_shards_loader=rows["description_sentences"].shards,
            geometry_shards_loader=rows["description_geometry"].shards,
            cell_for_location=cell_for_location,
            max_rows_per_shard=settings.max_rows_per_shard,
            min_language_score=settings.description_min_language_score,
            max_text_characters=settings.max_text_characters,
            max_join_entries=settings.max_join_entries,
            max_stream_workers=settings.stream_workers,
        ),
        wikipedia=WikipediaSentenceSource(
            sentence_shards_loader=rows["wikipedia_sentences"].shards,
            polygon_shards_loader=rows["wikipedia_polygons"].shards,
            cell_for_location=cell_for_location,
            max_rows_per_shard=settings.max_rows_per_shard,
            max_text_characters=settings.max_text_characters,
            max_join_entries=settings.max_join_entries,
            max_stream_workers=settings.stream_workers,
        ),
        website=WebsiteSentenceSource(
            row_shards_loader=rows["website_polygons"].shards,
            cell_for_location=cell_for_location,
            max_rows_per_shard=settings.max_rows_per_shard,
            min_language_probability=settings.website_min_language_probability,
            max_text_characters=settings.max_text_characters,
            max_stream_workers=settings.stream_workers,
        ),
    )


def build_v3_candidate_pool(
    settings: V3Settings,
    *,
    quotas: SourceLabelQuotas = V3_QUOTAS,
    adapters: V3SourceAdapters | None = None,
    cell_for_location: Callable[[float, float], str] | None = None,
    center_of_cell: Callable[[str], tuple[float, float]] | None = None,
    loader: HuggingFaceDatasetLoader | None = None,
) -> V3CandidatePoolResult:
    """Build or reuse the bounded, globally unique V3 candidate reservoir."""

    seed_plan = load_v2_seed_plan(settings.benchmark_path, quotas=quotas, seed=settings.seed)
    metadata = _v3_candidate_pool_metadata(settings, quotas)
    pool_store = CandidatePoolStore(settings.candidate_pool_path)
    cached = pool_store.load(metadata)
    if cached is not None:
        return _validated_v3_result(cached, seed_plan, quotas, settings.candidate_cells_per_source)

    progress_store = CandidateProgressStore(settings.candidate_progress_path)
    pool = _load_v3_candidate_pool(settings, progress_store, metadata, seed_plan.reserved_cells)
    resumed = _try_finalize_v3(pool, settings, seed_plan, quotas, center_of_cell)
    if resumed is not None:
        pool_store.save(resumed.pool, metadata)
        return resumed

    cell_for_location, center_of_cell = _resolve_v3_geometry(settings, cell_for_location, center_of_cell)
    source_adapters = (
        adapters
        if adapters is not None
        else build_v3_source_adapters(
            settings,
            cell_for_location=cell_for_location,
            loader=loader,
        )
    )
    _collect_v3_sources(
        source_adapters,
        pool,
        progress_store,
        metadata,
        seed_plan.reserved_cells,
        settings.candidate_cells_per_source,
    )
    finalized = _finalize_v3_pool(pool, settings, center_of_cell)
    result = _validated_v3_result(finalized, seed_plan, quotas, settings.candidate_cells_per_source)
    pool_store.save(finalized, metadata)
    return result


def build_v3_annotation_seed(
    settings: V3Settings,
    *,
    quotas: SourceLabelQuotas = V3_QUOTAS,
    pool_result: V3CandidatePoolResult | None = None,
    adapters: V3SourceAdapters | None = None,
    cell_for_location: Callable[[float, float], str] | None = None,
    center_of_cell: Callable[[str], tuple[float, float]] | None = None,
    loader: HuggingFaceDatasetLoader | None = None,
) -> V3AnnotationSeed:
    """Build or resume the deterministic, unlabeled V3 annotation state."""

    seed_plan = load_v2_seed_plan(settings.benchmark_path, quotas=quotas, seed=settings.seed)
    if pool_result is None:
        pool_result = build_v3_candidate_pool(
            settings,
            quotas=quotas,
            adapters=adapters,
            cell_for_location=cell_for_location,
            center_of_cell=center_of_cell,
            loader=loader,
        )
    preflight_v3_candidate_pool(
        pool_result.pool,
        seed_plan,
        quotas=quotas,
        candidate_cells_per_source=settings.candidate_cells_per_source,
    )
    benchmark_sha256 = _sha256_file(settings.benchmark_path)
    metadata = _v3_annotation_seed_metadata(settings, quotas, pool_result.pool, benchmark_sha256)
    store = V3AnnotationSeedStore(settings.annotation_seed_path)
    cached = store.load(metadata)
    if cached is not None:
        return cached
    state = select_v3_annotation_seed(
        seed_plan,
        pool_result.pool,
        quotas=quotas,
        seed=settings.seed,
        benchmark_sha256=benchmark_sha256,
    )
    store.save(state, metadata)
    return state


def _resolve_v3_geometry(
    settings: V3Settings,
    cell_for_location: Callable[[float, float], str] | None,
    center_of_cell: Callable[[str], tuple[float, float]] | None,
) -> tuple[Callable[[float, float], str], Callable[[str], tuple[float, float]]]:
    if cell_for_location is not None and center_of_cell is not None:
        return cell_for_location, center_of_cell
    default_cell_for_location, default_center_of_cell = _h3_geometry(settings)
    return cell_for_location or default_cell_for_location, center_of_cell or default_center_of_cell


def _load_v3_candidate_pool(
    settings: V3Settings,
    progress_store: CandidateProgressStore,
    metadata: Mapping[str, Any],
    reserved_cells: frozenset[str],
) -> BoundedCandidatePool:
    pool = BoundedCandidatePool(
        capacity_per_stratum=settings.candidate_capacity_per_stratum,
        seed=settings.seed,
        sources=V3_SOURCES,
    )
    candidates = progress_store.load(metadata)
    if candidates is None:
        return pool
    for candidate in candidates:
        _add_v3_candidate(pool, candidate, reserved_cells)
    logger.info(
        "Resuming V3 candidate progress from %s (%d compact candidates)",
        settings.candidate_progress_path,
        len(candidates),
    )
    return pool


def _collect_v3_sources(
    adapters: V3SourceAdapters,
    pool: BoundedCandidatePool,
    progress_store: CandidateProgressStore,
    metadata: Mapping[str, Any],
    reserved_cells: frozenset[str],
    target_cells_per_source: int,
) -> None:
    """Stream only the sources whose reservoir is still short of the target.

    A source that already read its upstream to the end is not read again: the
    pool keeps a hash-ranked winner per stratum, so a second full pass cannot
    change what it holds. Holding enough cells is deliberately not sufficient.
    Upstream shards arrive in a fixed order, so a source stopped part way
    through covers only the regions it reached, and re-reading it is what gives
    the reservoir its global spread.
    """
    adapter_for_source = {
        Source.WIKIPEDIA: adapters.wikipedia,
        Source.WEBSITE: adapters.website,
        Source.DESCRIPTION: adapters.description,
    }
    available = _v3_available_cells(pool)
    completed = set(progress_store.load_completed_sources(metadata))
    for source in V3_SOURCES:
        if source in completed and available.get(source.value, 0) >= target_cells_per_source:
            logger.info(
                "Skipping the %s stream: it finished earlier and holds %d cells for a %d-cell target",
                source.value,
                available[source.value],
                target_cells_per_source,
            )
            continue
        _collect_v3_source(
            adapter_for_source[source].iter_candidates(),
            source,
            pool,
            progress_store,
            metadata,
            reserved_cells,
            completed,
        )


def _collect_v3_source(
    candidates: Iterable[Candidate],
    source: Source,
    pool: BoundedCandidatePool,
    progress_store: CandidateProgressStore,
    metadata: Mapping[str, Any],
    reserved_cells: frozenset[str],
    completed: set[Source],
) -> int:
    """Stream one source, recording completion only when its rows truly ran out."""

    collected = 0
    try:
        for candidate in candidates:
            if candidate.source is not source:
                raise ValueError(f"V3 {source.value} adapter yielded {candidate.source.value} candidate")
            if candidate.h3_cell in reserved_cells:
                continue
            pool.add(candidate)
            collected += 1
            if collected % _V3_PROGRESS_CHECKPOINT_INTERVAL == 0:
                progress_store.save(pool.snapshot(), metadata, completed)
        completed.add(source)
    finally:
        progress_store.save(pool.snapshot(), metadata, completed)
    return collected


def _add_v3_candidate(
    pool: BoundedCandidatePool,
    candidate: Candidate,
    reserved_cells: frozenset[str],
) -> None:
    if candidate.source not in V3_SOURCES:
        raise ValueError(f"V3 checkpoint contains unexpected {candidate.source.value} candidate")
    if candidate.h3_cell not in reserved_cells:
        pool.add(candidate)


def _try_finalize_v3(
    pool: BoundedCandidatePool,
    settings: V3Settings,
    seed_plan: SeedPlan,
    quotas: SourceLabelQuotas,
    center_of_cell: Callable[[str], tuple[float, float]] | None,
) -> V3CandidatePoolResult | None:
    if center_of_cell is None:
        return None
    try:
        finalized = pool.finalize(
            target_cells_per_source=settings.candidate_cells_per_source,
            center_of_cell=center_of_cell,
            minimum_distance_km=settings.minimum_cell_distance_km,
        )
    except ValueError:
        return None
    return _validated_v3_result(finalized, seed_plan, quotas, settings.candidate_cells_per_source)


def _finalize_v3_pool(
    pool: BoundedCandidatePool,
    settings: V3Settings,
    center_of_cell: Callable[[str], tuple[float, float]],
) -> FinalizedCandidatePool:
    try:
        return pool.finalize(
            target_cells_per_source=settings.candidate_cells_per_source,
            center_of_cell=center_of_cell,
            minimum_distance_km=settings.minimum_cell_distance_km,
        )
    except ValueError as error:
        available = _v3_available_cells(pool)
        raise V3PreflightError(
            f"V3 preflight cannot select {settings.candidate_cells_per_source} disjoint cells per source; "
            f"available={available}"
        ) from error


def _v3_available_cells(pool: BoundedCandidatePool) -> dict[str, int]:
    snapshot = pool.snapshot()
    return {
        source.value: len({candidate.h3_cell for candidate in snapshot if candidate.source is source})
        for source in V3_SOURCES
    }


def _validated_v3_result(
    pool: FinalizedCandidatePool,
    seed_plan: SeedPlan,
    quotas: SourceLabelQuotas,
    candidate_cells_per_source: int,
) -> V3CandidatePoolResult:
    report = preflight_v3_candidate_pool(
        pool,
        seed_plan,
        quotas=quotas,
        candidate_cells_per_source=candidate_cells_per_source,
    )
    return V3CandidatePoolResult(pool=pool, preflight=report, seed_plan=seed_plan)


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


def build_v3_workflow(settings: V3Settings) -> V3AnnotationWorkflow:
    """Build the V3 UI only after the candidate pool and seed preflight pass."""

    logger.info("Starting V3 annotation workflow; validating candidate preflight")
    seed = build_v3_annotation_seed(settings)
    workflow = V3AnnotationWorkflow(seed, AnnotationStore(settings.session_path))
    logger.info(
        "V3 annotation state ready: %d rows (%d fresh) at %s",
        seed.total_rows,
        seed.pending_row_count,
        settings.session_path,
    )
    return workflow


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
            split=settings.output_dataset_split,
        ),
        defer_publish=True,
    )


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


def _v3_candidate_pool_metadata(
    settings: V3Settings,
    quotas: SourceLabelQuotas,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pool_kind": "v3-oversized",
        "seed": settings.seed,
        "h3_resolution": settings.h3_resolution,
        "benchmark": {
            "path": str(settings.benchmark_path),
            "sha256": _sha256_file(settings.benchmark_path),
        },
        "quotas": {
            source.value: {label.value: quotas.required(source, label) for label in quotas.labels}
            for source in quotas.sources
        },
        "streams": [
            {
                "name": spec.name,
                "dataset_id": spec.dataset_id,
                "revision": spec.revision,
                "config": spec.config,
                "split": spec.split,
                "columns": list(spec.columns),
            }
            for spec in v3_stream_specs(settings)
        ],
        "sampling": {
            "candidate_cells_per_source": settings.candidate_cells_per_source,
            "candidate_capacity_per_stratum": settings.candidate_capacity_per_stratum,
            "minimum_cell_distance_km": settings.minimum_cell_distance_km,
            "max_rows_per_shard": settings.max_rows_per_shard,
            "max_join_entries": settings.max_join_entries,
            "description_min_language_score": settings.description_min_language_score,
            "website_min_language_probability": settings.website_min_language_probability,
            "max_text_characters": settings.max_text_characters,
            "remote_file_sample_count": settings.remote_file_sample_count,
        },
    }


def _v3_annotation_seed_metadata(
    settings: V3Settings,
    quotas: SourceLabelQuotas,
    pool: FinalizedCandidatePool,
    benchmark_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state_kind": "v3-annotation-seed",
        "seed": settings.seed,
        "h3_resolution": settings.h3_resolution,
        "benchmark": {"path": str(settings.benchmark_path), "sha256": benchmark_sha256},
        "candidate_pool": {
            "path": str(settings.candidate_pool_path),
            "sha256": _candidate_pool_digest(pool),
            "candidate_count": len(pool.candidates),
        },
        "quotas": {
            source.value: {label.value: quotas.required(source, label) for label in quotas.labels}
            for source in quotas.sources
        },
        "selection": {
            "policy": "sha256-ranked source-by-target-label slots with candidate-id tie-break",
            "pending_rows_unlabeled": True,
            "v2_cells_reserved": True,
        },
    }


def _candidate_pool_digest(pool: FinalizedCandidatePool) -> str:
    payload = json.dumps(
        {
            "candidates": [candidate.to_dict() for candidate in pool.candidates],
            "cells": list(pool.cells),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
