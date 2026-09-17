"""Build the V3 candidate pool, annotation seed and workflow.

The V3 path is resumable end to end: the pool, the progress checkpoint and the
annotation seed are each keyed by a metadata fingerprint, so a changed pin
invalidates the record rather than silently reusing different upstream data.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from landuse_sentence_relevance.bootstrap.runtime import (
    _candidate_pool_digest,
    _h3_geometry,
    _sha256_file,
)
from landuse_sentence_relevance.bootstrap.streams import (
    _v3_rows,
    v3_remote_files,
    v3_stream_specs,
)
from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.profile import V3_QUOTAS, V3_SOURCES, SourceLabelQuotas
from landuse_sentence_relevance.domain.sampling import BoundedCandidatePool, FinalizedCandidatePool
from landuse_sentence_relevance.domain.seeding import SeedPlan
from landuse_sentence_relevance.domain.v3_annotation import V3AnnotationSeed, select_v3_annotation_seed
from landuse_sentence_relevance.domain.v3_preflight import (
    V3PreflightError,
    V3PreflightReport,
    preflight_v3_candidate_pool,
)
from landuse_sentence_relevance.sources.huggingface import HuggingFaceDatasetLoader
from landuse_sentence_relevance.sources.v3 import (
    DescriptionSentenceSource,
    V3SourceAdapters,
    WebsiteSentenceSource,
    WikipediaSentenceSource,
)
from landuse_sentence_relevance.storage.candidate_pool import CandidatePoolStore
from landuse_sentence_relevance.storage.candidate_progress import CandidateProgressStore
from landuse_sentence_relevance.storage.session import AnnotationStore
from landuse_sentence_relevance.storage.v3_annotation_seed import V3AnnotationSeedStore
from landuse_sentence_relevance.storage.v3_candidate_pool import load_v2_seed_plan
from landuse_sentence_relevance.workflow import V3AnnotationWorkflow

logger = logging.getLogger(__name__)
_V3_PROGRESS_CHECKPOINT_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class V3CandidatePoolResult:
    """The finalized oversized pool and the evidence that it is annotation-ready."""

    pool: FinalizedCandidatePool
    preflight: V3PreflightReport
    seed_plan: SeedPlan


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
    now: Callable[[], float] = time.monotonic,
) -> int:
    """Stream one source, recording completion only when its rows truly ran out.

    Checkpoints are paced by elapsed time rather than by a candidate count. Each
    save sorts and re-serialises the whole pool, so counting candidates ties the
    cost to throughput and throttles the stream to roughly the rate at which the
    pool can be rewritten. Saving often also gains nothing, because a resumed
    source re-reads its shards from the beginning: the checkpoint has only to
    preserve the candidate set, never a position in the stream.
    """

    collected = 0
    last_saved = now()
    try:
        for candidate in candidates:
            if candidate.source is not source:
                raise ValueError(f"V3 {source.value} adapter yielded {candidate.source.value} candidate")
            if candidate.h3_cell in reserved_cells:
                continue
            pool.add(candidate)
            collected += 1
            current = now()
            if current - last_saved >= _V3_PROGRESS_CHECKPOINT_SECONDS:
                progress_store.save(pool.snapshot(), metadata, completed)
                last_saved = current
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
