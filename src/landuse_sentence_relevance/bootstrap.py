from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from landuse_sentence_relevance.config import Settings
from landuse_sentence_relevance.domain.sampling import BoundedCandidatePool
from landuse_sentence_relevance.models.language_identifier import CommonLinguaIdentifier
from landuse_sentence_relevance.models.sentence_splitter import SaTSentenceSplitter
from landuse_sentence_relevance.sources.huggingface import HuggingFaceDatasetRows, HuggingFaceRowConfig
from landuse_sentence_relevance.sources.website import WebsiteCandidateSource
from landuse_sentence_relevance.sources.wikipedia import WikipediaCandidateSource
from landuse_sentence_relevance.storage.cache import ManagedCache
from landuse_sentence_relevance.storage.publisher import DatasetPublisher
from landuse_sentence_relevance.storage.session import AnnotationStore
from landuse_sentence_relevance.workflow import AnnotationWorkflow


def build_workflow(  # pragma: no cover - full startup needs remote datasets and model weights
    settings: Settings,
) -> AnnotationWorkflow:
    cache = ManagedCache(settings.model_cache_dir)
    cache.prepare()
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
    splitter = SaTSentenceSplitter(
        model_id=settings.sat_model_id,
        revision=settings.sat_model_revision,
        tokenizer_id=settings.sat_tokenizer_id,
        tokenizer_revision=settings.sat_tokenizer_revision,
        cache_dir=settings.model_cache_dir,
    )
    language_identifier = CommonLinguaIdentifier(
        model_id=settings.language_model_id,
        revision=settings.language_model_revision,
        min_confidence=settings.website_language_min_confidence,
        cache_dir=settings.model_cache_dir,
    )
    pool = BoundedCandidatePool(
        capacity_per_stratum=settings.candidate_capacity_per_stratum,
        seed=settings.seed,
    )

    wikipedia_source = WikipediaCandidateSource(
        row_loader=wikipedia_rows,
        splitter=splitter,
        cell_for_location=cell_for_location,
        max_polygons_per_cell=settings.max_polygons_per_cell,
        seed=settings.seed,
    )
    for candidate in wikipedia_source.iter_candidates():
        pool.add(candidate)

    website_source = WebsiteCandidateSource(
        row_loader=website_rows,
        splitter=splitter,
        language_identifier=language_identifier,
        cell_for_location=cell_for_location,
    )
    for candidate in website_source.iter_candidates():
        pool.add(candidate)

    finalized_pool = pool.finalize(
        target_cell_count=settings.target_cell_count,
        center_of_cell=center_of_cell,
    )
    return AnnotationWorkflow(
        pool=finalized_pool,
        store=AnnotationStore(settings.session_path),
        publisher=DatasetPublisher(
            settings.output_dataset_id,
            token=settings.hf_token,
            cleanup=cache.cleanup,
        ),
    )
