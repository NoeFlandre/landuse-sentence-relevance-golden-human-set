"""Runtime plumbing shared by every workflow build: caches, geometry, digests.

These are the pieces that neither the V2 nor the V3 path owns. They live apart
so that neither module has to import the other for a hashing helper.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from landuse_sentence_relevance.config import Settings, V3Settings
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.storage.auth import HuggingFaceAuth
from landuse_sentence_relevance.storage.cache import ManagedCache
from landuse_sentence_relevance.storage.publisher import DatasetPublisher
from landuse_sentence_relevance.storage.session import AnnotationStore
from landuse_sentence_relevance.workflow import AnnotationWorkflow

logger = logging.getLogger(__name__)


def prepare_runtime(settings: Settings) -> ManagedCache:
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


def h3_geometry(
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


def workflow(
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


def candidate_pool_digest(pool: FinalizedCandidatePool) -> str:
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
