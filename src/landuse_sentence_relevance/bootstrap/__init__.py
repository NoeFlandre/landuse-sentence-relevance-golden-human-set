"""Compose the runtime graph: settings in, a ready annotation workflow out.

The package keeps the two generations apart. :mod:`streams` declares the five
immutable V3 upstream streams, :mod:`v3` builds the V3 pool, seed and workflow
over them, :mod:`v2` builds the older two-dataset path, and :mod:`runtime`
holds what both need - the disposable cache, H3 geometry and digests.

Callers import from this package, not from the submodules.
"""

from __future__ import annotations

from landuse_sentence_relevance.bootstrap.streams import (
    V3StreamSpec,
    v3_remote_files,
    v3_row_config,
    v3_stream_specs,
    v3_streams_by_dataset,
)
from landuse_sentence_relevance.bootstrap.v2 import build_workflow
from landuse_sentence_relevance.bootstrap.v3 import (
    V3CandidatePoolResult,
    build_v3_annotation_seed,
    build_v3_candidate_pool,
    build_v3_source_adapters,
    build_v3_workflow,
)

__all__ = [
    "V3CandidatePoolResult",
    "V3StreamSpec",
    "build_v3_annotation_seed",
    "build_v3_candidate_pool",
    "build_v3_source_adapters",
    "build_v3_workflow",
    "build_workflow",
    "v3_remote_files",
    "v3_row_config",
    "v3_stream_specs",
    "v3_streams_by_dataset",
]
