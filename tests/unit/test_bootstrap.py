from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import pytest

from landuse_sentence_relevance.bootstrap import (
    _candidate_cells_to_collect,
    _candidate_pool_metadata,
    _collect_candidates,
    _remaining_candidate_cells,
    _retry_target,
)
from landuse_sentence_relevance.config import Settings
from landuse_sentence_relevance.domain.sampling import BoundedCandidatePool
from landuse_sentence_relevance.storage.candidate_progress import CandidateProgressStore
from tests.unit.test_models import make_candidate


def test_collect_candidates_checkpoints_before_a_source_failure(tmp_path: Path) -> None:
    candidate = make_candidate()

    def failing_source() -> Iterable:
        yield candidate
        raise RuntimeError("source failed")

    store = CandidateProgressStore(tmp_path / "candidate-progress.json")
    pool = BoundedCandidatePool(capacity_per_stratum=1, seed="test")
    metadata = {"schema_version": 2}

    with pytest.raises(RuntimeError, match="source failed"):
        _collect_candidates(failing_source(), pool, store, metadata)

    assert store.load(metadata) == (candidate,)


def test_candidate_metadata_ignores_execution_parallelism() -> None:
    first = _candidate_pool_metadata(Settings.from_env({"SAT_BATCH_SIZE": "1", "SAT_WORKERS": "8"}))
    second = _candidate_pool_metadata(Settings.from_env({"SAT_BATCH_SIZE": "4", "SAT_WORKERS": "16"}))

    assert first == second


def test_remaining_candidate_cells_accounts_for_a_resumable_source() -> None:
    pool = BoundedCandidatePool(capacity_per_stratum=1, seed="test")
    first = make_candidate()
    second = replace(make_candidate("second"), source=first.source, h3_cell="cell-2")
    pool.add(first)
    pool.add(second)

    assert _remaining_candidate_cells(pool, first.source, target_cells=5) == 3
    assert _remaining_candidate_cells(pool, first.source, target_cells=2) == 0


def test_candidate_cells_to_collect_retries_after_a_full_buffer() -> None:
    pool = BoundedCandidatePool(capacity_per_stratum=1, seed="test")
    first = make_candidate()
    second = replace(make_candidate("second"), source=first.source, h3_cell="cell-2")
    pool.add(first)
    pool.add(second)

    assert _candidate_cells_to_collect(pool, first.source, target_cells=5, retry_target=7) == 3
    assert _candidate_cells_to_collect(pool, first.source, target_cells=2, retry_target=7) == 7


def test_retry_target_matches_the_final_source_target() -> None:
    settings = Settings.from_env({"CANDIDATE_POOL_CELLS_PER_SOURCE": "256"})

    assert _retry_target(settings) == 256
