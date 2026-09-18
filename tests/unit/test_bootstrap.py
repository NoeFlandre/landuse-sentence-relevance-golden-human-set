from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import NoReturn, cast

import pytest

from landuse_sentence_relevance.bootstrap.v2 import (
    _candidate_cells_to_collect,
    _candidate_pool_metadata,
    _collect_candidates,
    _load_candidate_pool,
    _remaining_candidate_cells,
    _retry_target,
    _try_finalize,
    build_workflow,
)
from landuse_sentence_relevance.config import Settings
from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.domain.sampling import BoundedCandidatePool, FinalizedCandidatePool
from landuse_sentence_relevance.storage.candidate_pool import CandidatePoolStore
from landuse_sentence_relevance.storage.candidate_progress import CandidateProgressStore
from tests.builders import make_annotations, make_candidate


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


def test_try_finalize_waits_until_both_sources_have_enough_cells() -> None:
    pool = BoundedCandidatePool(capacity_per_stratum=1, seed="test")
    candidate = make_candidate()
    pool.add(candidate)

    assert _try_finalize(pool, 2, lambda _: (0.0, 0.0), 0.0) is None


def test_try_finalize_returns_none_when_distance_constraints_are_not_reachable() -> None:
    pool = BoundedCandidatePool(capacity_per_stratum=1, seed="test")
    for index in (0, 1, 50, 51):
        pool.add(make_annotations()[index].candidate)

    assert _try_finalize(pool, 2, lambda _: (0.0, 0.0), 1.0) is None


def test_load_candidate_pool_replays_saved_candidates() -> None:
    candidate = make_candidate()

    class SavedProgress:
        def load(self, metadata):
            return (candidate,)

    pool = _load_candidate_pool(
        Settings.from_env({}),
        cast(CandidateProgressStore, SavedProgress()),
        {},
    )

    assert pool.snapshot() == (candidate,)


def test_build_workflow_reuses_a_persisted_pool(tmp_path) -> None:
    settings = Settings.from_env(
        {
            "CANDIDATE_POOL_PATH": str(tmp_path / "candidate-pool.json"),
            "CANDIDATE_PROGRESS_PATH": str(tmp_path / "candidate-progress.json"),
            "SESSION_PATH": str(tmp_path / "annotations.jsonl"),
            "MODEL_CACHE_DIR": str(tmp_path / "runtime-cache"),
            "HF_AUTH_DIR": str(tmp_path / "huggingface-auth"),
        }
    )
    annotations = make_annotations()
    pool = FinalizedCandidatePool(
        tuple(annotation.candidate for annotation in annotations),
        tuple(annotation.candidate.h3_cell for annotation in annotations),
    )
    metadata = _candidate_pool_metadata(settings)
    CandidatePoolStore(settings.candidate_pool_path).save(pool, metadata)

    workflow = build_workflow(settings)
    try:
        assert workflow.current_candidate() is not None
    finally:
        workflow.close()


def test_build_workflow_collects_and_saves_a_new_pool(monkeypatch, tmp_path) -> None:
    import h3

    import landuse_sentence_relevance.bootstrap.runtime as runtime
    import landuse_sentence_relevance.bootstrap.v2 as bootstrap

    settings = replace(
        Settings.from_env(
            {
                "CANDIDATE_POOL_PATH": str(tmp_path / "candidate-pool.json"),
                "CANDIDATE_PROGRESS_PATH": str(tmp_path / "candidate-progress.json"),
                "SESSION_PATH": str(tmp_path / "annotations.jsonl"),
                "MODEL_CACHE_DIR": str(tmp_path / "runtime-cache"),
                "HF_AUTH_DIR": str(tmp_path / "huggingface-auth"),
            }
        ),
        candidate_cell_count=2,
        minimum_candidate_cells=1,
        candidate_pool_cells_per_source=1,
        minimum_cell_distance_km=0.0,
    )
    wikipedia_cell = h3.latlng_to_cell(45.0, 2.0, settings.h3_resolution)
    website_cell = h3.latlng_to_cell(-30.0, 140.0, settings.h3_resolution)
    wikipedia_candidate = replace(
        make_candidate("wikipedia"),
        h3_cell=wikipedia_cell,
        latitude=45.0,
        longitude=2.0,
    )
    website_candidate = replace(
        make_candidate("website"),
        source=Source.WEBSITE,
        source_field="website_text",
        h3_cell=website_cell,
        latitude=-30.0,
        longitude=140.0,
    )

    class Auth:
        def __init__(self, root):
            pass

        def prepare(self, **kwargs):
            return True

    class EmptyRows:
        def __init__(self, config):
            self.config = config

        def __call__(self):
            return iter(())

        def shards(self):
            return iter(())

    class WikipediaSource:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def iter_candidates(self):
            tuple(self.kwargs["row_loader"]("polygons"))
            tuple(self.kwargs["row_shards_loader"]("polygons"))
            return iter((wikipedia_candidate,))

    class WebsiteSource:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def iter_candidates(self):
            tuple(self.kwargs["row_loader"]())
            tuple(self.kwargs["row_shards_loader"]())
            return iter((website_candidate,))

    monkeypatch.setattr(runtime, "HuggingFaceAuth", Auth)
    monkeypatch.setattr(bootstrap, "HuggingFaceDatasetRows", EmptyRows)
    monkeypatch.setattr(bootstrap, "WikipediaCandidateSource", WikipediaSource)
    monkeypatch.setattr(bootstrap, "WebsiteCandidateSource", WebsiteSource)
    monkeypatch.setattr(
        bootstrap,
        "SaTSentenceSplitter",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        bootstrap,
        "CommonLinguaIdentifier",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        bootstrap,
        "pinned_remote_file_urls",
        lambda *args, **kwargs: {key: (f"{key}.parquet",) for key in args[2]},
    )

    workflow = build_workflow(settings)
    try:
        saved = CandidatePoolStore(settings.candidate_pool_path).load(_candidate_pool_metadata(settings))
        assert saved is not None
        assert {candidate.source for candidate in saved.candidates} == {Source.WIKIPEDIA, Source.WEBSITE}
    finally:
        workflow.close()


def test_build_workflow_finalizes_resumable_progress_before_streaming(monkeypatch, tmp_path) -> None:
    import h3

    import landuse_sentence_relevance.bootstrap.v2 as bootstrap

    settings = replace(
        Settings.from_env(
            {
                "CANDIDATE_POOL_PATH": str(tmp_path / "candidate-pool.json"),
                "CANDIDATE_PROGRESS_PATH": str(tmp_path / "candidate-progress.json"),
                "SESSION_PATH": str(tmp_path / "annotations.jsonl"),
                "MODEL_CACHE_DIR": str(tmp_path / "runtime-cache"),
                "HF_AUTH_DIR": str(tmp_path / "huggingface-auth"),
            }
        ),
        candidate_cell_count=2,
        minimum_candidate_cells=1,
        candidate_pool_cells_per_source=1,
        minimum_cell_distance_km=0.0,
    )
    wikipedia = replace(
        make_candidate("saved-wikipedia"),
        h3_cell=h3.latlng_to_cell(45.0, 2.0, settings.h3_resolution),
    )
    website = replace(
        make_candidate("saved-website"),
        source=Source.WEBSITE,
        source_field="website_text",
        h3_cell=h3.latlng_to_cell(-30.0, 140.0, settings.h3_resolution),
    )
    metadata = _candidate_pool_metadata(settings)
    CandidateProgressStore(settings.candidate_progress_path).save((wikipedia, website), metadata)

    def remote_stream_not_expected(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("remote streaming should not start")

    monkeypatch.setattr(
        bootstrap,
        "pinned_remote_file_urls",
        remote_stream_not_expected,
    )

    workflow = build_workflow(settings)
    try:
        saved = CandidatePoolStore(settings.candidate_pool_path).load(metadata)
        assert saved is not None
        assert {candidate.source for candidate in saved.candidates} == {Source.WIKIPEDIA, Source.WEBSITE}
    finally:
        workflow.close()
