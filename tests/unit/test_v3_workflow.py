from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from landuse_sentence_relevance.domain.models import Annotation, Candidate, Label, Source
from landuse_sentence_relevance.domain.profile import V3_SOURCES, balanced_quotas
from landuse_sentence_relevance.domain.v3_annotation import (
    V3AnnotationSeed,
    V3SeedRow,
    V3SelectionMetadata,
)
from landuse_sentence_relevance.domain.v3_progress import V3AnnotationProgressError
from landuse_sentence_relevance.storage.session import AnnotationStore
from landuse_sentence_relevance.workflow import (
    UnknownAnnotationError,
    UnknownCandidateError,
    V3AnnotationWorkflow,
)
from tests.unit.test_models import make_candidate


def _candidate(candidate_id: str, source: Source, cell: str) -> Candidate:
    return replace(
        make_candidate(candidate_id),
        sentence=f"Sentence for {candidate_id}.",
        source=source,
        h3_cell=cell,
    )


def _seed() -> V3AnnotationSeed:
    quotas = balanced_quotas(V3_SOURCES, rows_per_source_label=1)
    rows: list[V3SeedRow] = []
    for index, (source, label) in enumerate(quotas.counts):
        candidate = _candidate(
            f"{source.value}-{label.value}",
            source,
            f"{source.value}-{label.value}-cell",
        )
        annotation = Annotation(candidate, label) if index == 0 else None
        rows.append(
            V3SeedRow(
                candidate=candidate,
                quota_source=source,
                quota_label=label,
                origin="v2" if annotation is not None else "v3",
                annotation=annotation,
                selection=V3SelectionMetadata(seed="workflow-test", rank="0" * 64, slot_index=index),
            )
        )
    return V3AnnotationSeed(
        rows=tuple(rows),
        excluded_v2_rows=(),
        reserved_v2_cells=frozenset({"wikipedia-yes-cell"}),
        quotas=quotas,
        benchmark_sha256="0" * 64,
        seed="workflow-test",
    )


def _workflow(tmp_path: Path) -> tuple[V3AnnotationWorkflow, V3AnnotationSeed, AnnotationStore]:
    seed = _seed()
    store = AnnotationStore(tmp_path / "results" / "annotations" / "sessions" / "v3.jsonl")
    return V3AnnotationWorkflow(seed, store), seed, store


def test_v3_workflow_counts_seeded_rows_but_only_offers_fresh_rows(tmp_path: Path) -> None:
    workflow, seed, _ = _workflow(tmp_path)

    state = workflow.state()

    assert state.workflow_version == "v3"
    assert state.total_count == 6
    assert state.seeded_count == 1
    assert state.fresh_labeled_count == 0
    assert state.labeled_count == 1
    assert state.yes_count == 1
    assert state.no_count == 0
    assert state.current_candidate == seed.rows[1].candidate
    assert state.current_candidate != seed.rows[0].candidate
    assert state.annotations == ()


def test_v3_workflow_persists_and_resumes_the_fresh_session(tmp_path: Path) -> None:
    workflow, seed, store = _workflow(tmp_path)
    first = seed.rows[1].candidate

    workflow.annotate(first.candidate_id, Label.NO)

    assert store.load() == {first.candidate_id: Annotation(first, Label.NO)}
    workflow.close()
    resumed = V3AnnotationWorkflow(seed, store)
    try:
        state = resumed.state()
        assert state.current_candidate == seed.rows[2].candidate
        assert state.fresh_labeled_count == 1
        assert state.labeled_count == 2
        assert state.yes_count == 1
        assert state.no_count == 1
    finally:
        resumed.close()


def test_v3_workflow_relabels_and_removes_with_atomic_session_replacement(tmp_path: Path) -> None:
    workflow, seed, store = _workflow(tmp_path)
    first = seed.rows[1].candidate

    workflow.annotate(first.candidate_id, Label.NO)
    workflow.change_label(first.candidate_id, Label.YES)
    assert store.load()[first.candidate_id].label is Label.YES

    state = workflow.remove_annotation(first.candidate_id)

    assert store.load() == {}
    assert state.current_candidate == first
    assert state.fresh_labeled_count == 0


def test_v3_workflow_never_allows_a_seeded_v2_row_to_be_edited(tmp_path: Path) -> None:
    workflow, seed, _ = _workflow(tmp_path)

    with pytest.raises(UnknownCandidateError):
        workflow.annotate(seed.rows[0].candidate.candidate_id, Label.NO)
    with pytest.raises(UnknownAnnotationError, match="seeded V2"):
        workflow.change_label(seed.rows[0].candidate.candidate_id, Label.NO)
    with pytest.raises(UnknownAnnotationError, match="seeded V2"):
        workflow.remove_annotation(seed.rows[0].candidate.candidate_id)


def test_v3_workflow_rejects_a_session_that_contains_a_seeded_row(tmp_path: Path) -> None:
    seed = _seed()
    store = AnnotationStore(tmp_path / "v3.jsonl")
    seeded = seed.seeded_annotations[0]
    store.save((seeded,))

    with pytest.raises(V3AnnotationProgressError, match="seeded V2"):
        V3AnnotationWorkflow(seed, store)


def test_v3_workflow_keeps_memory_unchanged_when_atomic_save_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, seed, store = _workflow(tmp_path)
    first = seed.rows[1].candidate

    def fail_save(annotations: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "save", fail_save)

    with pytest.raises(OSError, match="disk full"):
        workflow.annotate(first.candidate_id, Label.YES)

    assert workflow.state().current_candidate == first


def test_v3_workflow_finishes_locally_without_publishing(tmp_path: Path) -> None:
    workflow, seed, _ = _workflow(tmp_path)

    for row in seed.pending_rows:
        workflow.annotate(row.candidate.candidate_id, row.quota_label)

    state = workflow.state()

    assert state.current_candidate is None
    assert state.final_ready is True
    assert state.published is False
    workflow.schedule_publish()
    assert workflow.state().published is False
