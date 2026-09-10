import logging
from threading import Event, Thread
from typing import cast

import pytest

import landuse_sentence_relevance.workflow as workflow_module
from landuse_sentence_relevance.domain.models import Label
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.storage.publisher import DatasetPublicationError, DatasetPublisher
from landuse_sentence_relevance.storage.session import AnnotationStore
from landuse_sentence_relevance.workflow import (
    AnnotationWorkflow,
    UnknownCandidateError,
    WorkflowCompleteError,
)
from tests.unit.test_constraints import make_annotations


class RecordingPublisher:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def publish_if_ready(self, annotations) -> bool:
        self.calls.append(tuple(annotations))
        return False


def test_workflow_records_a_label_and_exposes_progress(tmp_path, caplog) -> None:
    rows = make_annotations()[:2]
    workflow = AnnotationWorkflow(
        pool=FinalizedCandidatePool(
            tuple(row.candidate for row in rows),
            tuple(row.candidate.h3_cell for row in rows),
        ),
        store=AnnotationStore(tmp_path / "annotations.jsonl"),
        publisher=DatasetPublisher("dataset", uploader=lambda **kwargs: None),
    )

    caplog.set_level(logging.INFO)
    first = workflow.current_candidate()
    assert first is not None
    state = workflow.annotate(first.candidate_id, Label.YES)

    assert state.labeled_count == 1
    assert state.yes_count == 1
    assert state.no_count == 0
    assert state.current_candidate is not None
    assert state.current_candidate.candidate_id != first.candidate_id
    assert "Annotation saved: 1 labeled (Yes=1, No=0)" in caplog.text


def test_workflow_changes_a_saved_label_and_updates_metrics(tmp_path) -> None:
    rows = make_annotations()[:2]
    workflow = AnnotationWorkflow(
        pool=FinalizedCandidatePool(
            tuple(row.candidate for row in rows),
            tuple(row.candidate.h3_cell for row in rows),
        ),
        store=AnnotationStore(tmp_path / "annotations.jsonl"),
        publisher=DatasetPublisher("dataset", uploader=lambda **kwargs: None),
    )

    workflow.annotate(rows[0].candidate.candidate_id, Label.YES)
    state = workflow.change_label(rows[0].candidate.candidate_id, Label.NO)

    assert state.labeled_count == 1
    assert state.yes_count == 0
    assert state.no_count == 1
    assert state.annotations[0].label is Label.NO


def test_state_checks_balance_once_while_annotations_are_incomplete(tmp_path, monkeypatch) -> None:
    rows = make_annotations()[:2]
    workflow = AnnotationWorkflow(
        pool=FinalizedCandidatePool(
            tuple(row.candidate for row in rows),
            tuple(row.candidate.h3_cell for row in rows),
        ),
        store=AnnotationStore(tmp_path / "annotations.jsonl"),
        publisher=DatasetPublisher("dataset", uploader=lambda **kwargs: None),
    )
    workflow.annotate(rows[0].candidate.candidate_id, Label.YES)

    selection_calls = 0
    select = workflow_module.select_final_annotations

    def observe_selection(annotations):
        nonlocal selection_calls
        selection_calls += 1
        return select(annotations)

    monkeypatch.setattr(workflow_module, "select_final_annotations", observe_selection)

    state = workflow.state()

    assert state.current_candidate == rows[1].candidate
    assert (state.labeled_count, state.yes_count, state.no_count) == (1, 1, 0)
    assert state.final_ready is False
    assert selection_calls == 1


def test_deferred_workflow_persists_before_publication(tmp_path) -> None:
    rows = make_annotations()[:2]
    store = AnnotationStore(tmp_path / "annotations.jsonl")
    publisher = RecordingPublisher()
    workflow = AnnotationWorkflow(
        pool=FinalizedCandidatePool(
            tuple(row.candidate for row in rows),
            tuple(row.candidate.h3_cell for row in rows),
        ),
        store=store,
        publisher=cast(DatasetPublisher, publisher),
        defer_publish=True,
    )
    publisher.calls.clear()

    workflow.annotate(rows[0].candidate.candidate_id, Label.YES)

    assert publisher.calls == []
    assert store.load()[rows[0].candidate.candidate_id].label is Label.YES


def test_deferred_publisher_failure_keeps_the_local_edit(tmp_path) -> None:
    rows = make_annotations()[:2]
    store = AnnotationStore(tmp_path / "annotations.jsonl")

    class FailingPublisher:
        def publish_if_ready(self, annotations) -> bool:
            raise DatasetPublicationError("upload unavailable")

    workflow = AnnotationWorkflow(
        pool=FinalizedCandidatePool(
            tuple(row.candidate for row in rows),
            tuple(row.candidate.h3_cell for row in rows),
        ),
        store=store,
        publisher=cast(DatasetPublisher, FailingPublisher()),
        defer_publish=True,
    )

    workflow.annotate(rows[0].candidate.candidate_id, Label.YES)
    assert workflow.publish_if_ready() is False
    assert store.load()[rows[0].candidate.candidate_id].label is Label.YES


def test_scheduled_publication_returns_before_upload_finishes(tmp_path) -> None:
    rows = make_annotations()[:2]
    upload_started = Event()
    upload_finished = Event()
    release_upload = Event()

    class BlockingPublisher:
        calls = 0

        def publish_if_ready(self, annotations) -> bool:
            self.calls += 1
            if self.calls == 1:
                return False
            upload_started.set()
            release_upload.wait(timeout=2)
            upload_finished.set()
            return False

    publisher = BlockingPublisher()
    workflow = AnnotationWorkflow(
        pool=FinalizedCandidatePool(
            tuple(row.candidate for row in rows),
            tuple(row.candidate.h3_cell for row in rows),
        ),
        store=AnnotationStore(tmp_path / "annotations.jsonl"),
        publisher=cast(DatasetPublisher, publisher),
        defer_publish=True,
    )

    workflow.schedule_publish()

    assert upload_started.wait(timeout=2)
    assert not upload_finished.is_set()
    release_upload.set()
    assert upload_finished.wait(timeout=2)


def test_workflow_close_waits_for_scheduled_publication_and_rejects_new_work(tmp_path) -> None:
    rows = make_annotations()[:2]
    upload_started = Event()
    upload_finished = Event()
    release_upload = Event()

    class BlockingPublisher:
        calls = 0

        def publish_if_ready(self, annotations) -> bool:
            self.calls += 1
            if self.calls == 1:
                return False
            upload_started.set()
            release_upload.wait(timeout=2)
            upload_finished.set()
            return False

    workflow = AnnotationWorkflow(
        pool=FinalizedCandidatePool(
            tuple(row.candidate for row in rows),
            tuple(row.candidate.h3_cell for row in rows),
        ),
        store=AnnotationStore(tmp_path / "annotations.jsonl"),
        publisher=cast(DatasetPublisher, BlockingPublisher()),
        defer_publish=True,
    )
    workflow.schedule_publish()

    assert upload_started.wait(timeout=2)
    close_finished = Event()

    def close_workflow() -> None:
        workflow.close()
        close_finished.set()

    closer = Thread(target=close_workflow)
    closer.start()
    assert not close_finished.wait(timeout=0.1)
    release_upload.set()
    assert close_finished.wait(timeout=2)
    closer.join(timeout=2)

    assert upload_finished.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        workflow.schedule_publish()


def test_workflow_removes_a_saved_annotation_and_updates_metrics(tmp_path) -> None:
    rows = make_annotations()[:2]
    workflow = AnnotationWorkflow(
        pool=FinalizedCandidatePool(
            tuple(row.candidate for row in rows),
            tuple(row.candidate.h3_cell for row in rows),
        ),
        store=AnnotationStore(tmp_path / "annotations.jsonl"),
        publisher=DatasetPublisher("dataset", uploader=lambda **kwargs: None),
    )

    workflow.annotate(rows[0].candidate.candidate_id, Label.YES)
    state = workflow.remove_annotation(rows[0].candidate.candidate_id)

    assert state.labeled_count == 0
    assert state.yes_count == 0
    assert state.no_count == 0
    assert state.annotations == ()


def test_workflow_keeps_review_edits_available_after_upload(tmp_path) -> None:
    rows = make_annotations()
    calls = []
    workflow = AnnotationWorkflow(
        pool=FinalizedCandidatePool(
            tuple(row.candidate for row in rows),
            tuple(row.candidate.h3_cell for row in rows),
        ),
        store=AnnotationStore(tmp_path / "annotations.jsonl"),
        publisher=DatasetPublisher("dataset", uploader=lambda **kwargs: calls.append(kwargs)),
    )

    for row in rows:
        workflow.annotate(row.candidate.candidate_id, row.label)

    assert workflow.state().published is True
    changed = workflow.change_label(rows[0].candidate.candidate_id, Label.NO)

    assert changed.yes_count == 49
    assert changed.no_count == 51
    assert changed.published is False
    assert len(calls) == 1

    repaired = workflow.change_label(rows[50].candidate.candidate_id, Label.YES)

    assert repaired.yes_count == 50
    assert repaired.no_count == 50
    assert repaired.published is True
    assert len(calls) == 2


def test_workflow_removes_a_published_annotation_and_persists_the_edit(tmp_path) -> None:
    rows = make_annotations()
    calls = []
    store = AnnotationStore(tmp_path / "annotations.jsonl")
    workflow = AnnotationWorkflow(
        pool=FinalizedCandidatePool(
            tuple(row.candidate for row in rows),
            tuple(row.candidate.h3_cell for row in rows),
        ),
        store=store,
        publisher=DatasetPublisher("dataset", uploader=lambda **kwargs: calls.append(kwargs)),
    )

    for row in rows:
        workflow.annotate(row.candidate.candidate_id, row.label)

    state = workflow.remove_annotation(rows[0].candidate.candidate_id)

    assert state.labeled_count == 99
    assert state.published is False
    assert len(calls) == 1
    assert rows[0].candidate.candidate_id not in store.load()


def test_workflow_rejects_unknown_or_duplicate_annotations(tmp_path) -> None:
    rows = make_annotations()[:2]
    workflow = AnnotationWorkflow(
        pool=FinalizedCandidatePool(
            tuple(row.candidate for row in rows),
            tuple(row.candidate.h3_cell for row in rows),
        ),
        store=AnnotationStore(tmp_path / "annotations.jsonl"),
        publisher=DatasetPublisher("dataset", uploader=lambda **kwargs: None),
    )
    first = workflow.current_candidate()
    assert first is not None

    with pytest.raises(UnknownCandidateError):
        workflow.annotate("unknown", Label.YES)

    workflow.annotate(first.candidate_id, Label.YES)
    with pytest.raises(UnknownCandidateError):
        workflow.annotate(first.candidate_id, Label.NO)


def test_workflow_publishes_automatically_when_the_final_contract_is_met(tmp_path) -> None:
    rows = make_annotations()
    calls = []
    workflow = AnnotationWorkflow(
        pool=FinalizedCandidatePool(
            tuple(row.candidate for row in rows),
            tuple(row.candidate.h3_cell for row in rows),
        ),
        store=AnnotationStore(tmp_path / "annotations.jsonl"),
        publisher=DatasetPublisher("dataset", uploader=lambda **kwargs: calls.append(kwargs)),
    )

    for row in rows:
        workflow.annotate(row.candidate.candidate_id, row.label)

    state = workflow.state()
    assert state.published is True
    assert state.final_ready is True
    assert len(calls) == 1
    with pytest.raises(WorkflowCompleteError):
        workflow.annotate(rows[0].candidate.candidate_id, rows[0].label)
