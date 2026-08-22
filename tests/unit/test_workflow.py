from tests.unit.test_constraints import make_annotations

from landuse_sentence_relevance.domain.models import Label
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.storage.publisher import DatasetPublisher
from landuse_sentence_relevance.storage.session import AnnotationStore
from landuse_sentence_relevance.workflow import (
    AnnotationWorkflow,
    UnknownCandidateError,
    WorkflowCompleteError,
)


def test_workflow_records_a_label_and_exposes_progress(tmp_path) -> None:
    rows = make_annotations()[:2]
    workflow = AnnotationWorkflow(
        pool=FinalizedCandidatePool(tuple(row.candidate for row in rows), ("cell-00",)),
        store=AnnotationStore(tmp_path / "annotations.jsonl"),
        publisher=DatasetPublisher("dataset", uploader=lambda **kwargs: None),
    )

    first = workflow.current_candidate()
    assert first is not None
    state = workflow.annotate(first.candidate_id, Label.YES)

    assert state.labeled_count == 1
    assert state.yes_count == 1
    assert state.no_count == 0
    assert state.current_candidate is not None
    assert state.current_candidate.candidate_id != first.candidate_id


def test_workflow_rejects_unknown_or_duplicate_annotations(tmp_path) -> None:
    rows = make_annotations()[:2]
    workflow = AnnotationWorkflow(
        pool=FinalizedCandidatePool(tuple(row.candidate for row in rows), ("cell-00",)),
        store=AnnotationStore(tmp_path / "annotations.jsonl"),
        publisher=DatasetPublisher("dataset", uploader=lambda **kwargs: None),
    )
    first = workflow.current_candidate()
    assert first is not None

    try:
        workflow.annotate("unknown", Label.YES)
    except UnknownCandidateError:
        pass
    else:
        raise AssertionError("unknown candidate should be rejected")

    workflow.annotate(first.candidate_id, Label.YES)
    try:
        workflow.annotate(first.candidate_id, Label.NO)
    except UnknownCandidateError:
        pass
    else:
        raise AssertionError("duplicate candidate should be rejected")


def test_workflow_publishes_automatically_when_the_final_contract_is_met(tmp_path) -> None:
    rows = make_annotations()
    calls = []
    workflow = AnnotationWorkflow(
        pool=FinalizedCandidatePool(
            tuple(row.candidate for row in rows),
            tuple(f"cell-{i:02d}" for i in range(25)),
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
    try:
        workflow.annotate(rows[0].candidate.candidate_id, rows[0].label)
    except WorkflowCompleteError:
        pass
    else:
        raise AssertionError("a completed workflow must reject additional labels")
