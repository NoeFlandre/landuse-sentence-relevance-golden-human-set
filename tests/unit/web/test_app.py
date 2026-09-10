from dataclasses import dataclass, field

from fastapi.testclient import TestClient
from tests.unit.test_models import make_candidate

from landuse_sentence_relevance.domain.models import Annotation, Candidate, Label
from landuse_sentence_relevance.web.app import create_app
from landuse_sentence_relevance.workflow import UnknownAnnotationError, WorkflowState


@dataclass
class FakeWorkflow:
    candidate: Candidate
    calls: list[tuple[str, Label]]
    annotations: list[Annotation] = field(default_factory=list)
    schedule_calls: int = 0
    close_calls: int = 0

    def current_candidate(self) -> Candidate:
        return self.candidate

    def state(self) -> WorkflowState:
        return WorkflowState(
            current_candidate=self.candidate,
            labeled_count=len(self.annotations),
            yes_count=sum(annotation.label is Label.YES for annotation in self.annotations),
            no_count=sum(annotation.label is Label.NO for annotation in self.annotations),
            final_ready=False,
            published=False,
            annotations=tuple(self.annotations),
        )

    def annotate(self, candidate_id: str, label: Label) -> WorkflowState:
        if candidate_id != self.candidate.candidate_id:
            raise ValueError("unknown candidate")
        self.calls.append((candidate_id, label))
        self.annotations.append(Annotation(candidate=self.candidate, label=label))
        return self.state()

    def change_label(self, candidate_id: str, label: Label) -> WorkflowState:
        for index, annotation in enumerate(self.annotations):
            if annotation.candidate.candidate_id == candidate_id:
                self.annotations[index] = Annotation(annotation.candidate, label)
                return self.state()
        raise ValueError("unknown annotation")

    def remove_annotation(self, candidate_id: str) -> WorkflowState:
        remaining = [
            annotation for annotation in self.annotations if annotation.candidate.candidate_id != candidate_id
        ]
        if len(remaining) == len(self.annotations):
            raise UnknownAnnotationError("the annotation is unknown")
        self.annotations = remaining
        return self.state()

    def schedule_publish(self) -> None:
        self.schedule_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_ui_shows_sentence_minimal_metadata_and_two_actions() -> None:
    workflow = FakeWorkflow(make_candidate(), [])
    client = TestClient(create_app(workflow))

    response = client.get("/")

    assert response.status_code == 200
    assert "A sentence about a visible landscape." in response.text
    assert "Wikipedia" in response.text
    assert "A place" in response.text
    assert 'value="yes"' in response.text
    assert 'value="no"' in response.text
    assert "raw_source_row" not in response.text


def test_ui_shows_saved_annotations_and_live_metrics() -> None:
    annotated = Annotation(make_candidate("saved"), Label.YES)
    workflow = FakeWorkflow(make_candidate(), [], [annotated])
    client = TestClient(create_app(workflow))

    response = client.get("/")

    assert response.status_code == 200
    assert "Saved annotations" in response.text
    assert "A sentence about a visible landscape." in response.text
    assert "1 Yes" in response.text
    assert "0 No" in response.text
    assert 'action="/annotation/update"' in response.text
    assert 'action="/annotation/remove"' in response.text


def test_ui_uses_a_compact_annotation_layout() -> None:
    workflow = FakeWorkflow(make_candidate(), [])
    client = TestClient(create_app(workflow))

    response = client.get("/")

    assert response.status_code == 200
    assert '<main class="annotation-shell">' in response.text
    assert '<section class="annotation-card">' in response.text
    assert 'aria-label="Annotation progress"' in response.text
    assert 'class="metadata-grid"' in response.text
    assert 'class="decision-actions"' in response.text


def test_ui_places_continuation_before_saved_annotations() -> None:
    workflow = FakeWorkflow(make_candidate(), [])
    client = TestClient(create_app(workflow))

    response = client.get("/")

    assert response.status_code == 200
    assert response.text.index("Continue labeling") < response.text.index("Saved annotations")


def test_ui_posts_the_selected_label_and_redirects_to_next_candidate() -> None:
    workflow = FakeWorkflow(make_candidate(), [])
    client = TestClient(create_app(workflow))

    response = client.post(
        "/annotate",
        data={"candidate_id": "c-1", "label": "yes"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert workflow.calls == [("c-1", Label.YES)]
    assert workflow.schedule_calls == 1


def test_ui_returns_a_bad_request_for_an_invalid_annotation() -> None:
    workflow = FakeWorkflow(make_candidate(), [])
    client = TestClient(create_app(workflow))

    response = client.post(
        "/annotate",
        data={"candidate_id": "unknown", "label": "yes"},
    )

    assert response.status_code == 400


def test_ui_changes_a_saved_label_and_redirects() -> None:
    annotated = Annotation(make_candidate("saved"), Label.YES)
    workflow = FakeWorkflow(make_candidate(), [], [annotated])
    client = TestClient(create_app(workflow))

    response = client.post(
        "/annotation/update",
        data={"candidate_id": "saved", "label": "no"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert workflow.annotations[0].label is Label.NO
    assert workflow.schedule_calls == 1


def test_ui_removes_a_saved_annotation_and_redirects() -> None:
    annotated = Annotation(make_candidate("saved"), Label.YES)
    workflow = FakeWorkflow(make_candidate(), [], [annotated])
    client = TestClient(create_app(workflow))

    response = client.post(
        "/annotation/remove",
        data={"candidate_id": "saved"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert workflow.annotations == []
    assert workflow.schedule_calls == 1


def test_ui_refreshes_when_removing_an_already_removed_annotation() -> None:
    workflow = FakeWorkflow(make_candidate(), [])
    client = TestClient(create_app(workflow))

    response = client.post(
        "/annotation/remove",
        data={"candidate_id": "already-removed"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_health_endpoint_is_available() -> None:
    client = TestClient(create_app(FakeWorkflow(make_candidate(), [])))

    assert client.get("/health").json() == {"status": "ok"}


def test_ui_closes_the_workflow_when_the_server_lifespan_ends() -> None:
    workflow = FakeWorkflow(make_candidate(), [])

    with TestClient(create_app(workflow)):
        pass

    assert workflow.close_calls == 1
