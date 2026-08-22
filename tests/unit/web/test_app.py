from dataclasses import dataclass

from fastapi.testclient import TestClient
from tests.unit.test_models import make_candidate

from landuse_sentence_relevance.domain.models import Candidate, Label
from landuse_sentence_relevance.web.app import create_app
from landuse_sentence_relevance.workflow import WorkflowState


@dataclass
class FakeWorkflow:
    candidate: Candidate
    calls: list[tuple[str, Label]]

    def current_candidate(self) -> Candidate:
        return self.candidate

    def state(self) -> WorkflowState:
        return WorkflowState(
            current_candidate=self.candidate,
            labeled_count=len(self.calls),
            yes_count=sum(label is Label.YES for _, label in self.calls),
            no_count=sum(label is Label.NO for _, label in self.calls),
            final_ready=False,
            published=False,
        )

    def annotate(self, candidate_id: str, label: Label) -> WorkflowState:
        if candidate_id != self.candidate.candidate_id:
            raise ValueError("unknown candidate")
        self.calls.append((candidate_id, label))
        return self.state()


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


def test_ui_returns_a_bad_request_for_an_invalid_annotation() -> None:
    workflow = FakeWorkflow(make_candidate(), [])
    client = TestClient(create_app(workflow))

    response = client.post(
        "/annotate",
        data={"candidate_id": "unknown", "label": "yes"},
    )

    assert response.status_code == 400


def test_health_endpoint_is_available() -> None:
    client = TestClient(create_app(FakeWorkflow(make_candidate(), [])))

    assert client.get("/health").json() == {"status": "ok"}
