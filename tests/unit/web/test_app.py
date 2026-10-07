import pytest
from fastapi.testclient import TestClient
from tests.builders import make_candidate
from tests.web_builders import FakeWorkflow

from landuse_sentence_relevance.domain.models import Annotation, Label
from landuse_sentence_relevance.web.app import create_app


def test_ui_shows_sentence_minimal_metadata_and_two_actions() -> None:
    workflow = FakeWorkflow(make_candidate(), [])
    client = TestClient(
        create_app(workflow), base_url="http://127.0.0.1", headers={"Origin": "http://127.0.0.1"}
    )

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
    client = TestClient(
        create_app(workflow), base_url="http://127.0.0.1", headers={"Origin": "http://127.0.0.1"}
    )

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
    client = TestClient(
        create_app(workflow), base_url="http://127.0.0.1", headers={"Origin": "http://127.0.0.1"}
    )

    response = client.get("/")

    assert response.status_code == 200
    assert '<main class="annotation-shell">' in response.text
    assert '<section class="annotation-card">' in response.text
    assert 'aria-label="Annotation progress"' in response.text
    assert 'class="metadata-grid"' in response.text
    assert 'class="decision-actions"' in response.text


def test_ui_places_continuation_before_saved_annotations() -> None:
    workflow = FakeWorkflow(make_candidate(), [])
    client = TestClient(
        create_app(workflow), base_url="http://127.0.0.1", headers={"Origin": "http://127.0.0.1"}
    )

    response = client.get("/")

    assert response.status_code == 200
    assert response.text.index("Continue labeling") < response.text.index("Saved annotations")


def test_ui_posts_the_selected_label_and_redirects_to_next_candidate() -> None:
    workflow = FakeWorkflow(make_candidate(), [])
    client = TestClient(
        create_app(workflow), base_url="http://127.0.0.1", headers={"Origin": "http://127.0.0.1"}
    )

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
    client = TestClient(
        create_app(workflow), base_url="http://127.0.0.1", headers={"Origin": "http://127.0.0.1"}
    )

    response = client.post(
        "/annotate",
        data={"candidate_id": "unknown", "label": "yes"},
    )

    assert response.status_code == 400


def test_ui_returns_a_bad_request_for_an_unknown_label() -> None:
    workflow = FakeWorkflow(make_candidate(), [])
    client = TestClient(
        create_app(workflow), base_url="http://127.0.0.1", headers={"Origin": "http://127.0.0.1"}
    )

    response = client.post("/annotate", data={"candidate_id": "c-1", "label": "maybe"})

    assert response.status_code == 400
    assert workflow.calls == []
    assert workflow.schedule_calls == 0


def test_ui_returns_a_bad_request_when_changing_an_unknown_annotation() -> None:
    workflow = FakeWorkflow(make_candidate(), [])
    client = TestClient(
        create_app(workflow), base_url="http://127.0.0.1", headers={"Origin": "http://127.0.0.1"}
    )

    response = client.post(
        "/annotation/update",
        data={"candidate_id": "missing", "label": "no"},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert workflow.schedule_calls == 0


def test_ui_reports_a_closed_workflow_as_unavailable() -> None:
    workflow = FakeWorkflow(make_candidate(), [], closed=True)
    client = TestClient(
        create_app(workflow), base_url="http://127.0.0.1", headers={"Origin": "http://127.0.0.1"}
    )

    response = client.post(
        "/annotate",
        data={"candidate_id": "c-1", "label": "yes"},
        follow_redirects=False,
    )

    assert response.status_code == 503
    assert workflow.schedule_calls == 0


def test_ui_changes_a_saved_label_and_redirects() -> None:
    annotated = Annotation(make_candidate("saved"), Label.YES)
    workflow = FakeWorkflow(make_candidate(), [], [annotated])
    client = TestClient(
        create_app(workflow), base_url="http://127.0.0.1", headers={"Origin": "http://127.0.0.1"}
    )

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
    client = TestClient(
        create_app(workflow), base_url="http://127.0.0.1", headers={"Origin": "http://127.0.0.1"}
    )

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
    client = TestClient(
        create_app(workflow), base_url="http://127.0.0.1", headers={"Origin": "http://127.0.0.1"}
    )

    response = client.post(
        "/annotation/remove",
        data={"candidate_id": "already-removed"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_health_endpoint_is_available() -> None:
    client = TestClient(create_app(FakeWorkflow(make_candidate(), [])), base_url="http://127.0.0.1")

    assert client.get("/health").json() == {"status": "ok"}


def test_ui_closes_the_workflow_when_the_server_lifespan_ends() -> None:
    workflow = FakeWorkflow(make_candidate(), [])

    with TestClient(
        create_app(workflow), base_url="http://127.0.0.1", headers={"Origin": "http://127.0.0.1"}
    ):
        pass

    assert workflow.close_calls == 1


def test_requested_version_normalizes_and_defaults(monkeypatch) -> None:
    from landuse_sentence_relevance.web.app import _requested_version

    monkeypatch.delenv("ANNOTATION_VERSION", raising=False)
    assert _requested_version() == "v2"
    monkeypatch.setenv("ANNOTATION_VERSION", "  V3 ")
    assert _requested_version() == "v3"


def test_bind_address_defaults_and_overrides(monkeypatch) -> None:
    from landuse_sentence_relevance.web.app import _bind_address

    monkeypatch.delenv("ANNOTATION_HOST", raising=False)
    monkeypatch.delenv("ANNOTATION_PORT", raising=False)
    assert _bind_address() == ("127.0.0.1", 8000)
    monkeypatch.setenv("ANNOTATION_HOST", "127.0.0.1")
    monkeypatch.setenv("ANNOTATION_PORT", "9000")
    assert _bind_address() == ("127.0.0.1", 9000)


@pytest.mark.parametrize("value", ["abc", "0", "-1", "65536", "80.5"])
def test_bind_address_rejects_ports_outside_the_tcp_range(monkeypatch, value: str) -> None:
    from landuse_sentence_relevance.web.app import _bind_address

    monkeypatch.setenv("ANNOTATION_PORT", value)
    with pytest.raises(ValueError, match="ANNOTATION_PORT must be an integer in 1-65535"):
        _bind_address()
