from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from tests.builders import make_annotation_seed

from landuse_sentence_relevance.storage.session import AnnotationStore
from landuse_sentence_relevance.web.app import create_app
from landuse_sentence_relevance.workflow import V3AnnotationWorkflow


def _client(tmp_path: Path) -> tuple[TestClient, V3AnnotationWorkflow]:
    workflow = V3AnnotationWorkflow(
        make_annotation_seed(),
        AnnotationStore(tmp_path / "results" / "annotations" / "sessions" / "v3.jsonl"),
    )
    return TestClient(create_app(workflow)), workflow


def test_v3_ui_shows_total_source_and_quota_progress_without_seeded_rows(tmp_path: Path) -> None:
    client, workflow = _client(tmp_path)
    try:
        response = client.get("/")
    finally:
        client.close()

    assert response.status_code == 200
    assert "V3 human annotation" in response.text
    assert "Total progress" in response.text
    assert "1 / 6" in response.text
    assert 'data-source="wikipedia"' in response.text
    assert 'data-source="website"' in response.text
    assert 'data-source="description"' in response.text
    assert "Target: 3 Yes · 3 No" in response.text
    assert "Remaining quota" in response.text
    assert "Sentence for wikipedia-no." in response.text
    assert "Sentence for wikipedia-yes." not in response.text
    assert "Frozen V2 labels: 1 (read-only; never shown for reannotation)." in response.text
    assert workflow.state().current_candidate is not None


def test_v3_ui_submits_a_label_and_refreshes_progress(tmp_path: Path) -> None:
    client, workflow = _client(tmp_path)
    try:
        candidate = workflow.current_candidate()
        assert candidate is not None

        response = client.post(
            "/annotate",
            data={"candidate_id": candidate.candidate_id, "label": "no"},
            follow_redirects=False,
        )
        refreshed = client.get("/")
    finally:
        client.close()

    assert response.status_code == 303
    assert refreshed.status_code == 200
    assert "2 / 6" in refreshed.text
    assert "No" in refreshed.text
