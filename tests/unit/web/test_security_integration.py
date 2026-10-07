from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.builders import make_annotation_seed, make_annotations

from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.storage.publisher import DatasetPublisher
from landuse_sentence_relevance.storage.session import AnnotationStore
from landuse_sentence_relevance.web.app import create_app
from landuse_sentence_relevance.workflow import AnnotationWorkflow, V3AnnotationWorkflow

BASE_URL = "http://127.0.0.1:8000"


@pytest.mark.parametrize("route", ["/annotate", "/annotation/update", "/annotation/remove"])
def test_cross_site_requests_cannot_change_a_session_or_complete_v2_publication(
    tmp_path: Path, route: str
) -> None:
    rows = make_annotations()
    path = tmp_path / "session.jsonl"
    store = AnnotationStore(path)
    store.save(rows[:-1])
    original = path.read_bytes()
    uploads: list[dict] = []
    workflow = AnnotationWorkflow(
        FinalizedCandidatePool(
            tuple(row.candidate for row in rows), tuple(row.candidate.h3_cell for row in rows)
        ),
        store,
        DatasetPublisher("test/annotations", uploader=lambda **kwargs: uploads.append(kwargs)),
        defer_publish=True,
    )
    target = rows[-1] if route == "/annotate" else rows[0]
    with TestClient(create_app(workflow), base_url=BASE_URL) as client:
        response = client.post(
            route,
            data={"candidate_id": target.candidate.candidate_id, "label": "no"},
            headers={"Origin": "https://evil.example"},
        )
    assert response.status_code == 403
    assert path.read_bytes() == original
    assert workflow.state().labeled_count == 99
    assert not workflow.state().published
    assert uploads == []


def test_same_origin_final_v2_annotation_still_schedules_publication_with_injected_uploader(
    tmp_path: Path,
) -> None:
    rows = make_annotations()
    path = tmp_path / "session.jsonl"
    store = AnnotationStore(path)
    store.save(rows[:-1])
    uploads: list[dict] = []
    workflow = AnnotationWorkflow(
        FinalizedCandidatePool(
            tuple(row.candidate for row in rows), tuple(row.candidate.h3_cell for row in rows)
        ),
        store,
        DatasetPublisher("test/annotations", uploader=lambda **kwargs: uploads.append(kwargs)),
        defer_publish=True,
    )
    with TestClient(create_app(workflow), base_url=BASE_URL) as client:
        response = client.post(
            "/annotate",
            data={"candidate_id": rows[-1].candidate.candidate_id, "label": rows[-1].label.value},
            headers={"Origin": BASE_URL},
            follow_redirects=False,
        )
    assert response.status_code == 303
    assert workflow.state().published
    assert len(uploads) == 1
    assert len(uploads[0]["records"]) == 100
    assert len(store.load()) == 100


def test_same_origin_v3_completion_stays_unpublished(tmp_path: Path) -> None:
    seed = make_annotation_seed()
    store = AnnotationStore(tmp_path / "v3.jsonl")
    workflow = V3AnnotationWorkflow(seed, store)
    labels = {row.candidate.candidate_id: row.quota_label for row in seed.rows}
    with TestClient(create_app(workflow), base_url=BASE_URL) as client:
        while candidate := workflow.current_candidate():
            response = client.post(
                "/annotate",
                data={"candidate_id": candidate.candidate_id, "label": labels[candidate.candidate_id].value},
                headers={"Origin": BASE_URL},
                follow_redirects=False,
            )
            assert response.status_code == 303
    assert workflow.state().final_ready
    assert not workflow.state().published
    assert len(store.load()) == 5
