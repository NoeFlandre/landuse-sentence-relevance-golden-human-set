from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import Page, sync_playwright
from pytest_bdd import given, parsers, scenarios, then, when
from tests.builders import make_annotation_seed

from landuse_sentence_relevance.domain.models import Label
from landuse_sentence_relevance.storage.session import AnnotationStore
from landuse_sentence_relevance.web.app import create_app
from landuse_sentence_relevance.workflow import V3AnnotationWorkflow

scenarios("features/v3_annotation.feature")
pytestmark = pytest.mark.acceptance


@pytest.fixture
def v3_annotation_store(tmp_path: Path) -> AnnotationStore:
    return AnnotationStore(tmp_path / "results" / "annotations" / "sessions" / "v3.jsonl")


@pytest.fixture
def v3_browser_workflow(v3_annotation_store: AnnotationStore) -> V3AnnotationWorkflow:
    return V3AnnotationWorkflow(make_annotation_seed(), v3_annotation_store)


@pytest.fixture
def v3_live_url(v3_browser_workflow: V3AnnotationWorkflow) -> Iterator[str]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(v3_browser_workflow),
            host="127.0.0.1",
            port=port,
            log_level="error",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise AssertionError("V3 UI server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def v3_page() -> Iterator[Page]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        yield page
        browser.close()


@given("the V3 annotation app is running")
def v3_annotation_app_is_running(v3_page: Page, v3_live_url: str) -> None:
    response = v3_page.request.get(f"{v3_live_url}/health")
    assert response.status == 200
    assert response.json() == {"status": "ok"}


@when("I open the V3 annotation page")
def open_v3_annotation_page(v3_page: Page, v3_live_url: str) -> None:
    v3_page.goto(v3_live_url)


@then("I see V3 totals and per-source quotas")
def see_v3_totals_and_quotas(v3_page: Page) -> None:
    assert v3_page.get_by_text("V3 human annotation").is_visible()
    assert v3_page.get_by_text("Total progress").is_visible()
    assert v3_page.get_by_text("Per-source progress").is_visible()
    assert v3_page.get_by_text("Target: 3 Yes · 3 No").is_visible()
    for source in ("wikipedia", "website", "description"):
        assert v3_page.locator(f'[data-source="{source}"]').is_visible()


@then("the seeded V2 sentence is not shown")
def seeded_v2_sentence_is_not_shown(v3_page: Page) -> None:
    assert not v3_page.get_by_text("Sentence for wikipedia-yes.", exact=True).is_visible()


@when(parsers.parse('I label the fresh V3 sentence "{label}"'))
def label_fresh_v3_sentence(v3_page: Page, label: str) -> None:
    v3_page.get_by_role("button", name=label).click()


@then(parsers.parse('the V3 total progress is "{value}"'))
def v3_total_progress_is(v3_page: Page, value: str) -> None:
    metric = v3_page.locator(".metric").filter(has_text="Total progress")
    assert metric.locator(".metric-value").inner_text() == value


@then("the V3 session contains one fresh label")
def v3_session_contains_one_fresh_label(v3_annotation_store: AnnotationStore) -> None:
    annotations = v3_annotation_store.load()
    assert len(annotations) == 1
    assert next(iter(annotations.values())).label is Label.NO


@when("I reload the V3 annotation page")
def reload_v3_annotation_page(v3_page: Page) -> None:
    v3_page.reload()


@then("the next fresh V3 sentence is shown")
def next_fresh_v3_sentence_is_shown(v3_page: Page) -> None:
    assert v3_page.get_by_text("Sentence for website-yes.", exact=True).is_visible()
    assert not v3_page.get_by_text("Sentence for wikipedia-yes.", exact=True).is_visible()


@when(parsers.parse('I change the V3 saved label to "{label}"'))
def change_v3_saved_label(v3_page: Page, label: str) -> None:
    v3_page.get_by_role("button", name=f"Change label to {label}").click()


@then("the V3 Yes target shows one saved label")
def v3_yes_target_shows_one_saved_label(v3_page: Page) -> None:
    assert v3_page.locator(".metric-yes .metric-value").inner_text() == "1 / 3"
    assert v3_page.locator(".metric-no .metric-value").inner_text() == "1 / 3"


@when("I remove the V3 saved annotation")
def remove_v3_saved_annotation(v3_page: Page) -> None:
    v3_page.once("dialog", lambda dialog: dialog.accept())
    v3_page.get_by_role("button", name="Remove").click()


@then("the V3 fresh review is empty")
def v3_fresh_review_is_empty(v3_page: Page, v3_annotation_store: AnnotationStore) -> None:
    assert v3_page.locator(".section-count").inner_text() == "0 fresh records"
    assert v3_annotation_store.load() == {}
