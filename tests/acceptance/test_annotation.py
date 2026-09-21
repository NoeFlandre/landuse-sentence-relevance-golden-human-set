from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import Page, sync_playwright
from pytest_bdd import given, parsers, scenarios, then, when

from landuse_sentence_relevance.domain.models import Label
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.storage.publisher import DatasetPublisher
from landuse_sentence_relevance.storage.session import AnnotationStore
from landuse_sentence_relevance.web.app import create_app
from landuse_sentence_relevance.workflow import AnnotationWorkflow
from tests.builders import make_candidate

scenarios("features/annotation.feature")
pytestmark = pytest.mark.acceptance


@pytest.fixture
def annotation_store(tmp_path: Path) -> AnnotationStore:
    return AnnotationStore(tmp_path / "annotations.jsonl")


@pytest.fixture
def browser_workflow(annotation_store: AnnotationStore) -> AnnotationWorkflow:
    candidates = (
        make_candidate(),
        replace(make_candidate("c-2"), sentence="Another sentence.", h3_cell="832831fffffffff"),
    )
    return AnnotationWorkflow(
        pool=FinalizedCandidatePool(candidates, tuple(candidate.h3_cell for candidate in candidates)),
        store=annotation_store,
        publisher=DatasetPublisher("test/annotations", uploader=lambda **kwargs: None),
    )


@pytest.fixture
def live_url(browser_workflow: AnnotationWorkflow) -> Iterator[str]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(browser_workflow), host="127.0.0.1", port=port, log_level="error")
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
        raise AssertionError("UI server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def page() -> Iterator[Page]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        yield page
        browser.close()


@given("the annotation app is running")
def annotation_app_is_running(page: Page, live_url: str) -> None:
    response = page.request.get(f"{live_url}/health")
    assert response.status == 200
    assert response.json() == {"status": "ok"}


@when("I open the annotation page")
def open_annotation_page(page: Page, live_url: str) -> None:
    page.goto(live_url)


@then("I see the sentence and minimal place metadata")
def see_sentence_and_metadata(page: Page) -> None:
    assert page.get_by_text("A sentence about a visible landscape.").is_visible()
    assert page.get_by_text("A place").is_visible()
    assert page.get_by_text("H3 cell").is_visible()


@when(parsers.parse('I choose "{label}"'))
def choose_label(page: Page, label: str) -> None:
    page.get_by_role("button", name=label).click()


@then(parsers.parse("the Yes count is {count:d}"))
def yes_count_is(page: Page, count: int, annotation_store: AnnotationStore) -> None:
    assert page.locator(".metric-yes").get_by_text(str(count), exact=True).is_visible()
    assert sum(annotation.label is Label.YES for annotation in annotation_store.load().values()) == count


@then(parsers.parse("the No count is {count:d}"))
def no_count_is(page: Page, count: int, annotation_store: AnnotationStore) -> None:
    assert page.locator(".metric-no").get_by_text(str(count), exact=True).is_visible()
    assert sum(annotation.label is Label.NO for annotation in annotation_store.load().values()) == count


@then(parsers.re(r"the saved annotation list shows (?P<count>\d+) records?"))
def saved_annotation_list_shows(page: Page, count: str, annotation_store: AnnotationStore) -> None:
    assert page.locator(".section-count").inner_text() == f"{count} records"
    assert len(annotation_store.load()) == int(count)


@when(parsers.parse('I change the saved label to "{label}"'))
def change_saved_label(page: Page, label: str) -> None:
    page.get_by_role("button", name=f"Change label to {label}").click()


@when("I remove the saved annotation")
def remove_saved_annotation(page: Page) -> None:
    page.once("dialog", lambda dialog: dialog.accept())
    page.get_by_role("button", name="Remove").click()
