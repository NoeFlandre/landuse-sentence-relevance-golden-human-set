from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace

import pytest
import uvicorn
from playwright.sync_api import Page, sync_playwright
from pytest_bdd import given, parsers, scenarios, then, when
from tests.unit.test_models import make_candidate

from landuse_sentence_relevance.domain.models import Annotation, Candidate, Label
from landuse_sentence_relevance.web.app import create_app
from landuse_sentence_relevance.workflow import WorkflowState

scenarios("features/annotation.feature")
pytestmark = pytest.mark.acceptance


@dataclass
class BrowserWorkflow:
    candidates: tuple[Candidate, ...]
    calls: list[tuple[str, Label]]

    def state(self) -> WorkflowState:
        labeled = {candidate_id for candidate_id, _ in self.calls}
        annotations = tuple(
            Annotation(
                candidate=next(
                    candidate for candidate in self.candidates if candidate.candidate_id == candidate_id
                ),
                label=label,
            )
            for candidate_id, label in self.calls
        )
        current = next(
            (candidate for candidate in self.candidates if candidate.candidate_id not in labeled),
            None,
        )
        return WorkflowState(
            current_candidate=current,
            labeled_count=len(annotations),
            yes_count=sum(annotation.label is Label.YES for annotation in annotations),
            no_count=sum(annotation.label is Label.NO for annotation in annotations),
            final_ready=False,
            published=False,
            annotations=annotations,
        )

    def annotate(self, candidate_id: str, label: Label) -> WorkflowState:
        if candidate_id not in {candidate.candidate_id for candidate in self.candidates}:
            raise ValueError("unknown candidate")
        self.calls.append((candidate_id, label))
        return self.state()

    def change_label(self, candidate_id: str, label: Label) -> WorkflowState:
        for index, (saved_id, _) in enumerate(self.calls):
            if saved_id == candidate_id:
                self.calls[index] = (candidate_id, label)
                return self.state()
        raise ValueError("unknown annotation")

    def remove_annotation(self, candidate_id: str) -> WorkflowState:
        self.calls[:] = [(saved_id, label) for saved_id, label in self.calls if saved_id != candidate_id]
        return self.state()

    def schedule_publish(self) -> None:
        return None


@pytest.fixture
def browser_workflow() -> BrowserWorkflow:
    return BrowserWorkflow(
        candidates=(make_candidate(), replace(make_candidate("c-2"), sentence="Another sentence.")),
        calls=[],
    )


@pytest.fixture
def live_url(browser_workflow: BrowserWorkflow) -> Iterator[str]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(browser_workflow), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
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
def annotation_app_is_running(live_url: str) -> None:
    assert live_url.startswith("http://127.0.0.1:")


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
def yes_count_is(page: Page, count: int) -> None:
    assert page.locator(".metric-yes").get_by_text(str(count), exact=True).is_visible()


@then(parsers.parse("the No count is {count:d}"))
def no_count_is(page: Page, count: int) -> None:
    assert page.locator(".metric-no").get_by_text(str(count), exact=True).is_visible()


@then(parsers.re(r"the saved annotation list shows (?P<count>\d+) records?"))
def saved_annotation_list_shows(page: Page, count: str) -> None:
    assert page.locator(".section-count").inner_text() == f"{count} records"


@when(parsers.parse('I change the saved label to "{label}"'))
def change_saved_label(page: Page, label: str) -> None:
    page.get_by_role("button", name=f"Change label to {label}").click()


@when("I remove the saved annotation")
def remove_saved_annotation(page: Page) -> None:
    page.once("dialog", lambda dialog: dialog.accept())
    page.get_by_role("button", name="Remove").click()
