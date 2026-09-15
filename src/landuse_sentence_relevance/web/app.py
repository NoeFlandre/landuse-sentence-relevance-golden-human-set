from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from landuse_sentence_relevance.domain.models import Label
from landuse_sentence_relevance.observability import configure_logging
from landuse_sentence_relevance.workflow import (
    UnknownAnnotationError,
    V3WorkflowState,
    WorkflowState,
)

logger = logging.getLogger(__name__)

TEMPLATE_DIRECTORY = Path(__file__).parent / "templates"


class WebWorkflow(Protocol):
    def state(self) -> WorkflowState | V3WorkflowState: ...

    def annotate(self, candidate_id: str, label: Label) -> WorkflowState | V3WorkflowState: ...

    def change_label(self, candidate_id: str, label: Label) -> WorkflowState | V3WorkflowState: ...

    def remove_annotation(self, candidate_id: str) -> WorkflowState | V3WorkflowState: ...

    def schedule_publish(self) -> None: ...

    def close(self) -> None: ...


def create_app(workflow: WebWorkflow) -> FastAPI:
    templates = Jinja2Templates(directory=str(TEMPLATE_DIRECTORY))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            workflow.close()

    app = FastAPI(title="Land-use sentence relevance annotation", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        state = workflow.state()
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"state": state, "candidate": state.current_candidate},
        )

    @app.post("/annotate", response_model=None)
    def annotate(
        candidate_id: str = Form(...),
        label: str = Form(...),
    ) -> Response:
        try:
            workflow.annotate(candidate_id, Label(label))
        except (ValueError, KeyError) as error:
            return PlainTextResponse(str(error), status_code=400)
        workflow.schedule_publish()
        return RedirectResponse(url="/", status_code=303)

    @app.post("/annotation/update", response_model=None)
    def update_annotation(
        candidate_id: str = Form(...),
        label: str = Form(...),
    ) -> Response:
        try:
            workflow.change_label(candidate_id, Label(label))
        except (ValueError, KeyError) as error:
            return PlainTextResponse(str(error), status_code=400)
        workflow.schedule_publish()
        return RedirectResponse(url="/", status_code=303)

    @app.post("/annotation/remove", response_model=None)
    def remove_annotation(
        candidate_id: str = Form(...),
    ) -> Response:
        try:
            workflow.remove_annotation(candidate_id)
        except UnknownAnnotationError:
            return RedirectResponse(url="/", status_code=303)
        except (ValueError, KeyError) as error:
            return PlainTextResponse(str(error), status_code=400)
        workflow.schedule_publish()
        return RedirectResponse(url="/", status_code=303)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def build_annotation_workflow() -> WebWorkflow:
    """Select the isolated annotation profile requested by the environment."""

    from landuse_sentence_relevance.bootstrap import build_v3_workflow, build_workflow
    from landuse_sentence_relevance.config import Settings, V3Settings

    version = os.environ.get("ANNOTATION_VERSION", "v2").strip().casefold()
    if version == "v3":
        return build_v3_workflow(V3Settings.from_env())
    if version == "v2":
        return build_workflow(Settings.from_env())
    raise ValueError("ANNOTATION_VERSION must be v2 or v3")


def run() -> None:  # pragma: no cover - process entrypoint
    import uvicorn

    configure_logging()
    version = os.environ.get("ANNOTATION_VERSION", "v2").strip().casefold()
    logger.info("Starting %s annotation UI", version)
    workflow = build_annotation_workflow()
    logger.info("Candidate pool ready; starting annotation UI at http://127.0.0.1:8000")
    uvicorn.run(create_app(workflow), host="0.0.0.0", port=8000, log_config=None)
