from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from landuse_sentence_relevance.domain.models import Label
from landuse_sentence_relevance.observability import configure_logging
from landuse_sentence_relevance.workflow import UnknownAnnotationError, WorkflowState

logger = logging.getLogger(__name__)

TEMPLATE_DIRECTORY = Path(__file__).parent / "templates"


class WebWorkflow(Protocol):
    def state(self) -> WorkflowState: ...

    def annotate(self, candidate_id: str, label: Label) -> WorkflowState: ...

    def change_label(self, candidate_id: str, label: Label) -> WorkflowState: ...

    def remove_annotation(self, candidate_id: str) -> WorkflowState: ...

    def schedule_publish(self) -> None: ...


def create_app(workflow: WebWorkflow) -> FastAPI:
    templates = Jinja2Templates(directory=str(TEMPLATE_DIRECTORY))
    app = FastAPI(title="Land-use sentence relevance annotation")

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


def run() -> None:  # pragma: no cover - process entrypoint
    import uvicorn

    from landuse_sentence_relevance.bootstrap import build_workflow
    from landuse_sentence_relevance.config import Settings

    configure_logging()
    logger.info("Starting annotation UI; preparing streamed candidates and reusable model cache")
    workflow = build_workflow(Settings.from_env())
    logger.info("Candidate pool ready; starting annotation UI at http://127.0.0.1:8000")
    uvicorn.run(create_app(workflow), host="0.0.0.0", port=8000, log_config=None)
