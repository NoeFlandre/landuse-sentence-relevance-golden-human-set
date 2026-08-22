from __future__ import annotations

from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from landuse_sentence_relevance.domain.models import Label
from landuse_sentence_relevance.workflow import WorkflowState

TEMPLATE_DIRECTORY = Path(__file__).parent / "templates"


class WebWorkflow(Protocol):
    def state(self) -> WorkflowState: ...

    def annotate(self, candidate_id: str, label: Label) -> WorkflowState: ...


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
    def annotate(candidate_id: str = Form(...), label: str = Form(...)) -> Response:
        try:
            workflow.annotate(candidate_id, Label(label))
        except (ValueError, KeyError) as error:
            return PlainTextResponse(str(error), status_code=400)
        return RedirectResponse(url="/", status_code=303)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def run() -> None:  # pragma: no cover - process entrypoint
    import uvicorn

    from landuse_sentence_relevance.bootstrap import build_workflow
    from landuse_sentence_relevance.config import Settings

    uvicorn.run(create_app(build_workflow(Settings.from_env())), host="0.0.0.0", port=8000)
