from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from ipaddress import ip_address
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from landuse_sentence_relevance.domain.models import Label
from landuse_sentence_relevance.observability import configure_logging
from landuse_sentence_relevance.web.security import RequestSecurityMiddleware, trusted_hosts_from_env
from landuse_sentence_relevance.workflow import (
    UnknownAnnotationError,
    UnknownCandidateError,
    V3WorkflowState,
    WorkflowClosedError,
    WorkflowCompleteError,
    WorkflowState,
)

logger = logging.getLogger(__name__)

TEMPLATE_DIRECTORY = Path(__file__).parent / "templates"
_PORT_ERROR = "ANNOTATION_PORT must be an integer in 1-65535"


class InvalidLabelError(ValueError):
    """Raised when a submitted form value is not a known annotation label."""


class WebWorkflow(Protocol):
    def state(self) -> WorkflowState | V3WorkflowState: ...

    def annotate(self, candidate_id: str, label: Label) -> WorkflowState | V3WorkflowState: ...

    def change_label(self, candidate_id: str, label: Label) -> WorkflowState | V3WorkflowState: ...

    def remove_annotation(self, candidate_id: str) -> WorkflowState | V3WorkflowState: ...

    def schedule_publish(self) -> None: ...

    def close(self) -> None: ...


def create_app(workflow: WebWorkflow) -> FastAPI:
    trusted_hosts = trusted_hosts_from_env()
    templates = Jinja2Templates(directory=str(TEMPLATE_DIRECTORY))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            workflow.close()

    app = FastAPI(title="Land-use sentence relevance annotation", lifespan=lifespan)

    app.add_middleware(RequestSecurityMiddleware, trusted_hosts=trusted_hosts)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        state = workflow.state()
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"state": state, "candidate": state.current_candidate},
        )

    def _apply(action: Callable[[], object], *, unknown_annotation_is_noop: bool = False) -> Response:
        """Run one UI mutation, schedule publication, and redirect back to the candidate page."""

        try:
            action()
            workflow.schedule_publish()
        except UnknownAnnotationError as error:
            if unknown_annotation_is_noop:
                return RedirectResponse(url="/", status_code=303)
            return PlainTextResponse(str(error), status_code=400)
        except (InvalidLabelError, UnknownCandidateError, WorkflowCompleteError) as error:
            return PlainTextResponse(str(error), status_code=400)
        except WorkflowClosedError:
            logger.warning("Annotation request rejected because the workflow is closed")
            return PlainTextResponse("the annotation workflow is closed", status_code=503)
        return RedirectResponse(url="/", status_code=303)

    @app.post("/annotate", response_model=None)
    def annotate(
        candidate_id: str = Form(...),
        label: str = Form(...),
    ) -> Response:
        return _apply(lambda: workflow.annotate(candidate_id, _parse_label(label)))

    @app.post("/annotation/update", response_model=None)
    def update_annotation(
        candidate_id: str = Form(...),
        label: str = Form(...),
    ) -> Response:
        return _apply(lambda: workflow.change_label(candidate_id, _parse_label(label)))

    @app.post("/annotation/remove", response_model=None)
    def remove_annotation(
        candidate_id: str = Form(...),
    ) -> Response:
        return _apply(
            lambda: workflow.remove_annotation(candidate_id),
            unknown_annotation_is_noop=True,
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _requested_version() -> str:
    """Return the normalized annotation profile requested by the environment."""

    return os.environ.get("ANNOTATION_VERSION", "v2").strip().casefold()


def _port_from_env(raw: str) -> int:
    """Return a TCP port from ANNOTATION_PORT, rejecting anything outside 1-65535."""

    text = raw.strip() or "8000"
    try:
        port = int(text)
    except ValueError:
        raise ValueError(_PORT_ERROR) from None
    if not 1 <= port <= 65535:
        raise ValueError(_PORT_ERROR)
    return port


def _parse_label(raw: str) -> Label:
    try:
        return Label(raw)
    except ValueError as error:
        raise InvalidLabelError(str(error)) from error


def _bind_address() -> tuple[str, int]:
    """Return the host and port for the annotation UI (default 127.0.0.1:8000)."""

    host = os.environ.get("ANNOTATION_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = _port_from_env(os.environ.get("ANNOTATION_PORT", "8000"))
    if not _is_loopback_host(host):
        logger.warning(
            "Annotation UI has no authentication; binding to %s exposes labels and publication "
            "to reachable clients. Restrict network access and configure ANNOTATION_TRUSTED_HOSTS.",
            host,
        )
    return host, port


def _is_loopback_host(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def build_annotation_workflow() -> WebWorkflow:
    """Select the isolated annotation profile requested by the environment."""

    from landuse_sentence_relevance.bootstrap import build_v3_workflow, build_workflow
    from landuse_sentence_relevance.config import Settings, V3Settings

    version = _requested_version()
    if version == "v3":
        return build_v3_workflow(V3Settings.from_env())
    if version == "v2":
        return build_workflow(Settings.from_env())
    raise ValueError("ANNOTATION_VERSION must be v2 or v3")


def run() -> None:  # pragma: no cover - process entrypoint
    import uvicorn

    configure_logging()
    version = _requested_version()
    host, port = _bind_address()
    logger.info("Starting %s annotation UI", version)
    workflow = build_annotation_workflow()
    logger.info("Candidate pool ready; starting annotation UI at http://%s:%d", host, port)
    uvicorn.run(create_app(workflow), host=host, port=port, log_config=None, proxy_headers=False)
