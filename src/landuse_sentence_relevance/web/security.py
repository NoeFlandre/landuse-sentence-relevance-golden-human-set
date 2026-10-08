from __future__ import annotations

import os
import re
from ipaddress import ip_address
from urllib.parse import SplitResult, urlsplit

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp

Origin = tuple[str, str, int]
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _http_url(parts: SplitResult) -> bool:
    return (
        parts.scheme in {"http", "https"}
        and parts.hostname is not None
        and parts.username is None
        and parts.password is None
    )


def _has_url_suffix(parts: SplitResult) -> bool:
    return bool(parts.path or parts.query or parts.fragment)


def _origin_parts(parts: SplitResult, *, referer: bool) -> Origin:
    if not referer and _has_url_suffix(parts):
        raise ValueError("Origin must not contain a path, query, or fragment")
    port = parts.port
    if port is None:
        port = {"http": 80, "https": 443}[parts.scheme]
    return parts.scheme, parts.hostname or "", port


def _valid_url_characters(value: str, *, referer: bool) -> bool:
    forbidden = r"[\s\x00-\x1f\x7f\\]" if referer else r"[\s\x00-\x1f\x7f\\?#]"
    return re.search(forbidden, value) is None


def parse_origin(value: str, *, referer: bool = False) -> Origin | None:
    """Parse a full HTTP origin; only Referer may contain a path or query."""

    if not _valid_url_characters(value, referer=referer):
        return None
    try:
        parts = urlsplit(value)
        if not _http_url(parts):
            return None
        return _origin_parts(parts, referer=referer)
    except ValueError:
        return None


def trusted_hosts_from_env() -> frozenset[str]:
    """Add explicitly configured hosts to the local-only default. Wildcards are forbidden."""

    trusted = LOCAL_HOSTS | _loopback_bind_host()
    configured = os.environ.get("ANNOTATION_TRUSTED_HOSTS", "").strip()
    if not configured:
        return trusted
    return trusted | frozenset(_trusted_host(value.strip()) for value in configured.split(","))


def _loopback_bind_host() -> frozenset[str]:
    """Trust the configured bind address only when it is an explicit loopback IP literal."""

    try:
        address = ip_address(os.environ.get("ANNOTATION_HOST", "").strip())
    except ValueError:
        return frozenset()
    return frozenset({str(address)}) if address.is_loopback else frozenset()


def _trusted_host(value: str) -> str:
    origin = parse_origin(f"http://{value}")
    if origin is None or "*" in value:
        raise ValueError("ANNOTATION_TRUSTED_HOSTS must contain exact hostnames or bracketed IPv6 addresses")
    # Ports belong to each request's origin, not the hostname allowlist.
    if value.removeprefix("[").removesuffix("]").lower() != origin[1]:
        raise ValueError("ANNOTATION_TRUSTED_HOSTS must not contain ports")
    return origin[1]


def _single_header(request: Request, name: str) -> str:
    values = request.headers.getlist(name)
    return values[0] if len(values) == 1 else ""


def _request_origin(request: Request) -> Origin | None:
    host = _single_header(request, "host")
    return parse_origin(f"{request.scope['scheme']}://{host}")


def _mutation_origin(request: Request) -> Origin | None:
    if "origin" in request.headers:
        return parse_origin(_single_header(request, "origin"))
    return parse_origin(_single_header(request, "referer"), referer=True)


class RequestSecurityMiddleware(BaseHTTPMiddleware):
    """Reject untrusted hosts and cross-origin writes before reading any form body."""

    def __init__(self, app: ASGIApp, *, trusted_hosts: frozenset[str]) -> None:
        super().__init__(app)
        self.trusted_hosts = trusted_hosts

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        origin = _request_origin(request)
        if origin is None or origin[1] not in self.trusted_hosts:
            return PlainTextResponse("Invalid Host header", status_code=400)
        if request.method not in SAFE_METHODS and not self._same_origin_write(request, origin):
            return PlainTextResponse("A same-origin Origin or Referer header is required", status_code=403)
        return await call_next(request)

    @staticmethod
    def _same_origin_write(request: Request, origin: Origin) -> bool:
        if request.headers.get("sec-fetch-site") == "cross-site":
            return False
        return _mutation_origin(request) == origin
