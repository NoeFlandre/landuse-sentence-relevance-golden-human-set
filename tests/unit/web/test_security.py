from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient
from tests.builders import make_candidate
from tests.web_builders import FakeWorkflow

from landuse_sentence_relevance.domain.models import Annotation, Label
from landuse_sentence_relevance.web.app import _bind_address, create_app

BASE_URL = "http://127.0.0.1:8000"
ROUTES = ("/annotate", "/annotation/update", "/annotation/remove")


def make_client(base_url: str = BASE_URL) -> tuple[TestClient, FakeWorkflow]:
    candidate = make_candidate()
    workflow = FakeWorkflow(candidate, [], [Annotation(candidate, Label.YES)])
    return TestClient(create_app(workflow), base_url=base_url), workflow


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Origin": "https://evil.example"},
        {"Origin": "http://127.0.0.1:8001"},
        {"Origin": "https://127.0.0.1:8000"},
        {"Origin": "http://localhost:8000"},
        {"Origin": "null"},
        {"Origin": ""},
        {"Origin": "http://127.0.0.1:8000.evil.example"},
        {"Origin": "http://127.0.0.1:8000@evil.example"},
        {"Origin": "http://127.0.0.1:8000/path"},
        {"Origin": "http://127.0.0.1:8000?x=1"},
        {"Origin": "http://127.0.0.1:8000#fragment"},
        {"Origin": "http://127.0.0.1:8000 http://evil.example"},
        {"Origin": "http://127.0.0.1:8000,"},
        {"Origin": "http://127.0.0.1:8000", "Sec-Fetch-Site": "cross-site"},
        {"Origin": "http://evil.example", "Referer": BASE_URL + "/"},
        {"Origin": "null", "Referer": BASE_URL + "/"},
        {"Referer": "https://127.0.0.1:8000/"},
        {"Referer": "http://127.0.0.1:8001/"},
        {"Referer": "http://evil.example/"},
        {"Referer": "null"},
        {"Sec-Fetch-Site": "same-origin"},
        {"X-Forwarded-Host": "evil.example", "X-Forwarded-Proto": "http"},
    ],
)
def test_rejects_untrusted_mutations_before_workflow_calls(route: str, headers: dict[str, str]) -> None:
    client, workflow = make_client()
    original = workflow.annotations.copy()

    response = client.post(route, data={"candidate_id": "c-1", "label": "no"}, headers=headers)

    assert response.status_code == 403
    assert workflow.annotations == original
    assert workflow.calls == []
    assert workflow.schedule_calls == 0


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("headers", [{"Origin": BASE_URL}, {"Referer": BASE_URL + "/?view=1"}])
def test_accepts_same_origin_forms(route: str, headers: dict[str, str]) -> None:
    client, workflow = make_client()

    response = client.post(
        route, data={"candidate_id": "c-1", "label": "no"}, headers=headers, follow_redirects=False
    )

    assert response.status_code == 303
    assert workflow.schedule_calls == 1


@pytest.mark.parametrize("path", ["/", "/health", "/annotate"])
@pytest.mark.parametrize("host", ["evil.example:8000", "127.0.0.1.evil.example", "0.0.0.0:8000"])
def test_rejects_untrusted_hosts_including_dns_rebinding(path: str, host: str) -> None:
    client, workflow = make_client()

    response = client.get(path, headers={"Host": host})

    assert response.status_code == 400
    assert workflow.schedule_calls == 0


@pytest.mark.parametrize("header", ["Host", "Origin", "Referer"])
def test_rejects_ambiguous_duplicate_security_headers(header: str) -> None:
    client, workflow = make_client()
    values = {
        "Host": "127.0.0.1:8000",
        "Origin": BASE_URL,
        "Referer": BASE_URL + "/",
    }
    headers = [(header, values[header]), (header, "http://evil.example")]

    response = client.post("/annotate", headers=headers, data={"candidate_id": "c-1", "label": "no"})

    assert response.status_code in (400, 403)
    assert workflow.calls == []
    assert workflow.schedule_calls == 0


@pytest.mark.parametrize("base_url", ["http://localhost:8765", "http://[::1]:8000", "https://localhost"])
def test_all_loopback_hosts_and_full_origins_work(base_url: str) -> None:
    client, workflow = make_client()
    client.headers["Host"] = base_url.split("://", 1)[1]
    if base_url.startswith("https:"):
        client.base_url = base_url

    assert client.get("/").status_code == 200
    response = client.post(
        "/annotate",
        headers={"Origin": base_url},
        data={"candidate_id": "c-1", "label": "no"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert workflow.calls == [("c-1", Label.NO)]


def test_explicit_remote_host_opt_in_keeps_same_origin_protection(monkeypatch) -> None:
    monkeypatch.setenv("ANNOTATION_TRUSTED_HOSTS", "annotator.example, 192.0.2.1")
    client, workflow = make_client("https://annotator.example:8443")

    assert client.get("/").status_code == 200
    response = client.post(
        "/annotate",
        headers={"Origin": "https://annotator.example:8443"},
        data={"candidate_id": "c-1", "label": "no"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert client.post("/annotate", headers={"Origin": "https://annotator.example:8444"}).status_code == 403
    assert workflow.schedule_calls == 1


def test_default_and_blank_bind_are_loopback(monkeypatch) -> None:
    monkeypatch.delenv("ANNOTATION_HOST", raising=False)
    monkeypatch.delenv("ANNOTATION_PORT", raising=False)
    assert _bind_address() == ("127.0.0.1", 8000)
    monkeypatch.setenv("ANNOTATION_HOST", " ")
    monkeypatch.setenv("ANNOTATION_PORT", " ")
    assert _bind_address() == ("127.0.0.1", 8000)


def test_explicit_non_loopback_bind_warns_without_changing_config(monkeypatch, caplog) -> None:
    monkeypatch.setenv("ANNOTATION_HOST", "0.0.0.0")
    monkeypatch.setenv("ANNOTATION_PORT", "9000")
    with caplog.at_level(logging.WARNING):
        assert _bind_address() == ("0.0.0.0", 9000)
    assert "authentication" in caplog.text


@pytest.mark.parametrize(
    "host",
    [
        "",
        "localhost:bad",
        "[broken",
        "localhost/path",
        "user@localhost",
        "localhost\\evil",
        "localhost:8000?x=1",
        "localhost:65536",
    ],
)
def test_malformed_host_is_rejected_without_server_error(host: str) -> None:
    client, workflow = make_client()
    response = client.post(
        "/annotate", headers={"Host": host, "Origin": BASE_URL}, data={"candidate_id": "c-1", "label": "no"}
    )
    assert response.status_code == 400
    assert workflow.calls == []


@pytest.mark.parametrize(
    "origin",
    [
        "ftp://127.0.0.1:8000",
        "http://[broken",
        "http://user@127.0.0.1:8000",
        "http://user:pass@127.0.0.1:8000",
        "http://127.0.0.1:65536",
        "http://127.0.0.1:bad",
        "http://127.0.0.1:8000\\evil",
        "http://127.0.0.1:0",
    ],
)
def test_malformed_and_non_http_origins_are_forbidden(origin: str) -> None:
    client, workflow = make_client()
    response = client.post(
        "/annotate", headers={"Origin": origin}, data={"candidate_id": "c-1", "label": "no"}
    )
    assert response.status_code == 403
    assert workflow.calls == []


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "TRACE"])
def test_all_unsafe_methods_require_an_origin(method: str) -> None:
    client, workflow = make_client()
    response = client.request(method, "/health")
    assert response.status_code == 403
    assert workflow.schedule_calls == 0


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS"])
def test_safe_methods_do_not_require_an_origin(method: str) -> None:
    client, _ = make_client()
    response = client.request(method, "/health")
    assert response.status_code in (200, 405)


@pytest.mark.parametrize(
    "configured",
    [
        "*",
        "*.example",
        "https://example.org",
        "localhost:8000",
        "example.org/",
        "example.org,,localhost",
        "[invalid]",
        "user@example.org",
    ],
)
def test_invalid_trusted_host_configuration_fails_closed(configured: str, monkeypatch) -> None:
    monkeypatch.setenv("ANNOTATION_TRUSTED_HOSTS", configured)
    with pytest.raises(ValueError, match="ANNOTATION_TRUSTED_HOSTS"):
        make_client()


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "localhost", "LOCALHOST", "::1"])
def test_loopback_bind_does_not_warn(host: str, monkeypatch, caplog) -> None:
    monkeypatch.setenv("ANNOTATION_HOST", host)
    monkeypatch.setenv("ANNOTATION_PORT", "8000")
    with caplog.at_level(logging.WARNING):
        assert _bind_address() == (host, 8000)
    assert not caplog.records


def test_remote_hostname_bind_warns(monkeypatch, caplog) -> None:
    monkeypatch.setenv("ANNOTATION_HOST", "annotator.example")
    monkeypatch.setenv("ANNOTATION_PORT", "8000")
    with caplog.at_level(logging.WARNING):
        assert _bind_address() == ("annotator.example", 8000)
    assert "authentication" in caplog.text


def test_origin_normalizes_case_and_default_port() -> None:
    client, workflow = make_client("https://localhost")
    response = client.post(
        "/annotate",
        headers={"Origin": "HTTPS://LOCALHOST:443"},
        data={"candidate_id": "c-1", "label": "no"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert workflow.schedule_calls == 1


def test_forwarded_headers_do_not_authorize_external_origin() -> None:
    client, workflow = make_client()
    response = client.post(
        "/annotate",
        headers={
            "Origin": "https://evil.example",
            "X-Forwarded-Host": "evil.example",
            "X-Forwarded-Proto": "https",
            "Forwarded": "host=evil.example;proto=https",
        },
        data={"candidate_id": "c-1", "label": "no"},
    )
    assert response.status_code == 403
    assert workflow.calls == []


@pytest.mark.parametrize(
    "origin",
    [
        "\x00http://127.0.0.1:8000",
        "\x01http://127.0.0.1:8000",
        "http://127.0.0.1:8000?",
        "http://127.0.0.1:8000#",
    ],
)
def test_rejects_malformed_origins_that_urlsplit_would_normalize(origin: str) -> None:
    client, workflow = make_client()
    response = client.post(
        "/annotate", headers={"Origin": origin}, data={"candidate_id": "c-1", "label": "no"}
    )
    assert response.status_code == 403
    assert workflow.calls == []


@pytest.mark.parametrize("origin", ["http://localhost", "http://localhost:80"])
def test_http_default_port_is_equivalent_to_explicit_port(origin: str) -> None:
    client, workflow = make_client("http://localhost:80")
    response = client.post(
        "/annotate",
        headers={"Origin": origin},
        data={"candidate_id": "c-1", "label": "no"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert workflow.schedule_calls == 1


@pytest.mark.parametrize("host", [None, "0.0.0.0"])
def test_launcher_preserves_explicit_bind_and_disables_forwarded_header_trust(monkeypatch, host) -> None:
    import uvicorn

    import landuse_sentence_relevance.web.app as app_module

    workflow = FakeWorkflow(make_candidate(), [])
    captured = {}
    monkeypatch.delenv("ANNOTATION_HOST", raising=False)
    monkeypatch.setenv("ANNOTATION_PORT", "9000")
    if host is not None:
        monkeypatch.setenv("ANNOTATION_HOST", host)
    monkeypatch.setattr(app_module, "build_annotation_workflow", lambda: workflow)
    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.update(kwargs))

    app_module.run()

    assert captured["host"] == (host or "127.0.0.1")
    assert captured["port"] == 9000
    assert captured["proxy_headers"] is False


@pytest.mark.parametrize(
    "referer", ["\x01http://127.0.0.1:8000/", "http://127.0.0.1:8000/\\path", "http://127.0.0.1:8000/\tpath"]
)
def test_referer_rejects_control_characters_and_backslashes(referer: str) -> None:
    client, workflow = make_client()
    response = client.post(
        "/annotate", headers={"Referer": referer}, data={"candidate_id": "c-1", "label": "no"}
    )
    assert response.status_code == 403
    assert workflow.calls == []


def test_trusted_hosts_default_is_exact_and_blank_is_safe(monkeypatch) -> None:
    from landuse_sentence_relevance.web.security import trusted_hosts_from_env

    monkeypatch.delenv("ANNOTATION_TRUSTED_HOSTS", raising=False)
    assert trusted_hosts_from_env() == frozenset({"127.0.0.1", "localhost", "::1"})
    monkeypatch.setenv("ANNOTATION_TRUSTED_HOSTS", " ")
    assert trusted_hosts_from_env() == frozenset({"127.0.0.1", "localhost", "::1"})


def test_explicit_ipv6_and_mixed_case_hostnames_are_supported(monkeypatch) -> None:
    monkeypatch.setenv("ANNOTATION_TRUSTED_HOSTS", "[2001:db8::1], ANNOTATOR.example")
    client, _ = make_client()
    assert client.get("/health", headers={"Host": "[2001:db8::1]:8000"}).status_code == 200
    assert client.get("/health", headers={"Host": "annotator.example"}).status_code == 200


def test_security_denials_explain_the_requirement_without_echoing_headers() -> None:
    client, _ = make_client()
    response = client.get("/health", headers={"Host": "untrusted.example"})
    assert response.text == "Invalid Host header"
    response = client.post("/annotate", headers={"Origin": "https://untrusted.example"})
    assert response.text == "A same-origin Origin or Referer header is required"


@pytest.mark.parametrize("trusted_host", ["annotator.example", "XXXX"])
def test_explicit_remote_host_cannot_make_ambiguous_host_headers_valid(
    monkeypatch, trusted_host: str
) -> None:
    monkeypatch.setenv("ANNOTATION_TRUSTED_HOSTS", trusted_host)
    client, _ = make_client()
    response = client.get("/health", headers=[("Host", trusted_host), ("Host", "untrusted.example")])
    assert response.status_code == 400
