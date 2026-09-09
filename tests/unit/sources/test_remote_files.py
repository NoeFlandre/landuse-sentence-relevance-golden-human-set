from typing import Any

import httpx
import pytest

from landuse_sentence_relevance.sources import remote_files
from landuse_sentence_relevance.sources.remote_files import (
    align_remote_files,
    pinned_remote_file_urls,
    select_evenly_spaced,
)


def test_select_evenly_spaced_keeps_sorted_endpoints() -> None:
    assert select_evenly_spaced(("a", "b", "c", "d", "e"), 3) == ("a", "c", "e")


def test_select_evenly_spaced_returns_all_available_files() -> None:
    assert select_evenly_spaced(("b", "a"), 3) == ("a", "b")


def test_select_evenly_spaced_can_select_one_file() -> None:
    assert select_evenly_spaced(("b", "a"), 1) == ("a",)


def test_select_evenly_spaced_rejects_non_positive_target() -> None:
    with pytest.raises(ValueError, match="target_count must be positive"):
        select_evenly_spaced(("a",), 0)


def test_align_remote_files_uses_only_shared_file_stems() -> None:
    groups = {
        "polygons": (
            "polygons/z.parquet",
            "polygons/a.parquet",
            "polygons/m.parquet",
        ),
        "links": (
            "links/m.parquet",
            "links/a.parquet",
            "links/z.parquet",
        ),
    }

    assert align_remote_files(groups, 2) == {
        "polygons": ("polygons/a.parquet", "polygons/z.parquet"),
        "links": ("links/a.parquet", "links/z.parquet"),
    }


def test_align_remote_files_rejects_too_few_shared_files() -> None:
    groups = {
        "polygons": ("polygons/a.parquet", "polygons/b.parquet"),
        "links": ("links/a.parquet",),
    }

    with pytest.raises(ValueError, match="need 2 shared parquet files, found 1"):
        align_remote_files(groups, 2)


def test_align_remote_files_rejects_duplicate_stems() -> None:
    groups = {
        "polygons": ("a/region.parquet", "b/region.parquet"),
        "links": ("links/region.parquet",),
    }

    with pytest.raises(ValueError, match="duplicate parquet stem"):
        align_remote_files(groups, 1)


def test_align_remote_files_rejects_no_groups() -> None:
    with pytest.raises(ValueError, match="at least one parquet file group"):
        align_remote_files({}, 1)


def test_pinned_remote_file_urls_lists_and_aligns_hub_files(monkeypatch) -> None:
    import huggingface_hub

    class FakeApi:
        def __init__(self, token=None):
            assert token == "token"

        def list_repo_tree(self, **kwargs):
            paths = {
                "polygons": ["polygons/b.parquet", {"path": "polygons/a.parquet"}],
                "links": ["links/b.parquet", {"path": "links/a.parquet"}],
            }
            assert kwargs["repo_id"] == "owner/dataset"
            return paths[kwargs["path_in_repo"]]

    def fake_hub_url(repo_id, filename, *, repo_type, revision):
        return f"https://example.test/{repo_id}/{revision}/{filename}?type={repo_type}"

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_url", fake_hub_url)

    assert pinned_remote_file_urls(
        "owner/dataset",
        "revision",
        {"polygons": "polygons", "links": "links"},
        1,
        token="token",
    ) == {
        "polygons": ("https://example.test/owner/dataset/revision/polygons/a.parquet?type=dataset",),
        "links": ("https://example.test/owner/dataset/revision/links/a.parquet?type=dataset",),
    }


def test_remote_file_catalog_times_out_a_stalled_tree_request(monkeypatch) -> None:
    import time

    monkeypatch.setattr(remote_files, "_TREE_LIST_TIMEOUT_SECONDS", 0.01)

    def stalled_tree_lister(**kwargs):
        time.sleep(0.1)
        return ()

    catalog = remote_files.RemoteParquetCatalog(stalled_tree_lister, lambda repo_id, path, revision: path)

    with pytest.raises(remote_files.RemoteMetadataTimeoutError, match="timed out listing"):
        catalog.urls("owner/dataset", "revision", {"polygons": "polygons"}, 1)


def test_bounded_huggingface_client_retries_a_transport_timeout(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def flaky_get(self, url, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise httpx.ConnectTimeout("temporary handshake failure")
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.Client, "get", flaky_get)
    client = remote_files._BoundedHuggingFaceClient()
    try:
        response = client.get("https://example.test")
    finally:
        client.close()

    assert response.status_code == 200
    assert len(calls) == 2
    assert all(call["timeout"] == remote_files._TREE_LIST_TIMEOUT_SECONDS for call in calls)
