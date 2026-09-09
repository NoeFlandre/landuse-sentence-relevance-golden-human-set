from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from threading import Thread
from typing import Any

import httpx

TreeLister = Callable[..., Iterable[object]]
UrlBuilder = Callable[[str, str, str], str]
_TREE_LIST_TIMEOUT_SECONDS = 20.0
_TREE_LIST_RETRIES = 1
logger = logging.getLogger(__name__)


class RemoteMetadataTimeoutError(RuntimeError):
    """Raised when a remote Parquet catalog request does not complete promptly."""


def select_evenly_spaced(paths: Iterable[str], target_count: int) -> tuple[str, ...]:
    """Select deterministic, approximately uniform entries from sorted paths."""
    _require_target_count(target_count)
    available = tuple(sorted(set(paths)))
    if len(available) <= target_count:
        return available
    if target_count == 1:
        return (available[0],)
    last_index = len(available) - 1
    return tuple(available[index * last_index // (target_count - 1)] for index in range(target_count))


def align_remote_files(
    groups: Mapping[str, Iterable[str]],
    target_count: int,
) -> dict[str, tuple[str, ...]]:
    """Select the same deterministic parquet stems from every remote file group."""
    _require_target_count(target_count)
    indexed = _index_groups(groups)
    shared_stems = _shared_stems(indexed)
    _require_shared_stems(shared_stems, target_count)
    selected_stems = select_evenly_spaced(shared_stems, target_count)
    return _select_group_paths(indexed, selected_stems)


@dataclass(frozen=True, slots=True)
class RemoteParquetCatalog:
    """List and align remote Parquet files without reading their rows."""

    tree_lister: TreeLister
    url_builder: UrlBuilder

    def urls(
        self,
        dataset_id: str,
        revision: str,
        directories: Mapping[str, str],
        target_count: int,
        token: str | None = None,
    ) -> dict[str, tuple[str, ...]]:
        groups = {
            name: self._list_parquet_paths(dataset_id, revision, directory, token)
            for name, directory in directories.items()
        }
        aligned = align_remote_files(groups, target_count)
        return {
            name: tuple(self.url_builder(dataset_id, path, revision) for path in paths)
            for name, paths in aligned.items()
        }

    def _list_parquet_paths(
        self,
        dataset_id: str,
        revision: str,
        directory: str,
        token: str | None,
    ) -> tuple[str, ...]:
        entries = _list_tree_with_retry(
            self.tree_lister,
            dataset_id=dataset_id,
            directory=directory,
            revision=revision,
            token=token,
        )
        return tuple(
            path
            for entry in entries
            if (path := _entry_path(entry)) is not None and PurePosixPath(path).suffix == ".parquet"
        )


def _list_tree_with_retry(
    tree_lister: TreeLister,
    *,
    dataset_id: str,
    directory: str,
    revision: str,
    token: str | None,
) -> tuple[object, ...]:
    for attempt in range(_TREE_LIST_RETRIES + 1):
        try:
            return _list_tree_with_timeout(
                tree_lister,
                dataset_id=dataset_id,
                directory=directory,
                revision=revision,
                token=token,
            )
        except RemoteMetadataTimeoutError:
            if attempt == _TREE_LIST_RETRIES:
                raise
            logger.warning(
                "Hugging Face tree listing timed out; retrying %s/%s",
                dataset_id,
                directory,
            )
    raise AssertionError("remote tree listing retry loop did not return")  # pragma: no cover


def _list_tree_with_timeout(
    tree_lister: TreeLister,
    *,
    dataset_id: str,
    directory: str,
    revision: str,
    token: str | None,
) -> tuple[object, ...]:
    results: list[tuple[object, ...]] = []
    failures: list[Exception] = []

    def collect() -> None:
        try:
            results.append(
                tuple(
                    tree_lister(
                        repo_id=dataset_id,
                        path_in_repo=directory,
                        recursive=False,
                        expand=False,
                        revision=revision,
                        repo_type="dataset",
                        token=token,
                    )
                )
            )
        except Exception as error:  # pragma: no cover - exercised by remote failures
            failures.append(error)

    worker = Thread(target=collect, name="hf-tree-list", daemon=True)
    worker.start()
    worker.join(_TREE_LIST_TIMEOUT_SECONDS)
    if worker.is_alive():
        raise RemoteMetadataTimeoutError(
            f"timed out listing {dataset_id}/{directory} after {_TREE_LIST_TIMEOUT_SECONDS:g}s"
        )
    if failures:
        raise failures[0]
    return results[0] if results else ()


class _BoundedHuggingFaceClient(httpx.Client):
    """Use IPv4 and a finite timeout for Hugging Face requests."""

    def __init__(self) -> None:
        super().__init__(
            follow_redirects=True,
            timeout=_TREE_LIST_TIMEOUT_SECONDS,
            transport=httpx.HTTPTransport(local_address="0.0.0.0"),
        )

    def get(self, url: httpx.URL | str, **kwargs: Any) -> httpx.Response:
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = _TREE_LIST_TIMEOUT_SECONDS
        for attempt in range(_TREE_LIST_RETRIES + 1):
            try:
                return super().get(url, **kwargs)
            except httpx.TransportError:
                if attempt == _TREE_LIST_RETRIES:
                    raise
                logger.warning("Hugging Face request failed; retrying once")
        raise AssertionError("Hugging Face request retry loop did not return")  # pragma: no cover


def pinned_remote_file_urls(
    dataset_id: str,
    revision: str,
    directories: Mapping[str, str],
    target_count: int,
    token: str | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return uniformly sampled, revision-pinned Hugging Face Parquet URLs."""
    try:
        from huggingface_hub import HfApi, hf_hub_url, set_client_factory
    except ImportError as error:  # pragma: no cover - dependency installation boundary
        raise RuntimeError("Install project dependencies with `uv sync`") from error

    set_client_factory(_BoundedHuggingFaceClient)
    api = HfApi(token=token)
    return RemoteParquetCatalog(
        api.list_repo_tree,
        lambda repo_id, path, file_revision: hf_hub_url(
            repo_id,
            path,
            repo_type="dataset",
            revision=file_revision,
        ),
    ).urls(dataset_id, revision, directories, target_count, token)


def _index_stems(paths: Iterable[str]) -> dict[str, str]:
    indexed: dict[str, str] = {}
    for path in sorted(set(paths)):
        if PurePosixPath(path).suffix != ".parquet":
            continue
        stem = PurePosixPath(path).stem
        if stem in indexed:
            raise ValueError(f"duplicate parquet stem: {stem}")
        indexed[stem] = path
    return indexed


def _index_groups(groups: Mapping[str, Iterable[str]]) -> dict[str, dict[str, str]]:
    if not groups:
        raise ValueError("at least one parquet file group is required")
    return {name: _index_stems(paths) for name, paths in groups.items()}


def _shared_stems(indexed: Mapping[str, Mapping[str, str]]) -> set[str]:
    return set.intersection(*(set(paths) for paths in indexed.values()))


def _require_shared_stems(shared_stems: set[str], target_count: int) -> None:
    if len(shared_stems) < target_count:
        raise ValueError(f"need {target_count} shared parquet files, found {len(shared_stems)}")


def _select_group_paths(
    indexed: Mapping[str, Mapping[str, str]],
    selected_stems: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    return {name: tuple(paths[stem] for stem in selected_stems) for name, paths in indexed.items()}


def _entry_path(entry: object) -> str | None:
    if isinstance(entry, str):
        return entry
    path = entry.get("path") if isinstance(entry, Mapping) else getattr(entry, "path", None)
    return path if isinstance(path, str) else None


def _require_target_count(target_count: int) -> None:
    if target_count < 1:
        raise ValueError("target_count must be positive")
