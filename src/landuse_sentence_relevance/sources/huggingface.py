from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable


class HuggingFaceDatasetLoader(Protocol):
    """Load one immutable Hugging Face split in streaming mode."""

    def __call__(
        self,
        *,
        path: str,
        name: str | None,
        split: str,
        streaming: Literal[True],
        revision: str | None = None,
        columns: list[str] | None = None,
        data_files: dict[str, list[str]] | None = None,
    ) -> Iterable[Mapping[str, Any]]: ...


@runtime_checkable
class ShardableRows(Protocol):
    """Expose contiguous remote shards without loading their rows locally."""

    def shard(
        self,
        num_shards: int,
        index: int,
        contiguous: bool = True,
    ) -> Iterable[Mapping[str, Any]]: ...


@runtime_checkable
class ColumnSelectableRows(Protocol):
    """Project a streaming dataset before its rows are read."""

    def select_columns(self, column_names: list[str]) -> Iterable[Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class HuggingFaceRowConfig:
    dataset_id: str
    revision: str
    split: str
    config: str
    columns: tuple[str, ...] | None = None
    remote_files: tuple[str, ...] | None = None


class HuggingFaceDatasetRows:
    """Expose a Hugging Face split as a streaming-only iterable."""

    def __init__(
        self,
        config: HuggingFaceRowConfig,
        loader: HuggingFaceDatasetLoader | None = None,
    ) -> None:
        self._config = config
        self._loader = loader or self._default_loader()

    @staticmethod
    def _default_loader() -> HuggingFaceDatasetLoader:
        try:
            from datasets import load_dataset
        except ImportError as error:  # pragma: no cover - exercised only in a missing extra environment
            raise RuntimeError(
                "Install the project dependencies with `uv sync` to stream Hugging Face data"
            ) from error

        def stream_dataset(
            *,
            path: str,
            name: str | None,
            split: str,
            streaming: Literal[True],
            revision: str | None = None,
            columns: list[str] | None = None,
            data_files: dict[str, list[str]] | None = None,
        ) -> Iterable[Mapping[str, Any]]:
            kwargs: dict[str, Any] = {
                "path": path,
                "name": name,
                "split": split,
                "streaming": streaming,
                "on_bad_files": "warn",
            }
            if revision is not None:
                kwargs["revision"] = revision
            if columns is not None:
                kwargs["columns"] = columns
            if data_files is not None:
                kwargs["data_files"] = data_files
            return load_dataset(
                **kwargs,
            )

        return stream_dataset

    def __call__(self) -> Iterable[Mapping[str, Any]]:
        config = self._config
        logging.getLogger(__name__).info(
            "Opening Hugging Face stream: %s (config=%s, split=%s, revision=%s)",
            config.dataset_id,
            config.config,
            config.split,
            config.revision,
        )
        stream = self._loader(**_load_kwargs(config))
        if config.columns is None or not isinstance(stream, ColumnSelectableRows):
            return stream
        return stream.select_columns(list(config.columns))

    def shards(self, shard_count: int | None = None) -> tuple[Iterable[Mapping[str, Any]], ...]:
        """Partition one streaming dataset into bounded contiguous remote shards."""
        _validate_shard_count(shard_count)
        stream = self()
        if not isinstance(stream, ShardableRows):
            return (stream,)
        return _partition_stream(stream, _actual_shard_count(stream, shard_count))


def _load_kwargs(config: HuggingFaceRowConfig) -> dict[str, Any]:
    if config.remote_files is None:
        kwargs: dict[str, Any] = {
            "path": config.dataset_id,
            "name": config.config,
            "split": config.split,
            "streaming": True,
            "revision": config.revision,
        }
    else:
        kwargs = {
            "path": "parquet",
            "name": None,
            "data_files": {config.split: list(config.remote_files)},
            "split": config.split,
            "streaming": True,
        }
    if config.columns is not None:
        kwargs["columns"] = list(config.columns)
    return kwargs


def _validate_shard_count(shard_count: int | None) -> None:
    if shard_count is not None and shard_count < 1:
        raise ValueError("shard_count must be positive")


def _actual_shard_count(stream: ShardableRows, shard_count: int | None) -> int:
    return shard_count if shard_count is not None else int(getattr(stream, "n_shards", 1))


def _partition_stream(
    stream: ShardableRows,
    shard_count: int,
) -> tuple[Iterable[Mapping[str, Any]], ...]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    return tuple(stream.shard(shard_count, index, contiguous=True) for index in range(shard_count))
