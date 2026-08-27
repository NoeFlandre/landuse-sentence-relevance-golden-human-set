from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol


class HuggingFaceDatasetLoader(Protocol):
    """Load one immutable Hugging Face split in streaming mode."""

    def __call__(
        self,
        *,
        path: str,
        name: str,
        split: str,
        streaming: Literal[True],
        revision: str,
    ) -> Iterable[Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class HuggingFaceRowConfig:
    dataset_id: str
    revision: str
    split: str
    config: str


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
            name: str,
            split: str,
            streaming: Literal[True],
            revision: str,
        ) -> Iterable[Mapping[str, Any]]:
            return load_dataset(
                path=path,
                name=name,
                split=split,
                streaming=streaming,
                revision=revision,
                on_bad_files="warn",
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
        return self._loader(
            path=config.dataset_id,
            name=config.config,
            split=config.split,
            streaming=True,
            revision=config.revision,
        )
