from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any


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
        loader: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
    ) -> None:
        self._config = config
        self._loader = loader or self._default_loader()

    @staticmethod
    def _default_loader() -> Callable[..., Iterable[Mapping[str, Any]]]:
        try:
            from datasets import load_dataset
        except ImportError as error:  # pragma: no cover - exercised only in a missing extra environment
            raise RuntimeError(
                "Install the project dependencies with `uv sync` to stream Hugging Face data"
            ) from error
        return load_dataset

    def __call__(self) -> Iterable[Mapping[str, Any]]:
        config = self._config
        return self._loader(
            path=config.dataset_id,
            name=config.config,
            split=config.split,
            streaming=True,
            revision=config.revision,
        )
