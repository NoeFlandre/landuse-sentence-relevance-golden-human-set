from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


class SaTSentenceSplitter:
    """Use the pinned SaT-12L-sm model and never persist input source text."""

    def __init__(
        self,
        model: Any | None = None,
        model_id: str = "segment-any-text/sat-12l-sm",
        revision: str = "d70c72a9331b2d5a9e82baad00c64964a23a09bb",
        tokenizer_id: str = "FacebookAI/xlm-roberta-base",
        tokenizer_revision: str = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        cache_dir: Path | None = None,
    ) -> None:
        self._model = model or self._load_model(
            model_id=model_id,
            revision=revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            cache_dir=cache_dir,
        )

    @staticmethod
    def _load_model(  # pragma: no cover - model download integration is checked by the streaming smoke gate
        model_id: str,
        revision: str,
        tokenizer_id: str,
        tokenizer_revision: str,
        cache_dir: Path | None,
    ) -> Any:
        try:
            from huggingface_hub import snapshot_download
            from wtpsplit import SaT
        except ImportError as error:  # pragma: no cover - optional model extra
            raise RuntimeError("Install model support with `uv sync --extra models`") from error

        model_cache = str(cache_dir) if cache_dir is not None else None
        tokenizer_path = snapshot_download(
            repo_id=tokenizer_id,
            revision=tokenizer_revision,
            cache_dir=model_cache,
            allow_patterns=["*.json", "*.model", "*.txt"],
        )
        return SaT(
            model_id.rsplit("/", 1)[-1],
            tokenizer_name_or_path=tokenizer_path,
            from_pretrained_kwargs={"revision": revision, "cache_dir": model_cache},
            ort_providers=["CPUExecutionProvider"],
            hub_prefix=model_id.rsplit("/", 1)[0],
        )

    def split(self, text: str) -> tuple[str, ...]:
        segments: Iterable[str] = self._model.split(text, lang_code="en", strip_whitespace=True)
        return tuple(segment.strip() for segment in segments if segment.strip())
