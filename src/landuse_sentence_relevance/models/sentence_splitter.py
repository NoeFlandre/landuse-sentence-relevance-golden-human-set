from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
DEFAULT_MAX_INPUT_CHARACTERS = 2_000
_SENTENCE_BOUNDARY_MARKERS = tuple(f"{punctuation}{space}" for punctuation in ".!?" for space in " \t\n")
_WHITESPACE_MARKERS = (" ", "\t", "\n")
BatchTextSplitter = Callable[[tuple[str, ...]], tuple[tuple[str, ...], ...]]
_MODEL_BATCH_SIZE = 32


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
        max_input_characters: int = DEFAULT_MAX_INPUT_CHARACTERS,
        batch_size: int = 32,
        max_workers: int = 1,
    ) -> None:
        if max_input_characters < 1:
            raise ValueError("max_input_characters must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self._model = model or self._load_model(
            model_id=model_id,
            revision=revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            cache_dir=cache_dir,
        )
        self._max_input_characters = max_input_characters
        self._batch_size = batch_size
        self._max_workers = max_workers

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
        chunks = _bounded_chunks(text, self._max_input_characters)
        if len(chunks) > 1:
            logger.info(
                "SaT splitter: chunking %d input characters into %d chunks (max %d characters)",
                len(text),
                len(chunks),
                self._max_input_characters,
            )
        segments: list[str] = []
        for chunk in chunks:
            segments.extend(self._model.split(chunk, strip_whitespace=True, batch_size=_MODEL_BATCH_SIZE))
        return tuple(segment.strip() for segment in segments if segment.strip())

    def split_many(self, texts: Iterable[str]) -> tuple[tuple[str, ...], ...]:
        text_items = tuple(texts)
        if not text_items:
            return ()
        batches = _text_batches(text_items, self._batch_size)
        split_batches = _split_batches(batches, self._split_batch, self._max_workers)
        return tuple(segments for batch in split_batches for segments in batch)

    def _split_batch(self, texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        if len(texts) == 1:
            return (self.split(texts[0]),)
        chunks = _batched_input_chunks(texts, self._max_input_characters)
        return _group_batched_segments(texts, chunks, self._model)


def _split_batches(
    batches: tuple[tuple[str, ...], ...],
    split_batch: BatchTextSplitter,
    max_workers: int,
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    if max_workers == 1 or len(batches) == 1:
        return tuple(split_batch(batch) for batch in batches)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as executor:
        return tuple(executor.map(split_batch, batches))


def _text_batches(texts: tuple[str, ...], batch_size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(texts[index : index + batch_size] for index in range(0, len(texts), batch_size))


def _group_batched_segments(
    texts: tuple[str, ...],
    chunks: tuple[tuple[int, str], ...],
    model: Any,
) -> tuple[tuple[str, ...], ...]:
    grouped: list[list[str]] = [[] for _ in texts]
    model_segments = model.split(
        [chunk for _, chunk in chunks],
        strip_whitespace=True,
        batch_size=_MODEL_BATCH_SIZE,
    )
    for (text_index, _), segments in zip(chunks, model_segments, strict=True):
        grouped[text_index].extend(_clean_segments(segments))
    return tuple(tuple(segments) for segments in grouped)


def _clean_segments(segments: Iterable[str]) -> tuple[str, ...]:
    return tuple(segment.strip() for segment in segments if segment.strip())


def _batched_input_chunks(
    texts: tuple[str, ...],
    max_input_characters: int,
) -> tuple[tuple[int, str], ...]:
    chunks: list[tuple[int, str]] = []
    for text_index, text in enumerate(texts):
        text_chunks = _bounded_chunks(text, max_input_characters)
        if len(text_chunks) > 1:
            logger.info(
                "SaT splitter: chunking %d input characters into %d chunks (max %d characters)",
                len(text),
                len(text_chunks),
                max_input_characters,
            )
        chunks.extend((text_index, chunk) for chunk in text_chunks)
    return tuple(chunks)


def _bounded_chunks(text: str, max_characters: int) -> tuple[str, ...]:
    if len(text) <= max_characters:
        return (text,)

    chunks: list[str] = []
    start = 0
    while len(text) - start > max_characters:
        cut = _chunk_boundary(text, start, max_characters)
        chunks.append(text[start : start + cut])
        start = _skip_whitespace(text, start + cut)
    if start < len(text):
        chunks.append(text[start:])
    return tuple(chunks)


def _chunk_boundary(text: str, start: int, max_characters: int) -> int:
    window_end = start + max_characters
    if text[window_end].isspace():
        return max_characters
    return _boundary_from_markers(text, start, window_end, max_characters)


def _boundary_from_markers(text: str, start: int, window_end: int, max_characters: int) -> int:
    sentence_boundary = _last_marker(text, start, window_end + 1, _SENTENCE_BOUNDARY_MARKERS)
    if sentence_boundary >= start:
        return sentence_boundary - start + 1
    whitespace = _last_marker(text, start, window_end, _WHITESPACE_MARKERS)
    return _relative_boundary(whitespace, start, max_characters)


def _relative_boundary(boundary: int, start: int, fallback: int) -> int:
    return boundary - start + 1 if boundary >= start else fallback


def _last_marker(text: str, start: int, end: int, markers: tuple[str, ...]) -> int:
    return max((text.rfind(marker, start, end) for marker in markers), default=-1)


def _skip_whitespace(text: str, start: int) -> int:
    while start < len(text) and text[start].isspace():
        start += 1
    return start
