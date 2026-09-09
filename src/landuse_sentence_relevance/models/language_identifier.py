from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LanguagePrediction:
    language: str
    confidence: float


BatchPredictor = Callable[[tuple[str, ...]], Iterable[LanguagePrediction]]
_BATCH_SIZE = 32


class _LoadedPredictor:
    def __init__(self, predictor: Any) -> None:
        self._predictor = predictor
        self._batch_predictor = getattr(predictor, "predict_many", None)

    def __call__(self, text: str) -> LanguagePrediction:
        return _prediction_from_tuple(self._predictor(text))

    def predict_many(self, texts: tuple[str, ...]) -> tuple[LanguagePrediction, ...]:
        if self._batch_predictor is None:
            return tuple(self(text) for text in texts)
        return tuple(_prediction_from_tuple(result) for result in self._batch_predictor(texts))


class CommonLinguaIdentifier:
    """Apply conservative English filtering with the pinned CommonLingua model."""

    def __init__(
        self,
        predictor: Callable[[str], LanguagePrediction] | None = None,
        batch_predictor: BatchPredictor | None = None,
        min_confidence: float = 0.90,
        max_input_bytes: int = 512,
        model_id: str = "PleIAs/CommonLingua",
        revision: str = "43fe88d75e94b11283b66daccbbe4a73e7bc1361",
        cache_dir: Path | None = None,
        device: str = "auto",
    ) -> None:
        _validate_limits(min_confidence, max_input_bytes)
        self._min_confidence = min_confidence
        self._max_input_bytes = max_input_bytes
        self._predictor = predictor or self._load_predictor(model_id, revision, cache_dir, device)
        self._batch_predictor = batch_predictor or getattr(self._predictor, "predict_many", None)

    @staticmethod
    def _load_predictor(  # pragma: no cover - streaming smoke checks model download
        model_id: str,
        revision: str,
        cache_dir: Path | None,
        device: str,
    ) -> Callable[[str], LanguagePrediction]:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:  # pragma: no cover - optional model extra
            raise RuntimeError("Install model support with `uv sync --extra models`") from error
        from landuse_sentence_relevance.models.commonlingua_model import load_predictor

        checkpoint_path = hf_hub_download(
            repo_id=model_id,
            filename="model.pt",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        predictor, _, _ = load_predictor(checkpoint_path, device=device)
        return _LoadedPredictor(predictor)

    def is_english(self, text: str) -> bool:
        if not _within_input_limit(text, self._max_input_bytes):
            return False
        prediction = self._predictor(text)
        return _is_confident_english(prediction, self._min_confidence)

    def is_english_many(self, texts: Iterable[str]) -> tuple[bool, ...]:
        text_items = tuple(texts)
        results = [False] * len(text_items)
        eligible = _eligible_items(text_items, self._max_input_bytes)
        for batch in _batches(eligible, _BATCH_SIZE):
            self._set_batch_results(results, batch)
        return tuple(results)

    def _set_batch_results(
        self,
        results: list[bool],
        batch: tuple[tuple[int, str], ...],
    ) -> None:
        predictions = self._predict_batch(tuple(text for _, text in batch))
        for (index, _), prediction in zip(batch, predictions, strict=True):
            results[index] = _is_confident_english(prediction, self._min_confidence)

    def _predict_batch(self, texts: tuple[str, ...]) -> tuple[LanguagePrediction, ...]:
        if self._batch_predictor is None:
            return tuple(self._predictor(text) for text in texts)
        return tuple(self._batch_predictor(texts))


def _prediction_from_tuple(result: tuple[str, float]) -> LanguagePrediction:
    return LanguagePrediction(language=result[0], confidence=result[1])


def _within_input_limit(text: str, max_input_bytes: int) -> bool:
    return len(text.encode("utf-8", errors="replace")) <= max_input_bytes


def _is_confident_english(prediction: LanguagePrediction, min_confidence: float) -> bool:
    return prediction.language in {"eng", "en"} and prediction.confidence >= min_confidence


def _batches(
    values: tuple[tuple[int, str], ...],
    size: int,
) -> Iterable[tuple[tuple[int, str], ...]]:
    return (values[index : index + size] for index in range(0, len(values), size))


def _eligible_items(
    texts: tuple[str, ...],
    max_input_bytes: int,
) -> tuple[tuple[int, str], ...]:
    return tuple(
        (index, text) for index, text in enumerate(texts) if _within_input_limit(text, max_input_bytes)
    )


def _validate_limits(min_confidence: float, max_input_bytes: int) -> None:
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    if max_input_bytes < 1:
        raise ValueError("max_input_bytes must be positive")
