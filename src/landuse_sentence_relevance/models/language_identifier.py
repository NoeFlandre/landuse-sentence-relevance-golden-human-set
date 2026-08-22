from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LanguagePrediction:
    language: str
    confidence: float


class CommonLinguaIdentifier:
    """Apply conservative English filtering with the pinned CommonLingua model."""

    def __init__(
        self,
        predictor: Callable[[str], LanguagePrediction] | None = None,
        min_confidence: float = 0.90,
        max_input_bytes: int = 512,
        model_id: str = "PleIAs/CommonLingua",
        revision: str = "43fe88d75e94b11283b66daccbbe4a73e7bc1361",
        cache_dir: Path | None = None,
    ) -> None:
        _validate_limits(min_confidence, max_input_bytes)
        self._min_confidence = min_confidence
        self._max_input_bytes = max_input_bytes
        self._predictor = predictor or self._load_predictor(model_id, revision, cache_dir)

    @staticmethod
    def _load_predictor(  # pragma: no cover - streaming smoke checks model download
        model_id: str,
        revision: str,
        cache_dir: Path | None,
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
        predictor, _, _ = load_predictor(checkpoint_path)
        return lambda text: _prediction_from_tuple(predictor(text))

    def is_english(self, text: str) -> bool:
        if len(text.encode("utf-8", errors="replace")) > self._max_input_bytes:
            return False
        prediction = self._predictor(text)
        return prediction.language in {"eng", "en"} and prediction.confidence >= self._min_confidence


def _prediction_from_tuple(result: tuple[str, float]) -> LanguagePrediction:
    return LanguagePrediction(language=result[0], confidence=result[1])


def _validate_limits(min_confidence: float, max_input_bytes: int) -> None:
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    if max_input_bytes < 1:
        raise ValueError("max_input_bytes must be positive")
