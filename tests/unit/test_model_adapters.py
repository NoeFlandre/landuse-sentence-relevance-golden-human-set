from landuse_sentence_relevance.models.language_identifier import (
    CommonLinguaIdentifier,
    LanguagePrediction,
    _prediction_from_tuple,
)
from landuse_sentence_relevance.models.sentence_splitter import SaTSentenceSplitter


def test_sentence_splitter_strips_model_segments() -> None:
    calls = []

    class FakeModel:
        def split(self, text: str, **kwargs):
            calls.append((text, kwargs))
            return [" First. ", "", "Second.\n"]

    splitter = SaTSentenceSplitter(model=FakeModel())

    assert splitter.split("input") == ("First.", "Second.")
    assert calls == [("input", {"lang_code": "en", "strip_whitespace": True})]


def test_common_lingua_accepts_only_confident_english_sentences() -> None:
    predictions = {
        "english": LanguagePrediction(language="eng", confidence=0.95),
        "weak": LanguagePrediction(language="eng", confidence=0.89),
        "french": LanguagePrediction(language="fra", confidence=0.99),
    }
    identifier = CommonLinguaIdentifier(
        predictor=lambda text: predictions[text],
        min_confidence=0.90,
    )

    assert identifier.is_english("english") is True
    assert identifier.is_english("weak") is False
    assert identifier.is_english("french") is False


def test_common_lingua_rejects_sentences_longer_than_model_input() -> None:
    identifier = CommonLinguaIdentifier(
        predictor=lambda text: LanguagePrediction(language="eng", confidence=1.0),
        min_confidence=0.90,
        max_input_bytes=4,
    )

    assert identifier.is_english("short") is False


def test_model_adapters_validate_runtime_limits() -> None:
    import pytest

    with pytest.raises(ValueError, match="min_confidence"):
        CommonLinguaIdentifier(predictor=lambda text: LanguagePrediction("eng", 1.0), min_confidence=2)
    with pytest.raises(ValueError, match="max_input_bytes"):
        CommonLinguaIdentifier(predictor=lambda text: LanguagePrediction("eng", 1.0), max_input_bytes=0)
    assert _prediction_from_tuple(("eng", 0.91)) == LanguagePrediction("eng", 0.91)


def test_model_adapters_can_be_initialized_through_injected_loaders(monkeypatch) -> None:
    monkeypatch.setattr(
        CommonLinguaIdentifier,
        "_load_predictor",
        staticmethod(lambda model_id, revision, cache_dir: lambda text: LanguagePrediction("eng", 1.0)),
    )
    identifier = CommonLinguaIdentifier()
    assert identifier.is_english("English") is True

    monkeypatch.setattr(SaTSentenceSplitter, "_load_model", staticmethod(lambda **kwargs: object()))
    assert SaTSentenceSplitter(model=None)._model is not None
