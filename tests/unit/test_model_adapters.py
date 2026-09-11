import sys
from types import ModuleType

from landuse_sentence_relevance.models.language_identifier import (
    CommonLinguaIdentifier,
    LanguagePrediction,
    _LoadedPredictor,
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
    assert calls == [("input", {"strip_whitespace": True, "batch_size": 32})]


def test_sentence_splitter_bounds_long_inputs_before_model_inference() -> None:
    calls = []

    class FakeModel:
        def split(self, text: str, **kwargs):
            calls.append((text, kwargs))
            return [text]

    splitter = SaTSentenceSplitter(model=FakeModel(), max_input_characters=10)

    assert splitter.split("alpha beta gamma") == ("alpha beta", "gamma")
    assert calls == [
        ("alpha beta", {"strip_whitespace": True, "batch_size": 32}),
        ("gamma", {"strip_whitespace": True, "batch_size": 32}),
    ]


def test_sentence_splitter_batches_multiple_inputs_through_the_model() -> None:
    calls = []

    class FakeModel:
        def split(self, texts, **kwargs):
            calls.append((tuple(texts), kwargs))
            return ([f" {text} "] for text in texts)

    splitter = SaTSentenceSplitter(model=FakeModel())

    assert splitter.split_many(("first", "second")) == (("first",), ("second",))
    assert calls == [(("first", "second"), {"strip_whitespace": True, "batch_size": 32})]


def test_sentence_splitter_parallelizes_sub_batches_without_reordering_results() -> None:
    calls = []

    class FakeModel:
        def split(self, text, **kwargs):
            assert isinstance(text, str)
            calls.append(text)
            return [f" {text} sentence. "]

    splitter = SaTSentenceSplitter(model=FakeModel(), batch_size=1, max_workers=2)

    assert splitter.split_many(("first", "second", "third")) == (
        ("first sentence.",),
        ("second sentence.",),
        ("third sentence.",),
    )
    assert sorted(calls) == ["first", "second", "third"]


def test_sentence_splitter_loads_the_base_sat_model_without_an_adapter(monkeypatch) -> None:
    import huggingface_hub

    calls = []

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: "tokenizer-path")

    class FakeSaT:
        def __init__(self, model_name: str, **kwargs) -> None:
            calls.append((model_name, kwargs))

    fake_wtpsplit = ModuleType("wtpsplit")
    fake_wtpsplit.__dict__["SaT"] = FakeSaT
    monkeypatch.setitem(sys.modules, "wtpsplit", fake_wtpsplit)

    SaTSentenceSplitter._load_model(
        model_id="segment-any-text/sat-12l-sm",
        revision="model-revision",
        tokenizer_id="FacebookAI/xlm-roberta-base",
        tokenizer_revision="tokenizer-revision",
        cache_dir=None,
    )

    assert calls[0][0] == "sat-12l-sm"
    assert calls[0][1]["tokenizer_name_or_path"] == "tokenizer-path"
    assert "language" not in calls[0][1]
    assert "style_or_domain" not in calls[0][1]


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


def test_common_lingua_batches_predictions_without_changing_the_filter() -> None:
    calls: list[tuple[str, ...]] = []

    def batch_predictor(texts: tuple[str, ...]) -> tuple[LanguagePrediction, ...]:
        calls.append(texts)
        return tuple(LanguagePrediction(language="eng", confidence=1.0) for _ in texts)

    identifier = CommonLinguaIdentifier(
        predictor=lambda text: LanguagePrediction(language="fra", confidence=1.0),
        batch_predictor=batch_predictor,
        max_input_bytes=10,
    )

    assert identifier.is_english_many(("one", "two", "three")) == (True, True, True)
    assert calls == [("one", "two", "three")]


def test_common_lingua_rejects_oversized_inputs_in_a_batch() -> None:
    calls: list[tuple[str, ...]] = []

    def batch_predictor(texts: tuple[str, ...]) -> tuple[LanguagePrediction, ...]:
        calls.append(texts)
        return tuple(LanguagePrediction(language="eng", confidence=1.0) for _ in texts)

    identifier = CommonLinguaIdentifier(
        predictor=lambda text: LanguagePrediction(language="fra", confidence=1.0),
        batch_predictor=batch_predictor,
        max_input_bytes=4,
    )

    assert identifier.is_english_many(("ok", "longer")) == (True, False)
    assert calls == [("ok",)]


def test_common_lingua_uses_model_friendly_batches() -> None:
    calls: list[tuple[str, ...]] = []

    def batch_predictor(texts: tuple[str, ...]) -> tuple[LanguagePrediction, ...]:
        calls.append(texts)
        return tuple(LanguagePrediction(language="eng", confidence=1.0) for _ in texts)

    identifier = CommonLinguaIdentifier(
        predictor=lambda text: LanguagePrediction(language="fra", confidence=1.0),
        batch_predictor=batch_predictor,
    )

    assert identifier.is_english_many(tuple(str(index) for index in range(33))) == (True,) * 33
    assert tuple(map(len, calls)) == (32, 1)


def test_loaded_predictor_adapts_scalar_and_batch_predictions() -> None:
    class RawPredictor:
        def __call__(self, text: str) -> tuple[str, float]:
            return "eng", 0.95

        def predict_many(self, texts: tuple[str, ...]) -> tuple[tuple[str, float], ...]:
            return tuple(("eng", 0.95) for _ in texts)

    predictor = _LoadedPredictor(RawPredictor())

    assert predictor("one") == LanguagePrediction("eng", 0.95)
    assert predictor.predict_many(("one", "two")) == (
        LanguagePrediction("eng", 0.95),
        LanguagePrediction("eng", 0.95),
    )


def test_loaded_predictor_falls_back_to_scalar_predictions() -> None:
    predictor = _LoadedPredictor(lambda text: ("fra", 0.95))

    assert predictor.predict_many(("one", "two")) == (
        LanguagePrediction("fra", 0.95),
        LanguagePrediction("fra", 0.95),
    )


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
    with pytest.raises(ValueError, match="max_input_characters"):
        SaTSentenceSplitter(model=object(), max_input_characters=0)
    with pytest.raises(ValueError, match="batch_size"):
        SaTSentenceSplitter(model=object(), batch_size=0)
    with pytest.raises(ValueError, match="max_workers"):
        SaTSentenceSplitter(model=object(), max_workers=0)
    assert _prediction_from_tuple(("eng", 0.91)) == LanguagePrediction("eng", 0.91)


def test_model_adapters_can_be_initialized_through_injected_loaders(monkeypatch) -> None:
    monkeypatch.setattr(
        CommonLinguaIdentifier,
        "_load_predictor",
        staticmethod(
            lambda model_id, revision, cache_dir, device: lambda text: LanguagePrediction("eng", 1.0)
        ),
    )
    identifier = CommonLinguaIdentifier()
    assert identifier.is_english("English") is True

    monkeypatch.setattr(SaTSentenceSplitter, "_load_model", staticmethod(lambda **kwargs: object()))
    assert SaTSentenceSplitter(model=None)._model is not None
