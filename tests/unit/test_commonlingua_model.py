from pathlib import Path
from types import SimpleNamespace

import pytest

from landuse_sentence_relevance.config import Settings


def test_commonlingua_device_resolution_prefers_available_mps() -> None:
    from landuse_sentence_relevance.models.commonlingua_model import _resolve_device

    mps_torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
    cpu_torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))

    assert _resolve_device(mps_torch, "auto") == "mps"
    assert _resolve_device(cpu_torch, "auto") == "cpu"
    assert _resolve_device(mps_torch, "cpu") == "cpu"


def test_commonlingua_device_resolution_rejects_unavailable_or_unknown_devices() -> None:
    from landuse_sentence_relevance.models.commonlingua_model import _resolve_device

    cpu_torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))

    with pytest.raises(ValueError, match="device"):
        _resolve_device(cpu_torch, "cuda")
    with pytest.raises(RuntimeError, match="MPS"):
        _resolve_device(cpu_torch, "mps")


def test_cpu_execution_uses_one_torch_thread_for_small_inference_batches() -> None:
    from landuse_sentence_relevance.models.commonlingua_model import _configure_execution

    calls: list[int] = []
    torch = SimpleNamespace(set_num_threads=calls.append)

    _configure_execution(torch, "cpu")

    assert calls == [1]


def test_mps_execution_does_not_change_cpu_threading() -> None:
    from landuse_sentence_relevance.models.commonlingua_model import _configure_execution

    calls: list[int] = []
    torch = SimpleNamespace(set_num_threads=calls.append)

    _configure_execution(torch, "mps")

    assert calls == []


def test_cached_commonlingua_checkpoint_loads_with_the_pinned_model() -> None:
    pytest.importorskip("torch")

    settings = Settings.from_env()
    checkpoint = (
        settings.model_cache_dir
        / "models--PleIAs--CommonLingua"
        / "snapshots"
        / settings.language_model_revision
        / "model.pt"
    )
    if not Path(checkpoint).is_file():
        pytest.skip()

    from landuse_sentence_relevance.models.commonlingua_model import load_predictor

    predictor, _, max_len = load_predictor(str(checkpoint))

    language, confidence = predictor("A sentence about the landscape.")

    assert max_len == 512
    assert language == "eng"
    assert 0.0 <= confidence <= 1.0

    batch = predictor.predict_many(
        [
            "A sentence about the landscape.",
            "Another sentence about vegetation.",
        ]
    )

    assert len(batch) == 2
    assert all(language == "eng" and 0.0 <= confidence <= 1.0 for language, confidence in batch)
