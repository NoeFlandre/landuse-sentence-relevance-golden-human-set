from pathlib import Path

import pytest

from landuse_sentence_relevance.config import Settings


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
