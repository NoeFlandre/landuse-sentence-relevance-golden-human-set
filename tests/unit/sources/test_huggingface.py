from typing import Any, get_type_hints

from landuse_sentence_relevance.sources.huggingface import (
    HuggingFaceDatasetLoader,
    HuggingFaceDatasetRows,
    HuggingFaceRowConfig,
)


def test_rows_use_explicit_loader_contract() -> None:
    annotations = get_type_hints(HuggingFaceDatasetRows.__init__)

    assert annotations["loader"] == HuggingFaceDatasetLoader | None


def test_rows_are_loaded_in_streaming_mode_at_an_immutable_revision() -> None:
    calls: list[dict[str, Any]] = []

    def fake_load_dataset(**kwargs: Any):
        calls.append(kwargs)
        return [{"id": 1}]

    rows = HuggingFaceDatasetRows(
        HuggingFaceRowConfig(
            dataset_id="owner/dataset",
            revision="revision-sha",
            split="train",
            config="config-name",
        ),
        loader=fake_load_dataset,
    )

    assert list(rows()) == [{"id": 1}]
    assert calls == [
        {
            "path": "owner/dataset",
            "name": "config-name",
            "split": "train",
            "streaming": True,
            "revision": "revision-sha",
        }
    ]


def test_default_loader_is_used_when_no_loader_is_injected(monkeypatch) -> None:
    calls = []

    def fake_load_dataset(**kwargs: Any):
        calls.append(kwargs)
        return [{"id": 1}]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    rows = HuggingFaceDatasetRows(
        HuggingFaceRowConfig("owner/dataset", "revision-sha", "train", "config-name")
    )

    assert list(rows()) == [{"id": 1}]
    assert calls[0]["streaming"] is True
