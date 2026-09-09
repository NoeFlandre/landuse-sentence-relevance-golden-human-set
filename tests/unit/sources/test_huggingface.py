import logging
from typing import Any, get_type_hints

from landuse_sentence_relevance.sources.huggingface import (
    HuggingFaceDatasetLoader,
    HuggingFaceDatasetRows,
    HuggingFaceRowConfig,
)


def test_rows_use_explicit_loader_contract() -> None:
    annotations = get_type_hints(HuggingFaceDatasetRows.__init__)

    assert annotations["loader"] == HuggingFaceDatasetLoader | None


def test_rows_are_loaded_in_streaming_mode_at_an_immutable_revision(caplog) -> None:
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

    caplog.set_level(logging.INFO)

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
    assert "Opening Hugging Face stream: owner/dataset" in caplog.text


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
    assert calls[0]["on_bad_files"] == "warn"


def test_default_loader_accepts_direct_remote_files_without_a_dataset_revision(monkeypatch) -> None:
    calls = []

    def fake_load_dataset(**kwargs: Any):
        calls.append(kwargs)
        return [{"id": 1}]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    rows = HuggingFaceDatasetRows(
        HuggingFaceRowConfig(
            "owner/dataset",
            "revision-sha",
            "train",
            "config-name",
            remote_files=("https://example.test/data.parquet",),
        )
    )

    assert list(rows()) == [{"id": 1}]
    assert calls[0]["path"] == "parquet"
    assert "revision" not in calls[0]


def test_shards_partition_one_stream_without_materializing_rows() -> None:
    class ShardableStream:
        def shard(self, num_shards: int, index: int, contiguous: bool = True):
            assert num_shards == 3
            assert contiguous is True
            return [{"shard": index}]

    calls = []

    def fake_load_dataset(**kwargs: Any):
        calls.append(kwargs)
        return ShardableStream()

    rows = HuggingFaceDatasetRows(
        HuggingFaceRowConfig("owner/dataset", "revision-sha", "train", "config-name"),
        loader=fake_load_dataset,
    )

    shards = rows.shards(3)

    assert [list(shard) for shard in shards] == [[{"shard": 0}], [{"shard": 1}], [{"shard": 2}]]
    assert len(calls) == 1


def test_shards_fall_back_to_one_stream_for_plain_iterables() -> None:
    stream = [{"id": 1}]
    rows = HuggingFaceDatasetRows(
        HuggingFaceRowConfig("owner/dataset", "revision-sha", "train", "config-name"),
        loader=lambda **kwargs: stream,
    )

    assert rows.shards(3) == (stream,)


def test_shards_can_follow_the_streams_data_source_count() -> None:
    class ShardableStream:
        n_shards = 2

        def __iter__(self):
            return iter(())

        def shard(self, num_shards: int, index: int, contiguous: bool = True):
            return [{"shard": index, "count": num_shards, "contiguous": contiguous}]

    rows = HuggingFaceDatasetRows(
        HuggingFaceRowConfig("owner/dataset", "revision-sha", "train", "config-name"),
        loader=lambda **kwargs: ShardableStream(),
    )

    assert [list(shard) for shard in rows.shards()] == [
        [{"shard": 0, "count": 2, "contiguous": True}],
        [{"shard": 1, "count": 2, "contiguous": True}],
    ]


def test_rows_project_only_requested_columns_before_streaming() -> None:
    class SelectableStream:
        def __iter__(self):
            return iter([{"id": 1, "unused": "not returned"}])

        def select_columns(self, column_names: list[str]):
            assert column_names == ["id"]
            return [{"id": 1}]

    calls: list[dict[str, Any]] = []

    def fake_load_dataset(**kwargs: Any):
        calls.append(kwargs)
        return SelectableStream()

    rows = HuggingFaceDatasetRows(
        HuggingFaceRowConfig(
            "owner/dataset",
            "revision-sha",
            "train",
            "config-name",
            columns=("id",),
        ),
        loader=fake_load_dataset,
    )

    assert list(rows()) == [{"id": 1}]
    assert calls[0]["columns"] == ["id"]


def test_rows_can_stream_pinned_remote_parquet_files_directly() -> None:
    calls: list[dict[str, Any]] = []

    def fake_load_dataset(**kwargs: Any):
        calls.append(kwargs)
        return [{"id": 1}]

    rows = HuggingFaceDatasetRows(
        HuggingFaceRowConfig(
            "owner/dataset",
            "revision-sha",
            "train",
            "config-name",
            columns=("id",),
            remote_files=("https://example.test/data.parquet",),
        ),
        loader=fake_load_dataset,
    )

    assert list(rows()) == [{"id": 1}]
    assert calls == [
        {
            "path": "parquet",
            "name": None,
            "data_files": {"train": ["https://example.test/data.parquet"]},
            "split": "train",
            "streaming": True,
            "columns": ["id"],
        }
    ]
