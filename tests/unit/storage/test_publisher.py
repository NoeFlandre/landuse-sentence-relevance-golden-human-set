import pytest
from tests.unit.test_constraints import make_annotations

from landuse_sentence_relevance.storage.cache import ManagedCache
from landuse_sentence_relevance.storage.publisher import DatasetPublisher


@pytest.fixture
def block_default_hub_upload(monkeypatch):
    def block_upload(self, **kwargs):
        raise AssertionError("unexpected real Hub upload during a unit test")

    monkeypatch.setattr(DatasetPublisher, "_upload_to_hub", block_upload)


def test_publisher_uploads_only_when_all_quotas_are_met(block_default_hub_upload) -> None:
    calls = []
    publisher = DatasetPublisher(
        dataset_id="NoeFlandre/landuse-sentence-relevance-golden-human-set",
        uploader=lambda **kwargs: calls.append(kwargs),
    )

    assert publisher.publish_if_ready(make_annotations()[:-1]) is False
    assert calls == []

    assert publisher.publish_if_ready(make_annotations()) is True
    assert len(calls) == 1
    assert calls[0]["dataset_id"] == "NoeFlandre/landuse-sentence-relevance-golden-human-set"
    assert calls[0]["private"] is False
    assert len(calls[0]["records"]) == 100


def test_publisher_does_not_reorder_the_deterministic_final_selection(block_default_hub_upload) -> None:
    records = []
    publisher = DatasetPublisher(
        dataset_id="dataset",
        uploader=lambda **kwargs: records.extend(kwargs["records"]),
    )

    assert publisher.publish_if_ready(make_annotations()) is True
    assert [record["candidate_id"] for record in records] == sorted(
        record["candidate_id"] for record in records
    )


def test_publisher_cleans_the_application_cache_after_a_successful_upload(
    tmp_path, block_default_hub_upload
) -> None:
    cache = ManagedCache(tmp_path / "runtime-cache")
    cache.prepare({})
    (cache.root / "model.bin").write_bytes(b"weights")
    events = []

    def cleanup() -> None:
        events.append("cleanup")
        cache.cleanup()

    publisher = DatasetPublisher(
        dataset_id="dataset",
        uploader=lambda **kwargs: events.append("upload"),
        cleanup=cleanup,
    )

    assert publisher.publish_if_ready(make_annotations()) is True
    assert events == ["upload", "cleanup"]
    assert not cache.root.exists()


def test_publisher_keeps_the_application_cache_when_upload_fails(tmp_path, block_default_hub_upload) -> None:
    cache = ManagedCache(tmp_path / "runtime-cache")
    cache.prepare({})
    (cache.root / "model.bin").write_bytes(b"weights")
    cleanup_calls = []

    def fail_upload(**kwargs):
        raise RuntimeError("upload failed")

    publisher = DatasetPublisher(
        dataset_id="dataset",
        uploader=fail_upload,
        cleanup=lambda: cleanup_calls.append("cleanup"),
    )

    with pytest.raises(RuntimeError, match="upload failed"):
        publisher.publish_if_ready(make_annotations())

    assert cleanup_calls == []
    assert (cache.root / "model.bin").exists()


def test_real_hub_boundary_creates_a_public_dataset(monkeypatch) -> None:
    import datasets
    import huggingface_hub

    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("api", token))

        def create_repo(self, **kwargs):
            calls.append(("repo", kwargs))

    class FakeDataset:
        @classmethod
        def from_list(cls, records):
            calls.append(("records", records))
            return cls()

        def push_to_hub(self, *args, **kwargs):
            calls.append(("push", args, kwargs))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(datasets, "Dataset", FakeDataset)
    publisher = DatasetPublisher("dataset", token="token")

    assert publisher.publish_if_ready(make_annotations()) is True
    assert calls[0] == ("api", "token")
    assert calls[1] == (
        "repo",
        {
            "repo_id": "dataset",
            "repo_type": "dataset",
            "private": False,
            "exist_ok": True,
        },
    )
    assert calls[2][0] == "records"
    assert len(calls[2][1]) == 100
    assert calls[-1] == (
        "push",
        ("dataset",),
        {
            "split": "train",
            "token": "token",
            "commit_message": "Publish balanced human annotations",
        },
    )


def test_real_hub_boundary_reports_missing_dependencies(monkeypatch) -> None:
    import builtins

    original_import = builtins.__import__

    def fail_hub_import(name, *args, **kwargs):
        if name in {"datasets", "huggingface_hub"}:
            raise ImportError("missing publishing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_hub_import)

    with pytest.raises(RuntimeError) as error:
        DatasetPublisher("dataset")._upload_to_hub(
            dataset_id="dataset",
            records=[],
            token=None,
            private=False,
        )

    assert str(error.value) == "Install the project dependencies with `uv sync` to publish the dataset"
