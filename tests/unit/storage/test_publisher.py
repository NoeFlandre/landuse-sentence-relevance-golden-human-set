import logging
from threading import Event, Lock, Thread

import pytest
from tests.unit.test_constraints import make_annotations

from landuse_sentence_relevance.storage.cache import ManagedCache
from landuse_sentence_relevance.storage.publisher import DatasetPublisher, DatasetUploader


def test_dataset_publisher_uses_explicit_uploader_contract() -> None:
    from typing import get_type_hints

    annotations = get_type_hints(DatasetPublisher.__init__)

    assert annotations["uploader"] == DatasetUploader | None


@pytest.fixture
def block_default_hub_upload(monkeypatch):
    def block_upload(self, **kwargs):
        raise AssertionError("unexpected real Hub upload during a unit test")

    monkeypatch.setattr(DatasetPublisher, "_upload_to_hub", block_upload)


def test_publisher_uploads_only_when_all_quotas_are_met(block_default_hub_upload, caplog) -> None:
    calls = []
    publisher = DatasetPublisher(
        dataset_id="NoeFlandre/landuse-sentence-relevance-golden-human-set",
        uploader=lambda **kwargs: calls.append(kwargs),
    )

    assert publisher.publish_if_ready(make_annotations()[:-1]) is False
    assert calls == []

    caplog.set_level(logging.INFO)
    assert publisher.publish_if_ready(make_annotations()) is True
    assert len(calls) == 1
    assert calls[0]["dataset_id"] == "NoeFlandre/landuse-sentence-relevance-golden-human-set"
    assert calls[0]["private"] is False
    assert len(calls[0]["records"]) == 100
    assert [record.message for record in caplog.records] == [
        "Final contract satisfied; uploading 100 annotations to "
        "NoeFlandre/landuse-sentence-relevance-golden-human-set",
        "Public upload complete for NoeFlandre/landuse-sentence-relevance-golden-human-set",
    ]


def test_publisher_uses_the_configured_dataset_split(block_default_hub_upload) -> None:
    calls = []
    publisher = DatasetPublisher(
        dataset_id="dataset",
        split="v2",
        uploader=lambda **kwargs: calls.append(kwargs),
    )

    assert publisher.publish_if_ready(make_annotations()) is True

    assert calls[0]["split"] == "v2"


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
    tmp_path, block_default_hub_upload, caplog
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

    caplog.set_level(logging.INFO)
    assert publisher.publish_if_ready(make_annotations()) is True
    assert events == ["upload", "cleanup"]
    assert not cache.root.exists()
    assert [record.message for record in caplog.records] == [
        "Final contract satisfied; uploading 100 annotations to dataset",
        "Public upload complete for dataset",
        "Removing disposable runtime cache after successful upload",
    ]


def test_publisher_prepares_the_cache_before_reuploading_after_cleanup(
    caplog, block_default_hub_upload
) -> None:
    events = []
    publisher = DatasetPublisher(
        dataset_id="dataset",
        prepare=lambda: events.append("prepare"),
        uploader=lambda **kwargs: events.append("upload"),
        cleanup=lambda: events.append("cleanup"),
    )

    caplog.set_level(logging.INFO)
    assert publisher.publish_if_ready(make_annotations()) is True
    assert events == ["prepare", "upload", "cleanup"]
    assert [record.message for record in caplog.records] == [
        "Final contract satisfied; uploading 100 annotations to dataset",
        "Preparing the disposable runtime cache for upload",
        "Public upload complete for dataset",
        "Removing disposable runtime cache after successful upload",
    ]


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


def test_publisher_serializes_concurrent_uploads(block_default_hub_upload) -> None:
    active_uploads = 0
    maximum_active_uploads = 0
    upload_count = 0
    state_lock = Lock()
    first_upload_started = Event()
    second_upload_started = Event()
    release_first_upload = Event()
    errors = []

    def upload(**kwargs) -> None:
        nonlocal active_uploads, maximum_active_uploads, upload_count
        with state_lock:
            active_uploads += 1
            maximum_active_uploads = max(maximum_active_uploads, active_uploads)
            upload_count += 1
            current_upload = upload_count
        if current_upload == 1:
            first_upload_started.set()
            assert release_first_upload.wait(timeout=2)
        else:
            second_upload_started.set()
            assert release_first_upload.wait(timeout=2)
        with state_lock:
            active_uploads -= 1

    publisher = DatasetPublisher(dataset_id="dataset", uploader=upload)

    def publish() -> None:
        try:
            assert publisher.publish_if_ready(make_annotations()) is True
        except BaseException as error:  # pragma: no cover - only reports a thread failure
            errors.append(error)

    first = Thread(target=publish)
    second = Thread(target=publish)
    first.start()
    assert first_upload_started.wait(timeout=2)
    second.start()
    assert not second_upload_started.wait(timeout=1)
    release_first_upload.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert errors == []
    assert upload_count == 2
    assert maximum_active_uploads == 1


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


def test_real_hub_boundary_defaults_to_the_train_split(monkeypatch) -> None:
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

    DatasetPublisher("dataset", token="token")._upload_to_hub(
        dataset_id="dataset",
        records=[],
        token="token",
        private=False,
    )

    assert calls[-1][2]["split"] == "train"


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
