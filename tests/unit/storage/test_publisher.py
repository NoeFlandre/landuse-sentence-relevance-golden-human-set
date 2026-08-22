from tests.unit.test_constraints import make_annotations

from landuse_sentence_relevance.storage.publisher import DatasetPublisher


def test_publisher_uploads_only_when_all_quotas_are_met() -> None:
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


def test_publisher_does_not_reorder_the_deterministic_final_selection() -> None:
    records = []
    publisher = DatasetPublisher(
        dataset_id="dataset",
        uploader=lambda **kwargs: records.extend(kwargs["records"]),
    )

    assert publisher.publish_if_ready(make_annotations()) is True
    assert [record["candidate_id"] for record in records] == sorted(
        record["candidate_id"] for record in records
    )


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
    assert calls[1][1]["private"] is False
    assert calls[-1][0] == "push"
