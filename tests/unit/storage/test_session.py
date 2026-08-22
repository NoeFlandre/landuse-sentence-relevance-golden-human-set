import json

from tests.unit.test_constraints import make_annotations

from landuse_sentence_relevance.storage.session import AnnotationStore


def test_annotation_store_persists_only_annotation_records(tmp_path) -> None:
    path = tmp_path / "state" / "annotations.jsonl"
    store = AnnotationStore(path)
    first, second = make_annotations()[:2]

    store.record(first)
    store.record(second)

    restored = store.load()

    assert restored == {first.candidate.candidate_id: first, second.candidate.candidate_id: second}
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "raw_source_row" not in json.loads(lines[0])


def test_annotation_store_replaces_duplicate_candidate_records(tmp_path) -> None:
    store = AnnotationStore(tmp_path / "annotations.jsonl")
    first, second = make_annotations()[:2]

    store.record(first)
    store.record(second)
    store.record(first)

    assert store.load() == {
        first.candidate.candidate_id: first,
        second.candidate.candidate_id: second,
    }
