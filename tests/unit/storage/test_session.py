import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import ANY, Mock

from tests.unit.test_constraints import make_annotations

from landuse_sentence_relevance.domain.models import Label
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


def test_annotation_store_append_and_replace_write_identical_sorted_utf8_json(tmp_path, monkeypatch) -> None:
    path = tmp_path / "annotations.jsonl"
    store = AnnotationStore(path)
    first = make_annotations()[0]
    annotation = replace(first, candidate=replace(first.candidate, sentence="Héllö — hills"))
    serialize = Mock(wraps=json.dumps)
    monkeypatch.setattr(json, "dumps", serialize)
    expected = (
        '{"candidate_id": "wikipedia-cell-00", "h3_cell": "cell-00", "h3_resolution": 3, '
        '"label": "yes", "language": "en", "latitude": 45.0, "longitude": 2.0, '
        '"place_name": "A place", "region": "A region", "sentence": "Héllö — hills", '
        '"source": "wikipedia", "source_field": "wikipedia_section", '
        '"source_record_id": "polygon-1", "source_url": "https://example.test/article"}\n'
    ).encode()

    store.record(annotation)
    store.record(annotation)
    assert path.read_bytes() == expected * 2
    assert store.load() == {annotation.candidate.candidate_id: annotation}
    serialize.assert_called_with(ANY, ensure_ascii=False, sort_keys=True)

    store.save((annotation,))
    assert path.read_bytes() == expected
    assert store.load() == {annotation.candidate.candidate_id: annotation}
    serialize.assert_called_with(ANY, ensure_ascii=False, sort_keys=True)


def test_annotation_store_saves_revised_annotations_without_stale_records(tmp_path) -> None:
    path = tmp_path / "nested" / "deeper" / "annotations.jsonl"
    store = AnnotationStore(path)
    first, second = make_annotations()[:2]
    store.save((first, second))

    revised = first.__class__(candidate=first.candidate, label=Label.NO)
    store.save((revised,))

    assert store.load() == {first.candidate.candidate_id: revised}
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_annotation_store_creates_missing_parent_directories(tmp_path) -> None:
    path = tmp_path / "nested" / "state" / "annotations.jsonl"

    AnnotationStore(path).record(make_annotations()[0])

    assert path.is_file()


def test_annotation_store_records_and_loads_using_explicit_utf8(tmp_path, monkeypatch) -> None:
    path = tmp_path / "annotations.jsonl"
    first = make_annotations()[0]
    annotation = replace(first, candidate=replace(first.candidate, sentence="Héllö — hills"))
    encodings = []
    original_open = Path.open

    def spy_open(self, *args, **kwargs):
        encodings.append(kwargs.get("encoding"))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)

    AnnotationStore(path).record(annotation)
    assert AnnotationStore(path).load() == {annotation.candidate.candidate_id: annotation}
    assert encodings == ["utf-8", "utf-8"]
