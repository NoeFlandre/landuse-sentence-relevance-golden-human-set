import json
from dataclasses import replace
from pathlib import Path

from tests.unit.test_constraints import make_annotations

import landuse_sentence_relevance.storage.session as session_module
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


def test_annotation_store_uses_one_record_writer_for_append_and_replace(tmp_path, monkeypatch) -> None:
    path = tmp_path / "annotations.jsonl"
    store = AnnotationStore(path)
    first, second = make_annotations()[:2]
    second = replace(second, candidate=replace(second.candidate, sentence="Héllö — hills"))

    write_calls = 0
    original_writer = store._write_annotation

    def observe_writer(handle, annotation) -> None:
        nonlocal write_calls
        write_calls += 1
        original_writer(handle, annotation)

    monkeypatch.setattr(store, "_write_annotation", observe_writer)

    store.record(first)
    assert (
        path.read_text(encoding="utf-8")
        == json.dumps(first.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
    )

    store.save((second,))

    assert write_calls == 2
    assert (
        path.read_text(encoding="utf-8")
        == json.dumps(second.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
    )


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


def test_annotation_store_save_writes_utf8_and_sorted_json(tmp_path: Path) -> None:
    path = tmp_path / "annotations.jsonl"
    first = make_annotations()[0]
    annotation = replace(first, candidate=replace(first.candidate, sentence="Héllö — hills"))

    AnnotationStore(path).save((annotation,))

    line = path.read_bytes().decode("utf-8").strip()
    assert '"Héllö — hills"' in line
    assert list(json.loads(line)) == sorted(json.loads(line))


def test_annotation_store_save_passes_explicit_json_options(tmp_path: Path, monkeypatch) -> None:
    options = {}
    original_dumps = session_module.json.dumps

    def spy_dumps(payload: object, **kwargs: object) -> str:
        options.update(kwargs)
        return original_dumps(payload, ensure_ascii=False, sort_keys=True)

    monkeypatch.setattr(session_module.json, "dumps", spy_dumps)
    AnnotationStore(tmp_path / "annotations.jsonl").save((make_annotations()[0],))

    assert options == {"ensure_ascii": False, "sort_keys": True}


def test_annotation_store_writes_utf8_and_sorted_json(tmp_path, monkeypatch) -> None:
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

    line = path.read_bytes().decode("utf-8").strip()
    assert encodings[0] == "utf-8"
    assert '"Héllö — hills"' in line
    assert list(json.loads(line)) == sorted(json.loads(line))


def test_annotation_store_passes_explicit_json_options(tmp_path, monkeypatch) -> None:
    options = {}
    original_dumps = session_module.json.dumps

    def spy_dumps(*args, **kwargs):
        options.update(kwargs)
        return original_dumps(*args, **kwargs)

    monkeypatch.setattr(session_module.json, "dumps", spy_dumps)
    AnnotationStore(tmp_path / "annotations.jsonl").record(make_annotations()[0])

    assert options == {"ensure_ascii": False, "sort_keys": True}


def test_annotation_store_loads_using_utf8(tmp_path, monkeypatch) -> None:
    path = tmp_path / "annotations.jsonl"
    first = make_annotations()[0]
    annotation = replace(first, candidate=replace(first.candidate, sentence="Héllö — hills"))
    AnnotationStore(path).record(annotation)
    encodings = []
    original_read_text = Path.read_text

    def spy_read_text(self, *args, **kwargs):
        encodings.append(kwargs.get("encoding"))
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy_read_text)

    assert AnnotationStore(path).load() == {annotation.candidate.candidate_id: annotation}
    assert encodings == ["utf-8"]
