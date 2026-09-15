from __future__ import annotations

import json
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import cast

import pytest
from tests.unit.test_models import make_candidate

import landuse_sentence_relevance.storage.v3_candidate_pool as v3_storage
from landuse_sentence_relevance.domain.models import Candidate, Label, Source
from landuse_sentence_relevance.storage.v3_candidate_pool import (
    V3CandidateProgress,
    V3CandidateProgressStore,
    load_v2_benchmark,
)


class ForeignSource(StrEnum):
    OTHER = "other"
    SECOND = "second"


def test_v3_progress_store_round_trips_candidates_and_completed_sources(tmp_path: Path) -> None:
    candidate = make_candidate()
    metadata = {"schema_version": 1, "seed": "v3-test"}
    store = V3CandidateProgressStore(tmp_path / "progress.json")

    store.save((candidate,), {Source.WIKIPEDIA}, metadata)

    assert store.load(metadata) == V3CandidateProgress(
        candidates=(candidate,),
        completed_sources=frozenset({Source.WIKIPEDIA}),
    )


def test_v3_progress_store_writes_sorted_compact_utf8_json(tmp_path: Path) -> None:
    store = V3CandidateProgressStore(tmp_path / "progress.json")
    candidate = replace(make_candidate(), sentence="Héllö — hills")

    store.save(
        (candidate,),
        {Source.DESCRIPTION, Source.WIKIPEDIA},
        {"schema_version": 1, "seed": "deterministic"},
    )

    assert store.load({"schema_version": 1, "seed": "deterministic"}) is not None
    assert (
        store._path.read_bytes()
        == (
            '{"candidates":[{"candidate_id":"c-1","h3_cell":"832830fffffffff",'
            '"h3_resolution":3,"language":"en","latitude":45.0,"longitude":2.0,'
            '"place_name":"A place","region":"A region","sentence":"Héllö — hills",'
            '"source":"wikipedia","source_field":"wikipedia_section",'
            '"source_record_id":"polygon-1","source_url":"https://example.test/article"}],'
            '"completed_sources":["description","wikipedia"],'
            '"metadata":{"schema_version":1,"seed":"deterministic"}}\n'
        ).encode()
    )


def test_v3_progress_store_loads_with_lowercase_utf8(monkeypatch, tmp_path: Path) -> None:
    store = V3CandidateProgressStore(tmp_path / "progress.json")
    metadata = {"schema_version": 1, "seed": "v3-test"}
    store.save((make_candidate(),), (), metadata)
    original_read_text = Path.read_text
    calls: list[dict[str, object]] = []

    def read_text(path, *args, **kwargs):
        calls.append(kwargs)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert store.load(metadata) is not None
    assert any(call.get("encoding") == "utf-8" for call in calls)


def test_v3_progress_store_defaults_missing_completed_sources_to_empty(tmp_path: Path) -> None:
    candidate = make_candidate()
    metadata = {"schema_version": 1, "seed": "v3-test"}
    path = tmp_path / "progress.json"
    path.write_text(
        json.dumps({"metadata": metadata, "candidates": [candidate.to_dict()]}),
        encoding="utf-8",
    )

    assert V3CandidateProgressStore(path).load(metadata) == V3CandidateProgress(
        candidates=(candidate,),
        completed_sources=frozenset(),
    )


def test_v3_progress_store_rejects_changed_metadata(tmp_path: Path) -> None:
    store = V3CandidateProgressStore(tmp_path / "progress.json")
    store.save((make_candidate(),), (), {"schema_version": 1, "seed": "saved"})

    with pytest.raises(ValueError) as error:
        store.load({"schema_version": 1, "seed": "changed"})
    assert str(error.value) == "V3 candidate progress metadata does not match the current configuration"


def test_v3_progress_store_rejects_unknown_completed_sources(tmp_path: Path) -> None:
    store = V3CandidateProgressStore(tmp_path / "progress.json")

    with pytest.raises(ValueError) as error:
        store.save(
            (make_candidate(),),
            {cast(Source, ForeignSource.OTHER), cast(Source, ForeignSource.SECOND)},
            {"schema_version": 1, "seed": "v3-test"},
        )
    assert str(error.value) == "completed sources are outside the V3 profile: other, second"


def test_v3_progress_store_disables_ascii_escaping(monkeypatch, tmp_path: Path) -> None:
    store = V3CandidateProgressStore(tmp_path / "progress.json")
    ensure_ascii_values: list[object] = []
    original_dump = v3_storage.json.dump

    def dump(payload, handle, *args, **kwargs):
        ensure_ascii_values.append(kwargs.get("ensure_ascii"))
        return original_dump(payload, handle, *args, **kwargs)

    monkeypatch.setattr(v3_storage.json, "dump", dump)
    store.save(
        (replace(make_candidate(), sentence="Héllö"),),
        (),
        {"schema_version": 1, "seed": "v3-test"},
    )

    assert ensure_ascii_values == [False]


def test_load_v2_benchmark_returns_annotations_and_all_reserved_cells(tmp_path: Path) -> None:
    path = tmp_path / "v2.csv"
    path.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        '"A Wikipedia sentence.",yes,Place,cell-wiki,45.0,2.0,wikipedia,region,https://example.test/wiki\n'
        '"A website sentence.",no,Other,cell-web,46.0,3.0,website,region,https://example.test/web\n',
        encoding="utf-8",
    )

    benchmark = load_v2_benchmark(path)

    assert benchmark.reserved_cells == frozenset({"cell-wiki", "cell-web"})
    assert [(row.candidate.source, row.label) for row in benchmark.annotations] == [
        (Source.WIKIPEDIA, Label.YES),
        (Source.WEBSITE, Label.NO),
    ]
    assert benchmark.annotations[0].candidate == Candidate(
        candidate_id="v2:0:wikipedia:cell-wiki",
        sentence="A Wikipedia sentence.",
        source=Source.WIKIPEDIA,
        source_record_id="v2:0",
        source_field="v2_benchmark",
        h3_cell="cell-wiki",
        h3_resolution=3,
        latitude=45.0,
        longitude=2.0,
        place_name="Place",
        region="region",
        source_url="https://example.test/wiki",
    )
    assert benchmark.annotations[1].candidate.candidate_id == "v2:1:website:cell-web"
    assert benchmark.annotations[0].candidate.place_name == "Place"
    assert benchmark.annotations[1].candidate.source_url == "https://example.test/web"
    assert len(benchmark.fingerprint) == 64


def test_load_v2_benchmark_opens_the_csv_with_utf8_and_newline_control(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "v2.csv"
    path.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        '"A sentence.",yes,Place,cell,45.0,2.0,wikipedia,region,https://example.test/wiki\n',
        encoding="utf-8",
    )
    original_open = Path.open
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def open_spy(path, *args, **kwargs):
        calls.append((args, kwargs))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_spy)
    load_v2_benchmark(path)

    assert any(not args and kwargs == {"newline": "", "encoding": "utf-8"} for args, kwargs in calls)


def test_load_v2_benchmark_rejects_a_changed_header_exactly(tmp_path: Path) -> None:
    path = tmp_path / "v2.csv"
    path.write_text("sentence,label\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_v2_benchmark(path)
    assert str(error.value) == "V2 benchmark columns do not match the frozen benchmark contract"


def test_v2_benchmark_parser_helper_errors_are_explicit() -> None:
    with pytest.raises(ValueError) as error:
        v3_storage._required_text({}, "sentence")
    assert str(error.value) == "V2 benchmark field sentence must be non-empty"

    with pytest.raises(ValueError) as error:
        v3_storage._required_float({"latitude": "not-a-number"}, "latitude")
    assert str(error.value) == "V2 benchmark field latitude must be numeric"

    with pytest.raises(ValueError) as error:
        v3_storage._source("not-a-source")
    assert str(error.value) == "V2 benchmark source is invalid: not-a-source"

    with pytest.raises(ValueError) as error:
        v3_storage._label("not-a-label")
    assert str(error.value) == "V2 benchmark label is invalid: not-a-label"
