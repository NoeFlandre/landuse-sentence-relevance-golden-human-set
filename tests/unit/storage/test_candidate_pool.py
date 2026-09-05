from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import ANY, Mock

import pytest
from tests.unit.test_models import make_candidate

from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.storage import CandidatePoolStore


def _pool() -> FinalizedCandidatePool:
    wikipedia = make_candidate("wikipedia-1")
    website = make_candidate("website-1")
    object.__setattr__(website, "source", Source.WEBSITE)
    object.__setattr__(website, "h3_cell", "832831fffffffff")
    return FinalizedCandidatePool(
        candidates=(wikipedia, website),
        cells=(wikipedia.h3_cell, website.h3_cell),
    )


def test_candidate_pool_store_round_trips_candidates_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "candidate-pool.json"
    store = CandidatePoolStore(path)
    metadata = {"schema_version": 1, "fingerprint": "stable-test-fingerprint"}

    store.save(_pool(), metadata)

    assert store.load(metadata) == _pool()
    assert json.loads(path.read_text(encoding="utf-8"))["metadata"] == metadata


def test_candidate_pool_store_reads_with_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "candidate-pool.json"
    store = CandidatePoolStore(path)
    metadata = {"schema_version": 1, "fingerprint": "encoding"}
    store.save(_pool(), metadata)
    read_encodings: list[str | None] = []
    original_read_text = Path.read_text

    def read_text(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        read_encodings.append(encoding)
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", read_text)

    assert store.load(metadata) == _pool()
    assert read_encodings == ["utf-8"]


def test_candidate_pool_store_returns_none_when_the_pool_is_missing(tmp_path: Path) -> None:
    store = CandidatePoolStore(tmp_path / "missing.json")

    assert store.load({"schema_version": 1, "fingerprint": "missing"}) is None


def test_candidate_pool_store_rejects_a_different_fingerprint(tmp_path: Path) -> None:
    store = CandidatePoolStore(tmp_path / "candidate-pool.json")
    store.save(_pool(), {"schema_version": 1, "fingerprint": "saved"})

    with pytest.raises(
        ValueError,
        match=r"^candidate pool metadata does not match the current configuration$",
    ):
        store.load({"schema_version": 1, "fingerprint": "different"})


def test_candidate_pool_store_writes_compact_sorted_utf8_json(tmp_path, monkeypatch) -> None:
    path = tmp_path / "candidate-pool.json"
    store = CandidatePoolStore(path)
    candidate = replace(make_candidate(), sentence="Héllö — hills")
    pool = FinalizedCandidatePool(candidates=(candidate,), cells=(candidate.h3_cell,))
    metadata = {"schema_version": 1, "fingerprint": "deterministic"}
    serialize = Mock(wraps=json.dump)
    monkeypatch.setattr(json, "dump", serialize)

    store.save(pool, metadata)

    expected = (
        '{"candidates":[{"candidate_id":"c-1","h3_cell":"832830fffffffff","h3_resolution":3,'
        '"language":"en","latitude":45.0,"longitude":2.0,"place_name":"A place","region":"A region",'
        '"sentence":"Héllö — hills","source":"wikipedia","source_field":"wikipedia_section",'
        '"source_record_id":"polygon-1","source_url":"https://example.test/article"}],'
        '"cells":["832830fffffffff"],"metadata":{"fingerprint":"deterministic","schema_version":1}}\n'
    ).encode()
    assert path.read_bytes() == expected
    assert store.load(metadata) == pool
    serialize.assert_called_once_with(ANY, ANY, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_candidate_pool_store_reuses_a_saved_pool_without_calling_the_builder(tmp_path: Path) -> None:
    store = CandidatePoolStore(tmp_path / "candidate-pool.json")
    metadata = {"schema_version": 1, "fingerprint": "saved"}
    expected = _pool()
    store.save(expected, metadata)
    calls: list[str] = []

    actual = store.load_or_build(metadata, lambda: calls.append("built") or expected)

    assert actual == expected
    assert calls == []


def test_candidate_pool_store_builds_and_saves_a_missing_pool(tmp_path: Path) -> None:
    store = CandidatePoolStore(tmp_path / "candidate-pool.json")
    metadata = {"schema_version": 1, "fingerprint": "new"}
    expected = _pool()

    actual = store.load_or_build(metadata, lambda: expected)

    assert actual == expected
    assert store.load(metadata) == expected
