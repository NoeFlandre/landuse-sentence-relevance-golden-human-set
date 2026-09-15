from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import landuse_sentence_relevance.storage.v3_annotation_seed as seed_storage
from landuse_sentence_relevance.domain.models import Label, Source
from landuse_sentence_relevance.domain.profile import SourceLabelQuotas
from landuse_sentence_relevance.domain.v3_annotation import V3AnnotationSeed
from landuse_sentence_relevance.storage.v3_annotation_seed import V3AnnotationSeedStore


def test_v3_annotation_seed_store_round_trips_the_seed_and_metadata(tmp_path: Path) -> None:
    seed = V3AnnotationSeed(
        rows=(),
        excluded_v2_rows=(),
        reserved_v2_cells=frozenset(),
        quotas=SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 0}),
        benchmark_sha256="0" * 64,
        seed="test-seed",
    )
    metadata = {"schema_version": 1, "state_kind": "v3-annotation-seed"}
    store = V3AnnotationSeedStore(tmp_path / "state" / "v3-seed.json")

    store.save(seed, metadata)

    assert store.load(metadata) == seed


def test_v3_annotation_seed_store_returns_none_before_the_first_checkpoint(tmp_path: Path) -> None:
    store = V3AnnotationSeedStore(tmp_path / "missing.json")

    assert store.load({"schema_version": 1}) is None


def test_v3_annotation_seed_store_rejects_metadata_drift(tmp_path: Path) -> None:
    seed = V3AnnotationSeed(
        rows=(),
        excluded_v2_rows=(),
        reserved_v2_cells=frozenset(),
        quotas=SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 0}),
        benchmark_sha256="0" * 64,
        seed="test-seed",
    )
    store = V3AnnotationSeedStore(tmp_path / "state" / "v3-seed.json")
    store.save(seed, {"benchmark_sha256": "a" * 64})

    with pytest.raises(ValueError) as error:
        store.load({"benchmark_sha256": "b" * 64})
    assert str(error.value) == (
        "V3 annotation seed metadata does not match the current configuration; "
        "the benchmark SHA-256 may have changed"
    )


def test_v3_annotation_seed_store_rejects_a_non_object_state(tmp_path: Path) -> None:
    path = tmp_path / "v3-seed.json"
    path.write_text(json.dumps({"metadata": {}, "state": []}), encoding="utf-8")

    with pytest.raises(ValueError) as error:
        V3AnnotationSeedStore(path).load({})
    assert str(error.value) == "V3 annotation seed state must be an object"


def test_v3_annotation_seed_store_writes_canonical_utf8_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = V3AnnotationSeed(
        rows=(),
        excluded_v2_rows=(),
        reserved_v2_cells=frozenset(),
        quotas=SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 0}),
        benchmark_sha256="0" * 64,
        seed="séed",
    )
    metadata = {"schema_version": 1, "note": "café", "z": 1, "a": 2}
    path = tmp_path / "v3-seed.json"
    store = V3AnnotationSeedStore(path)
    dump = Mock(wraps=seed_storage.json.dump)
    monkeypatch.setattr(seed_storage.json, "dump", dump)

    store.save(seed, metadata)

    assert dump.call_args is not None
    assert dump.call_args.kwargs == {
        "ensure_ascii": False,
        "sort_keys": True,
        "separators": (",", ":"),
    }
    text = path.read_text(encoding="utf-8")
    assert text.startswith('{"metadata":{"a":2,"note":"café","schema_version":1,"z":1},')
    assert "\\u00e9" not in text
    assert store.load(metadata) == seed
