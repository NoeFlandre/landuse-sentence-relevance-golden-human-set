import json
from pathlib import Path

import pytest
from tests.unit.test_models import make_candidate

import landuse_sentence_relevance.storage.candidate_progress as candidate_progress_module
from landuse_sentence_relevance.storage.atomic import TextWriter
from landuse_sentence_relevance.storage.candidate_progress import (
    CandidateProgressStore,
    _is_remote_sample_expansion,
    _remote_sample_counts,
    _without_remote_sample_count,
)


def test_candidate_progress_store_round_trips_candidates_and_metadata(tmp_path: Path) -> None:
    store = CandidateProgressStore(tmp_path / "candidate-progress.json")
    candidate = make_candidate()
    metadata = {"schema_version": 2, "fingerprint": "stable"}

    store.save((candidate,), metadata)

    assert store.load(metadata) == (candidate,)


def test_candidate_progress_store_returns_none_when_checkpoint_is_missing(tmp_path: Path) -> None:
    store = CandidateProgressStore(tmp_path / "missing.json")

    assert store.load({"schema_version": 2}) is None


def test_candidate_progress_store_rejects_different_metadata(tmp_path: Path) -> None:
    store = CandidateProgressStore(tmp_path / "candidate-progress.json")
    store.save((make_candidate(),), {"schema_version": 2, "fingerprint": "saved"})

    with pytest.raises(
        ValueError,
        match=r"^candidate progress metadata does not match the current configuration$",
    ):
        store.load({"schema_version": 2, "fingerprint": "different"})


def test_candidate_progress_store_reuses_checkpoint_across_worker_changes(tmp_path: Path) -> None:
    path = tmp_path / "candidate-progress.json"
    candidate = make_candidate()
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": 2,
                    "sentence_splitter": {"id": "sat", "batch_size": 1, "workers": 8},
                },
                "candidates": [candidate.to_dict()],
            }
        ),
        encoding="utf-8",
    )

    assert CandidateProgressStore(path).load({"schema_version": 2, "sentence_splitter": {"id": "sat"}}) == (
        candidate,
    )


def test_candidate_progress_store_reuses_checkpoint_across_text_bound_changes(tmp_path: Path) -> None:
    path = tmp_path / "candidate-progress.json"
    candidate = make_candidate()
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": 2,
                    "sampling": {"website_max_text_characters": 800},
                },
                "candidates": [candidate.to_dict()],
            }
        ),
        encoding="utf-8",
    )

    assert CandidateProgressStore(path).load(
        {"schema_version": 2, "sampling": {"website_max_text_characters": 400}}
    ) == (candidate,)


def test_candidate_progress_store_reuses_checkpoint_when_remote_sample_expands(tmp_path: Path) -> None:
    path = tmp_path / "candidate-progress.json"
    candidate = make_candidate()
    CandidateProgressStore(path).save(
        (candidate,),
        {"schema_version": 2, "sampling": {"remote_file_sample_count": 32}},
    )

    assert CandidateProgressStore(path).load(
        {"schema_version": 2, "sampling": {"remote_file_sample_count": 128}}
    ) == (candidate,)


def test_candidate_progress_store_rejects_a_smaller_remote_sample(tmp_path: Path) -> None:
    path = tmp_path / "candidate-progress.json"
    CandidateProgressStore(path).save(
        (make_candidate(),),
        {"schema_version": 2, "sampling": {"remote_file_sample_count": 128}},
    )

    with pytest.raises(ValueError, match="candidate progress metadata does not match"):
        CandidateProgressStore(path).load({"schema_version": 2, "sampling": {"remote_file_sample_count": 32}})


def test_remote_sample_expansion_requires_a_strict_increase() -> None:
    metadata = {"schema_version": 2, "sampling": {"remote_file_sample_count": 32}}

    assert not _is_remote_sample_expansion(metadata, metadata)


def test_remote_sample_counts_require_both_counts() -> None:
    assert (
        _remote_sample_counts(
            {"sampling": {}},
            {"sampling": {"remote_file_sample_count": 128}},
        )
        is None
    )


def test_without_remote_sample_count_preserves_sampling_without_the_key() -> None:
    metadata = {"schema_version": 2, "sampling": {"other": "value"}}

    assert _without_remote_sample_count(metadata) == metadata


def test_without_remote_sample_count_keeps_the_sampling_mapping() -> None:
    metadata = {"schema_version": 2, "sampling": {"remote_file_sample_count": 32}}

    assert _without_remote_sample_count(metadata) == {"schema_version": 2, "sampling": {}}


def test_candidate_progress_store_reads_with_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "candidate-progress.json"
    store = CandidateProgressStore(path)
    metadata = {"schema_version": 2, "fingerprint": "encoding"}
    store.save((make_candidate(),), metadata)
    read_encodings: list[str | None] = []
    original_read_text = Path.read_text

    def read_text(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        read_encodings.append(encoding)
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", read_text)

    assert store.load(metadata) == (make_candidate(),)
    assert read_encodings == ["utf-8"]


def test_candidate_progress_store_passes_explicit_json_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options: dict[str, object] = {}

    def spy_dump(_payload: object, _handle: TextWriter, **kwargs: object) -> None:
        options.update(kwargs)

    monkeypatch.setattr(candidate_progress_module.json, "dump", spy_dump)

    CandidateProgressStore(tmp_path / "candidate-progress.json").save(
        (make_candidate(),), {"schema_version": 2, "fingerprint": "deterministic"}
    )

    assert options == {
        "ensure_ascii": False,
        "sort_keys": True,
        "separators": (",", ":"),
    }


def test_candidate_progress_store_rejects_a_changed_semantic_nested_value(tmp_path: Path) -> None:
    path = tmp_path / "candidate-progress.json"
    saved_metadata = {"schema_version": 2, "sentence_splitter": {"id": "sat"}}
    CandidateProgressStore(path).save((make_candidate(),), saved_metadata)

    with pytest.raises(
        ValueError,
        match=r"^candidate progress metadata does not match the current configuration$",
    ):
        CandidateProgressStore(path).load({"schema_version": 2, "sentence_splitter": {"id": "different"}})
