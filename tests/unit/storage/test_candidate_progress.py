import json
from pathlib import Path

import pytest
from tests.builders import make_candidate

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


# ------------------------------------------------- completed sources (shard-level resume)


def test_completed_sources_is_empty_when_no_checkpoint_exists(tmp_path: Path) -> None:
    """Nothing recorded means nothing may be skipped."""
    store = CandidateProgressStore(tmp_path / "missing.json")
    assert store.load_completed_sources({"schema_version": 2}) == frozenset()


def test_a_checkpoint_written_before_the_field_existed_reports_nothing(tmp_path: Path) -> None:
    """The safe default, and the one the docstring promises.

    A source stopped part way through holds only the shards it reached, so treating an absent
    record as "complete" would freeze a partial read into the pool with nothing saying so.
    """
    path = tmp_path / "progress.json"
    metadata = {"schema_version": 2, "fingerprint": "stable"}
    path.write_text(json.dumps({"metadata": metadata, "candidates": []}, sort_keys=True), encoding="utf-8")
    assert CandidateProgressStore(path).load_completed_sources(metadata) == frozenset()


def test_a_malformed_completed_list_reports_nothing_rather_than_guessing(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    metadata = {"schema_version": 2, "fingerprint": "stable"}
    path.write_text(
        json.dumps(
            {"metadata": metadata, "candidates": [], "completed_sources": "wikipedia"},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    assert CandidateProgressStore(path).load_completed_sources(metadata) == frozenset()


def test_completed_sources_round_trip_through_save(tmp_path: Path) -> None:
    from landuse_sentence_relevance.domain.models import Source

    store = CandidateProgressStore(tmp_path / "progress.json")
    metadata = {"schema_version": 2, "fingerprint": "stable"}
    store.save([], metadata, completed_sources=[Source.WIKIPEDIA])
    assert store.load_completed_sources(metadata) == frozenset({Source.WIKIPEDIA})


def test_a_changed_configuration_refuses_the_checkpoint(tmp_path: Path) -> None:
    """A changed pin must invalidate the record rather than silently skip different data."""
    store = CandidateProgressStore(tmp_path / "progress.json")
    store.save([], {"schema_version": 2, "fingerprint": "stable"}, completed_sources=[])
    with pytest.raises(
        ValueError,
        match=r"^candidate progress metadata does not match the current configuration$",
    ):
        store.load_completed_sources({"schema_version": 2, "fingerprint": "changed"})


def test_completed_sources_are_read_with_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkpoint is written as UTF-8, so it must be read back as UTF-8 rather than as
    whatever the platform happens to default to -- a source name outside ASCII would otherwise
    decode differently on another machine and re-stream a source that was already finished."""

    from landuse_sentence_relevance.domain.models import Source

    path = tmp_path / "progress.json"
    store = CandidateProgressStore(path)
    metadata = {"schema_version": 2, "fingerprint": "encoding"}
    store.save([], metadata, completed_sources=[Source.WIKIPEDIA])
    read_encodings: list[str | None] = []
    original_read_text = Path.read_text

    def read_text(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        read_encodings.append(encoding)
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", read_text)

    assert store.load_completed_sources(metadata) == frozenset({Source.WIKIPEDIA})
    assert read_encodings == ["utf-8"]


# ------------------------------------------------------ shard-level resume (#20)


def test_completed_shards_are_empty_without_a_checkpoint(tmp_path: Path) -> None:
    store = CandidateProgressStore(tmp_path / "missing.json")
    assert store.load_completed_shards({"schema_version": 2}) == {}


def test_completed_shards_round_trip(tmp_path: Path) -> None:
    from landuse_sentence_relevance.domain.models import Source

    store = CandidateProgressStore(tmp_path / "progress.json")
    metadata = {"schema_version": 2, "fingerprint": "stable"}
    store.save([], metadata, completed_shards={Source.WEBSITE: frozenset({0, 1, 7})})

    assert store.load_completed_shards(metadata) == {Source.WEBSITE: frozenset({0, 1, 7})}


def test_a_checkpoint_written_before_shard_tracking_reports_nothing(tmp_path: Path) -> None:
    """The safe default: every shard is re-read rather than assumed done."""
    path = tmp_path / "progress.json"
    metadata = {"schema_version": 2, "fingerprint": "stable"}
    path.write_text(json.dumps({"metadata": metadata, "candidates": []}), encoding="utf-8")

    assert CandidateProgressStore(path).load_completed_shards(metadata) == {}


def test_a_changed_sample_count_invalidates_the_shard_record(tmp_path: Path) -> None:
    """Stricter than the candidate list, deliberately.

    A larger `remote_file_sample_count` is allowed to reuse *candidates*: they are still valid
    rows, just from a smaller draw. It cannot reuse *shard indices*, because the resolved shard
    list is a different list -- shard 3 of a 32-shard sample is not shard 3 of a 128-shard one,
    and skipping it would silently omit data the run believes it read.
    """
    from landuse_sentence_relevance.domain.models import Source

    path = tmp_path / "progress.json"
    saved = {"schema_version": 2, "sampling": {"remote_file_sample_count": 32}}
    CandidateProgressStore(path).save([], saved, completed_shards={Source.WEBSITE: frozenset({3})})

    expanded = {"schema_version": 2, "sampling": {"remote_file_sample_count": 128}}
    assert CandidateProgressStore(path).load_completed_shards(expanded) == {}
    assert CandidateProgressStore(path).load(expanded) == ()


def test_a_changed_pin_refuses_the_shard_record(tmp_path: Path) -> None:
    from landuse_sentence_relevance.domain.models import Source

    path = tmp_path / "progress.json"
    store = CandidateProgressStore(path)
    store.save([], {"schema_version": 2, "revision": "a"}, completed_shards={Source.WEBSITE: frozenset({1})})

    with pytest.raises(ValueError, match="metadata does not match"):
        store.load_completed_shards({"schema_version": 2, "revision": "b"})


def test_save_writes_the_shard_record_in_a_stable_shape(tmp_path: Path) -> None:
    """The payload is sorted and the indices are sorted, because an unstable checkpoint would
    rewrite itself on every save and make a diff of two runs meaningless."""
    from landuse_sentence_relevance.domain.models import Source

    path = tmp_path / "progress.json"
    CandidateProgressStore(path).save(
        [],
        {"schema_version": 2},
        completed_sources=[Source.WEBSITE],
        completed_shards={Source.WIKIPEDIA: frozenset({5, 1}), Source.WEBSITE: frozenset({2})},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["completed_shards"] == {"website": [2], "wikipedia": [1, 5]}
    assert list(payload["completed_shards"]) == ["website", "wikipedia"]


def test_save_records_no_shards_when_none_are_given(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    CandidateProgressStore(path).save([], {"schema_version": 2})

    assert json.loads(path.read_text(encoding="utf-8"))["completed_shards"] == {}


def test_a_malformed_shard_record_reports_nothing(tmp_path: Path) -> None:
    """Same rule the rest of the checkpoint follows: unreadable means re-read, never guess."""
    path = tmp_path / "progress.json"
    metadata = {"schema_version": 2}
    path.write_text(
        json.dumps({"metadata": metadata, "candidates": [], "completed_shards": "website"}),
        encoding="utf-8",
    )

    assert CandidateProgressStore(path).load_completed_shards(metadata) == {}


def test_a_shard_entry_that_is_not_a_list_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    metadata = {"schema_version": 2}
    path.write_text(
        json.dumps(
            {"metadata": metadata, "candidates": [], "completed_shards": {"website": 3}},
        ),
        encoding="utf-8",
    )

    assert CandidateProgressStore(path).load_completed_shards(metadata) == {}


def test_completed_shards_are_read_with_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Asserted at ``read_text``: ``io.text_encoding`` turns ``None`` into ``utf-8`` wherever
    UTF-8 mode is on, so a spy any lower cannot tell a pinned encoding from an unpinned one."""
    from landuse_sentence_relevance.domain.models import Source

    path = tmp_path / "progress.json"
    store = CandidateProgressStore(path)
    metadata = {"schema_version": 2, "fingerprint": "encoding"}
    store.save([], metadata, completed_shards={Source.WEBSITE: frozenset({1})})
    requested: list[str | None] = []
    original_read_text = Path.read_text

    def spy_read_text(self, encoding=None, errors=None):
        requested.append(encoding)
        return original_read_text(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", spy_read_text)

    assert store.load_completed_shards(metadata) == {Source.WEBSITE: frozenset({1})}
    assert requested == ["utf-8"]
