from __future__ import annotations

import json
import os
import stat

import pytest
from tests.builders import make_annotations

from landuse_sentence_relevance.storage.atomic import atomic_write
from landuse_sentence_relevance.storage.session import AnnotationStore


def encoded(annotation) -> bytes:
    return (json.dumps(annotation.to_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


@pytest.mark.parametrize(
    "tail", [b'{"label":', b'{"sentence":"caf\xc3', b'{"label":"ye', b' \t{"sentence":"caf\xc3']
)
def test_recovery_preserves_original_bytes_and_supports_append_reload(tmp_path, caplog, tail) -> None:
    path = tmp_path / "annotations.jsonl"
    first, second = make_annotations()[:2]
    original = encoded(first) + tail
    path.write_bytes(original)

    assert AnnotationStore(path).load() == {first.candidate.candidate_id: first}
    assert path.read_bytes() == encoded(first)
    backups = list(tmp_path.glob(".annotations.jsonl.*.recovery"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert "recover" in caplog.text.lower()
    assert first.candidate.sentence not in caplog.text

    AnnotationStore(path).record(second)

    assert AnnotationStore(path).load() == {
        first.candidate.candidate_id: first,
        second.candidate.candidate_id: second,
    }
    assert path.read_bytes() == encoded(first) + encoded(second)
    assert list(tmp_path.glob(".annotations.jsonl.*.recovery")) == backups
    assert backups[0].read_bytes() == original


@pytest.mark.parametrize("bad", [b'{"label":\n', b'{"label": nope}', b'{"label":"\xff"}', b"{}", b"[]"])
def test_corrupt_complete_or_ambiguous_records_remain_untouched(tmp_path, bad) -> None:
    path = tmp_path / "annotations.jsonl"
    original = encoded(make_annotations()[0]) + bad
    path.write_bytes(original)

    with pytest.raises((ValueError, KeyError, TypeError)):
        AnnotationStore(path).load()

    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}


def test_valid_unterminated_record_survives_append(tmp_path) -> None:
    path = tmp_path / "annotations.jsonl"
    first, second = make_annotations()[:2]
    path.write_bytes(encoded(first).rstrip(b"\n"))

    assert AnnotationStore(path).load() == {first.candidate.candidate_id: first}
    AnnotationStore(path).record(second)

    assert path.read_bytes() == encoded(first) + encoded(second)
    assert AnnotationStore(path).load() == {
        first.candidate.candidate_id: first,
        second.candidate.candidate_id: second,
    }


def test_append_syncs_written_bytes_then_directory_before_success(tmp_path, monkeypatch) -> None:
    path = tmp_path / "annotations.jsonl"
    annotation = make_annotations()[0]
    synced = []
    real_fsync = os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode) and os.fstat(fd).st_ino != tmp_path.stat().st_ino:
            real_fsync(fd)
            return
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced.append("directory")
        else:
            assert path.read_bytes() == encoded(annotation)
            synced.append("annotation")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    AnnotationStore(path).record(annotation)

    assert path.read_bytes() == encoded(annotation)
    assert synced == ["annotation", "directory"]


def test_atomic_replace_syncs_payload_before_replace_and_directory_after(tmp_path, monkeypatch) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode) and os.fstat(fd).st_ino != tmp_path.stat().st_ino:
            real_fsync(fd)
            return
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            assert path.read_bytes() == b"new"
            events.append("directory")
        else:
            assert os.pread(fd, 10, 0) == b"new"
            assert path.read_bytes() == b"old"
            events.append("file")
        real_fsync(fd)

    def replace(source, destination):
        assert events == ["file"]
        real_replace(source, destination)
        events.append("replace")

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    atomic_write(path, lambda handle: handle.write("new"))

    assert path.read_bytes() == b"new"
    assert events == ["file", "replace", "directory"]


@pytest.mark.parametrize("operation", ["append", "replace"])
@pytest.mark.parametrize("failed_sync", [1, 2])
def test_write_sync_failure_is_reported_without_claiming_success(
    tmp_path, monkeypatch, operation, failed_sync
) -> None:
    path = tmp_path / "annotations.jsonl"
    first, second = make_annotations()[:2]
    original = encoded(first)
    path.write_bytes(original)
    calls = 0
    real_fsync = os.fsync

    def fsync(fd):
        nonlocal calls
        if stat.S_ISDIR(os.fstat(fd).st_mode) and os.fstat(fd).st_ino != tmp_path.stat().st_ino:
            real_fsync(fd)
            return
        calls += 1
        if calls == failed_sync:
            raise OSError("injected sync failure")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    store = AnnotationStore(path)
    with pytest.raises(OSError, match="injected sync failure"):
        if operation == "append":
            store.record(second)
        else:
            store.save([second])

    expected = original + encoded(second) if operation == "append" else encoded(second)
    if operation == "replace" and failed_sync == 1:
        expected = original
    assert path.read_bytes() == expected
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("failed_sync", [1, 2, 3, 4])
def test_recovery_sync_failure_never_loses_the_original_bytes(tmp_path, monkeypatch, failed_sync) -> None:
    path = tmp_path / "annotations.jsonl"
    first = make_annotations()[0]
    original = encoded(first) + b'{"sentence":"caf\xc3'
    path.write_bytes(original)
    calls = 0
    real_fsync = os.fsync

    def fsync(fd):
        nonlocal calls
        if stat.S_ISDIR(os.fstat(fd).st_mode) and os.fstat(fd).st_ino != tmp_path.stat().st_ino:
            real_fsync(fd)
            return
        calls += 1
        if calls == failed_sync:
            raise OSError("injected sync failure")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(OSError, match="injected sync failure"):
        AnnotationStore(path).load()

    assert path.read_bytes() == (encoded(first) if failed_sync == 4 else original)
    backups = list(tmp_path.glob(".annotations.jsonl.*.recovery"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("crash_after_sync", [1, 2, 3, 4])
def test_process_crash_during_recovery_keeps_a_restartable_log_and_original_bytes(
    tmp_path, crash_after_sync
) -> None:
    import subprocess
    import sys

    path = tmp_path / "annotations.jsonl"
    first, second = make_annotations()[:2]
    original = encoded(first) + b'{"sentence":"caf\xe2\x82'
    path.write_bytes(original)
    script = """
import os
import stat
import sys
from pathlib import Path
from landuse_sentence_relevance.storage.session import AnnotationStore
original_fsync = os.fsync
calls = 0
def crash(fd):
    global calls
    original_fsync(fd)
    if stat.S_ISDIR(os.fstat(fd).st_mode) and os.fstat(fd).st_ino != Path(sys.argv[1]).parent.stat().st_ino:
        return
    calls += 1
    if calls == int(sys.argv[2]):
        os._exit(73)
os.fsync = crash
AnnotationStore(Path(sys.argv[1])).load()
"""

    result = subprocess.run([sys.executable, "-c", script, str(path), str(crash_after_sync)], check=False)

    assert result.returncode == 73
    assert path.read_bytes() == (encoded(first) if crash_after_sync == 4 else original)
    assert original in [backup.read_bytes() for backup in tmp_path.glob(".annotations.jsonl.*.recovery")]
    assert AnnotationStore(path).load() == {first.candidate.candidate_id: first}
    AnnotationStore(path).record(second)
    assert path.read_bytes() == encoded(first) + encoded(second)
    assert AnnotationStore(path).load() == {
        first.candidate.candidate_id: first,
        second.candidate.candidate_id: second,
    }
    assert original in [backup.read_bytes() for backup in tmp_path.glob(".annotations.jsonl.*.recovery")]


@pytest.mark.parametrize("tail", [b"{", b'{"label": "yes",', b'  {"label":"ye', b'{"a":1'])
def test_recovery_of_a_first_torn_record_preserves_it(tmp_path, tail) -> None:
    path = tmp_path / "annotations.jsonl"
    path.write_bytes(tail)

    assert AnnotationStore(path).load() == {}
    assert path.read_bytes() == b""
    backups = list(tmp_path.glob(".annotations.jsonl.*.recovery"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == tail
    annotation = make_annotations()[0]
    AnnotationStore(path).record(annotation)
    assert path.read_bytes() == encoded(annotation)
    assert AnnotationStore(path).load() == {annotation.candidate.candidate_id: annotation}


@pytest.mark.parametrize(
    "tail", [b'"unterminated', b'{"a": invalid', b'{"a":"\\x', b'{"a":"\\u00', b"{}\xc3"]
)
def test_ambiguous_unterminated_corruption_stays_loud(tmp_path, tail) -> None:
    path = tmp_path / "annotations.jsonl"
    path.write_bytes(tail)

    with pytest.raises(ValueError):
        AnnotationStore(path).record(make_annotations()[0])

    assert path.read_bytes() == tail
    assert set(tmp_path.iterdir()) == {path}


def test_invalid_earlier_record_prevents_any_tail_recovery(tmp_path) -> None:
    path = tmp_path / "annotations.jsonl"
    original = b'{"invalid": true}\n' + encoded(make_annotations()[0]) + b'{"label":"ye'
    path.write_bytes(original)

    with pytest.raises(KeyError):
        AnnotationStore(path).load()

    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}


def test_recovery_keeps_valid_prefix_bytes_duplicates_blank_lines_and_crlf(tmp_path) -> None:
    path = tmp_path / "annotations.jsonl"
    first, second = make_annotations()[:2]
    prefix = b" \r\n" + encoded(first).replace(b"\n", b"\r\n") + b"\n" + encoded(first) + encoded(second)
    original = prefix + b'{"label":"ye'
    path.write_bytes(original)

    assert AnnotationStore(path).load() == {
        first.candidate.candidate_id: first,
        second.candidate.candidate_id: second,
    }
    assert path.read_bytes() == prefix
    assert next(tmp_path.glob(".annotations.jsonl.*.recovery")).read_bytes() == original


def test_unicode_line_separator_inside_a_valid_sentence_is_preserved(tmp_path) -> None:
    from dataclasses import replace

    path = tmp_path / "annotations.jsonl"
    first = make_annotations()[0]
    annotation = replace(first, candidate=replace(first.candidate, sentence="hills\u2028and fields"))
    AnnotationStore(path).record(annotation)

    assert AnnotationStore(path).load() == {annotation.candidate.candidate_id: annotation}
    assert path.read_bytes() == encoded(annotation)


def test_recovery_syncs_complete_backup_before_replacing_the_log(tmp_path, monkeypatch, caplog) -> None:
    path = tmp_path / "annotations.jsonl"
    first = make_annotations()[0]
    prefix = encoded(first)
    original = prefix + b'{"label":"ye'
    path.write_bytes(original)
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode) and os.fstat(fd).st_ino != tmp_path.stat().st_ino:
            real_fsync(fd)
            return
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            assert os.fstat(fd).st_ino == tmp_path.stat().st_ino
            events.append("directory")
        else:
            backups = list(tmp_path.glob(".annotations.jsonl.*.recovery"))
            assert len(backups) == 1
            assert backups[0].read_bytes() == original
            events.append("file")
        real_fsync(fd)

    def replace(source, destination):
        assert destination == path
        assert path.read_bytes() == original
        assert source.read_bytes() == prefix
        assert events == ["file", "directory", "file"]
        real_replace(source, destination)
        events.append("replace")

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    assert AnnotationStore(path).load() == {first.candidate.candidate_id: first}

    assert events == ["file", "directory", "file", "replace", "directory"]
    assert path.read_bytes() == prefix
    backup = next(tmp_path.glob(".annotations.jsonl.*.recovery"))
    assert str(path) in caplog.text
    assert str(backup) in caplog.text
    assert caplog.messages == [
        f"Recovered torn annotation tail in {path}; original bytes preserved in {backup}"
    ]
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"


@pytest.mark.parametrize("failure", [False, True])
def test_directory_descriptors_close_even_on_sync_failure(tmp_path, monkeypatch, failure) -> None:
    from landuse_sentence_relevance.storage.atomic import sync_directory

    opened = []
    real_open, real_fsync = os.open, os.fsync

    def open_directory(path, flags):
        descriptor = real_open(path, flags)
        opened.append(descriptor)
        return descriptor

    def fsync(fd):
        assert stat.S_ISDIR(os.fstat(fd).st_mode)
        assert os.fstat(fd).st_ino == tmp_path.stat().st_ino
        if failure:
            raise OSError("directory sync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "open", open_directory)
    monkeypatch.setattr(os, "fsync", fsync)
    if failure:
        with pytest.raises(OSError, match="directory sync failed"):
            sync_directory(tmp_path)
    else:
        sync_directory(tmp_path)

    assert len(opened) == 1
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(opened[0])


def test_new_parent_directory_entries_are_synced_before_annotation_success(tmp_path, monkeypatch) -> None:
    path = tmp_path / "nested" / "state" / "annotations.jsonl"
    annotation = make_annotations()[0]
    directories = []
    real_fsync = os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directories.append(os.fstat(fd).st_ino)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    AnnotationStore(path).record(annotation)

    assert path.read_bytes() == encoded(annotation)
    assert directories[-3:] == [
        tmp_path.stat().st_ino,
        path.parent.parent.stat().st_ino,
        path.parent.stat().st_ino,
    ]


def test_recovery_can_append_directly_without_a_prior_load(tmp_path) -> None:
    path = tmp_path / "annotations.jsonl"
    first, second = make_annotations()[:2]
    original = encoded(first) + b'{"sentence":"caf\xc3'
    path.write_bytes(original)

    AnnotationStore(path).record(second)

    assert path.read_bytes() == encoded(first) + encoded(second)
    assert AnnotationStore(path).load() == {
        first.candidate.candidate_id: first,
        second.candidate.candidate_id: second,
    }
    assert next(tmp_path.glob(".annotations.jsonl.*.recovery")).read_bytes() == original


@pytest.mark.parametrize(
    "tail",
    [b'{"a":"broken\t', b'{"a":"broken\r', b'{"a":\x0b', '{"a":\u00a0'.encode("utf-8")],
)
def test_invalid_trailing_control_characters_are_not_hidden_by_recovery(tmp_path, tail) -> None:
    path = tmp_path / "annotations.jsonl"
    original = encoded(make_annotations()[0]) + tail
    path.write_bytes(original)

    with pytest.raises(json.JSONDecodeError):
        AnnotationStore(path).load()

    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}


def test_retry_after_failed_parent_creation_sync_persists_the_existing_entry(tmp_path, monkeypatch) -> None:
    path = tmp_path / "new-parent" / "nested" / "annotations.jsonl"
    annotation = make_annotations()[0]
    root_inode = tmp_path.stat().st_ino
    root_synced = False
    failed = False
    real_fsync = os.fsync

    def fsync(fd):
        nonlocal root_synced, failed
        if os.fstat(fd).st_ino == root_inode:
            if not failed:
                failed = True
                raise OSError("parent entry was not synced")
            root_synced = True
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(OSError, match="parent entry was not synced"):
        AnnotationStore(path).record(annotation)
    assert path.parent.parent.is_dir()
    assert not path.exists()

    AnnotationStore(path).record(annotation)

    assert root_synced
    assert path.read_bytes() == encoded(annotation)
    assert AnnotationStore(path).load() == {annotation.candidate.candidate_id: annotation}


def test_independent_sessions_can_share_a_concurrently_created_parent(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    directory = tmp_path / "shared-state"
    path = directory / "annotations.jsonl"
    annotation = make_annotations()[0]
    original_mkdir = Path.mkdir

    def mkdir(self, *args, **kwargs):
        if self == directory:
            original_mkdir(self)
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    AnnotationStore(path).record(annotation)

    assert path.read_bytes() == encoded(annotation)
    assert AnnotationStore(path).load() == {annotation.candidate.candidate_id: annotation}


def test_failed_recovery_file_wrapper_closes_the_owned_descriptor(tmp_path, monkeypatch) -> None:
    import landuse_sentence_relevance.storage.session as session_module

    path = tmp_path / "annotations.jsonl"
    original = encoded(make_annotations()[0]) + b'{"label":"ye'
    path.write_bytes(original)
    descriptors = []
    real_mkstemp = session_module.mkstemp

    def mkstemp(**kwargs):
        descriptor, name = real_mkstemp(**kwargs)
        descriptors.append(descriptor)
        return descriptor, name

    def fdopen(*args, **kwargs):
        raise OSError("recovery file wrapper failed")

    monkeypatch.setattr(session_module, "mkstemp", mkstemp)
    monkeypatch.setattr(os, "fdopen", fdopen)
    with pytest.raises(OSError, match="recovery file wrapper failed"):
        AnnotationStore(path).load()

    assert path.read_bytes() == original
    assert len(descriptors) == 1
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(descriptors[0])


@pytest.mark.parametrize(
    "tail",
    [
        b'{"place_name": n',
        b'{"place_name": nu',
        b'{"place_name": nul',
        b'{"flag": t',
        b'{"flag": tr',
        b'{"flag": tru',
        b'{"flag": f',
        b'{"flag": fa',
        b'{"flag": fal',
        b'{"flag": fals',
        b'{"latitude": -',
        b'{"latitude": 45.',
        b'{"latitude": -0.',
        b'{"latitude": -45.',
        b'{"longitude": 2e',
        b'{"longitude": 2E',
        b'{"longitude": 2e+',
        b'{"longitude": 2e-',
        b'{"longitude": 2E+',
        b'{"longitude": 2E-',
        b'{"longitude": -0.25e-',
    ],
)
def test_crash_inside_a_literal_or_number_preserves_bytes_and_can_resume(tmp_path, tail) -> None:
    path = tmp_path / "annotations.jsonl"
    first, second = make_annotations()[:2]
    original = encoded(first) + tail
    path.write_bytes(original)

    assert AnnotationStore(path).load() == {first.candidate.candidate_id: first}
    assert path.read_bytes() == encoded(first)
    assert next(tmp_path.glob(".annotations.jsonl.*.recovery")).read_bytes() == original

    AnnotationStore(path).record(second)

    assert path.read_bytes() == encoded(first) + encoded(second)
    assert AnnotationStore(path).load() == {
        first.candidate.candidate_id: first,
        second.candidate.candidate_id: second,
    }


@pytest.mark.parametrize(
    "tail",
    [
        b'{"place_name": nul ',
        b'{"flag": tru\t',
        b'{"flag": fals\r',
        b'{"latitude": - ',
        b'{"latitude": 45. ',
        b'{"longitude": 2e+ ',
        b'{"longitude": 2 e',
        b'{"longitude": 2.5.',
        b'{"longitude": 2e3.',
        b'{"longitude": 2e3e',
        b'{"longitude": 2e3E-',
        b'{"longitude": 2E3e',
        b'{"longitude": 2E3E-',
        b'{"longitude": 2E3.',
        b'{"longitude": 01.',
        b'{"longitude": 2e++',
        b'{"longitude": .',
        b'{"longitude": -.',
        b'{"longitude": "2".',
        b'{"longitude": truee',
        b'{"longitude": [2].',
        b'{"longitude": 2.}',
        b'{"place_name": nul}',
        b'{"place_name": nu\n',
        b'{"longitude": 2e+\n',
        b'{"longitude": 2.E',
        b'{"flag": nope',
    ],
)
def test_malformed_token_lookalikes_are_never_recovered(tmp_path, tail) -> None:
    path = tmp_path / "annotations.jsonl"
    original = encoded(make_annotations()[0]) + tail
    path.write_bytes(original)

    with pytest.raises(json.JSONDecodeError):
        AnnotationStore(path).record(make_annotations()[1])

    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("tail", [b'{"x":\xc3', b'{"x": n\xc3', b'{"x": 1.\xc3', b"{}\xe2\x82"])
def test_incomplete_utf8_outside_strings_is_not_a_valid_json_prefix(tmp_path, tail) -> None:
    path = tmp_path / "annotations.jsonl"
    original = encoded(make_annotations()[0]) + tail
    path.write_bytes(original)

    with pytest.raises(UnicodeDecodeError):
        AnnotationStore(path).load()

    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("backslashes", [1, 3, 5])
def test_truncated_utf8_cannot_complete_a_json_escape(tmp_path, backslashes) -> None:
    path = tmp_path / "annotations.jsonl"
    original = encoded(make_annotations()[0]) + b'{"x":"' + b"\\" * backslashes + b"\xc3"
    path.write_bytes(original)

    with pytest.raises(UnicodeDecodeError):
        AnnotationStore(path).load()

    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("backslashes", [0, 2, 4])
def test_truncated_utf8_after_escaped_backslashes_can_resume(tmp_path, backslashes) -> None:
    path = tmp_path / "annotations.jsonl"
    first, second = make_annotations()[:2]
    original = encoded(first) + b'{"x":"' + b"\\" * backslashes + b"\xf0\x9f"
    path.write_bytes(original)

    AnnotationStore(path).record(second)

    assert AnnotationStore(path).load() == {
        first.candidate.candidate_id: first,
        second.candidate.candidate_id: second,
    }
    assert path.read_bytes() == encoded(first) + encoded(second)
    assert next(tmp_path.glob(".annotations.jsonl.*.recovery")).read_bytes() == original
