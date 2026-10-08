from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

import landuse_sentence_relevance.storage.atomic as atomic_module
from landuse_sentence_relevance.storage.atomic import TextWriter, atomic_write


def test_atomic_write_replaces_the_destination(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("old", encoding="utf-8")

    atomic_write(path, lambda handle: handle.write("new"))

    assert path.read_text(encoding="utf-8") == "new"


def test_atomic_write_creates_parent_and_writes_utf8(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "state.txt"

    atomic_write(path, lambda handle: handle.write("Héllö — hills"))

    assert path.read_text(encoding="utf-8") == "Héllö — hills"


def test_atomic_write_preserves_the_destination_when_writing_fails(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("old", encoding="utf-8")

    def fail_after_writing(handle: TextWriter) -> None:
        handle.write("new")
        raise RuntimeError("write failed")

    with pytest.raises(RuntimeError, match="write failed"):
        atomic_write(path, fail_after_writing)

    assert path.read_text(encoding="utf-8") == "old"
    assert set(tmp_path.iterdir()) == {path}


def test_atomic_write_uses_a_sibling_utf8_tempfile_and_replaces_the_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")
    original_replace = atomic_module.os.replace
    create_tempfile = Mock(wraps=atomic_module.NamedTemporaryFile)
    monkeypatch.setattr(atomic_module, "NamedTemporaryFile", create_tempfile)

    def replace(source: Path, destination: Path) -> None:
        assert source != destination == path
        assert source.parent == destination.parent
        assert source.read_bytes() == "Héllö — hills".encode()
        assert destination.read_bytes() == b"old"
        original_replace(source, destination)

    monkeypatch.setattr(atomic_module.os, "replace", replace)

    atomic_write(path, lambda handle: handle.write("Héllö — hills"))

    create_tempfile.assert_called_once_with(
        mode="w", encoding="utf-8", dir=tmp_path, prefix=".state.json.", suffix=".tmp", delete=False
    )
    assert path.read_bytes() == "Héllö — hills".encode()
    assert set(tmp_path.iterdir()) == {path}


def test_atomic_write_reraises_a_tempfile_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")

    def fail(**_kwargs: object) -> object:
        raise OSError("cannot create temporary file")

    monkeypatch.setattr(atomic_module, "NamedTemporaryFile", fail)

    with pytest.raises(OSError, match="cannot create temporary file") as error:
        atomic_write(path, lambda _handle: None)

    assert not hasattr(error.value, "__notes__")
    assert path.read_bytes() == b"old"
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("remove_temporary", [False, True])
def test_atomic_write_cleans_up_after_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remove_temporary: bool
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")
    failure = RuntimeError("replace failed")

    def replace(source: Path, destination: Path) -> None:
        assert source != destination == path
        assert source.parent == destination.parent
        assert source.read_bytes() == b"new"
        assert destination.read_bytes() == b"old"
        if remove_temporary:
            source.unlink()
        raise failure

    monkeypatch.setattr(atomic_module.os, "replace", replace)

    with pytest.raises(RuntimeError, match="replace failed") as error:
        atomic_write(path, lambda handle: handle.write("new"))

    assert error.value is failure
    assert not hasattr(failure, "__notes__")
    assert path.read_bytes() == b"old"
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("writer_failure", [RuntimeError("write failed"), KeyboardInterrupt()])
def test_atomic_write_keeps_the_original_error_when_tempfile_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer_failure: BaseException
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")
    original_unlink = Path.unlink
    cleanup_failure = PermissionError("read-only directory")

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name.startswith(f".{path.name}."):
            raise cleanup_failure
        original_unlink(self, missing_ok=missing_ok)

    def fail_after_writing(handle: TextWriter) -> None:
        handle.write("new")
        raise writer_failure

    monkeypatch.setattr(Path, "unlink", unlink)

    with pytest.raises(type(writer_failure)) as error:
        atomic_write(path, fail_after_writing)

    assert error.value is writer_failure
    assert any("read-only directory" in note for note in error.value.__notes__)
    assert path.read_bytes() == b"old"
    leftovers = [child for child in tmp_path.iterdir() if child != path]
    assert len(leftovers) == 1
    assert leftovers[0].name.startswith(f".{path.name}.")


def test_atomic_write_reports_directory_sync_failure_after_visible_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")
    original_replace = atomic_module.os.replace
    original_sync = atomic_module.sync_directory
    replaced = False

    def replace(source: Path, destination: Path) -> None:
        nonlocal replaced
        original_replace(source, destination)
        replaced = True

    def sync(directory: Path) -> None:
        if replaced:
            raise OSError("directory sync failed")
        original_sync(directory)

    monkeypatch.setattr(atomic_module.os, "replace", replace)
    monkeypatch.setattr(atomic_module, "sync_directory", sync)

    with pytest.raises(OSError, match="directory sync failed") as error:
        atomic_write(path, lambda handle: handle.write("new"))

    assert not hasattr(error.value, "__notes__")
    assert path.read_bytes() == b"new"
    assert set(tmp_path.iterdir()) == {path}
