from __future__ import annotations

from pathlib import Path

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
    replacements: list[Path] = []

    def replace(source: Path, destination: Path) -> None:
        replacements.append(source)
        assert source != destination == path
        assert source.parent == destination.parent
        assert source.read_bytes() == "Héllö — hills".encode()
        assert destination.read_bytes() == b"old"
        original_replace(source, destination)

    monkeypatch.setattr(atomic_module.os, "replace", replace)

    atomic_write(path, lambda handle: handle.write("Héllö — hills"))

    assert len(replacements) == 1
    assert path.read_bytes() == "Héllö — hills".encode()
    assert set(tmp_path.iterdir()) == {path}


def test_atomic_write_reraises_a_tempfile_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")

    def fail(**_kwargs: object) -> object:
        raise OSError("cannot create temporary file")

    monkeypatch.setattr(atomic_module, "NamedTemporaryFile", fail)

    with pytest.raises(OSError, match="cannot create temporary file"):
        atomic_write(path, lambda _handle: None)

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
    assert path.read_bytes() == b"old"
    assert set(tmp_path.iterdir()) == {path}
