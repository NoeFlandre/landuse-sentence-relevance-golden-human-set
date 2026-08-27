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
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_atomic_write_uses_a_sibling_utf8_tempfile_and_replaces_the_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    temporary_path = tmp_path / ".state.json.fake.tmp"
    captured_tempfile: dict[str, object] = {}
    captured_replace: list[tuple[Path, Path]] = []
    writes: list[str] = []

    class Handle:
        name = str(temporary_path)

        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def write(self, value: str) -> int:
            writes.append(value)
            return len(value)

    def named_temporary_file(**kwargs: object) -> Handle:
        captured_tempfile.update(kwargs)
        return Handle()

    def replace(source: Path, destination: Path) -> None:
        captured_replace.append((source, destination))

    monkeypatch.setattr(atomic_module, "NamedTemporaryFile", named_temporary_file)
    monkeypatch.setattr(atomic_module.os, "replace", replace)

    atomic_write(path, lambda handle: handle.write("new"))

    assert captured_tempfile == {
        "mode": "w",
        "encoding": "utf-8",
        "dir": tmp_path,
        "prefix": ".state.json.",
        "suffix": ".tmp",
        "delete": False,
    }
    assert writes == ["new"]
    assert captured_replace == [(temporary_path, path)]


def test_atomic_write_reraises_a_tempfile_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "state.json"

    def fail(**_kwargs: object) -> object:
        raise OSError("cannot create temporary file")

    monkeypatch.setattr(atomic_module, "NamedTemporaryFile", fail)

    with pytest.raises(OSError, match="cannot create temporary file"):
        atomic_write(path, lambda _handle: None)


def test_atomic_write_cleans_up_after_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    temporary_path = tmp_path / ".state.json.fake.tmp"
    unlink_calls: list[tuple[Path, bool | None]] = []

    class Handle:
        name = str(temporary_path)

        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def write(self, _value: str) -> int:
            return 0

    def replace(_source: Path, _destination: Path) -> None:
        raise RuntimeError("replace failed")

    def unlink(path: Path, missing_ok: bool | None = None) -> None:
        unlink_calls.append((path, missing_ok))

    monkeypatch.setattr(atomic_module, "NamedTemporaryFile", lambda **_kwargs: Handle())
    monkeypatch.setattr(atomic_module.os, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)

    with pytest.raises(RuntimeError, match="replace failed"):
        atomic_write(path, lambda _handle: None)

    assert unlink_calls == [(temporary_path, True)]
