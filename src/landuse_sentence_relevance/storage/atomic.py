from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol


class TextWriter(Protocol):
    def write(self, value: str, /) -> int: ...


def sync_directory(path: Path) -> None:
    """Persist directory entries, propagating unsupported or failed syncs."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def make_directory(path: Path) -> None:
    """Create missing parents and persist each new directory entry."""
    if path.exists():
        sync_directory(path.parent)
        return
    make_directory(path.parent)
    path.mkdir(exist_ok=True)
    sync_directory(path.parent)


def atomic_write(path: Path, writer: Callable[[TextWriter], object]) -> None:
    """Sync a sibling temporary file, replace the destination, then sync its directory."""
    make_directory(path.parent)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        sync_directory(path.parent)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
