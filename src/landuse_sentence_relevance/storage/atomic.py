from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol


class TextWriter(Protocol):
    def write(self, value: str, /) -> int: ...


def atomic_write(path: Path, writer: Callable[[TextWriter], object]) -> None:
    """Write a file through a sibling temporary file and replace it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
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
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
