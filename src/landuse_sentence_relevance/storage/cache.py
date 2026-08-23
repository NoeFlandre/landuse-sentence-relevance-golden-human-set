from __future__ import annotations

import os
import shutil
from collections.abc import MutableMapping
from pathlib import Path
from typing import ClassVar

from landuse_sentence_relevance.config import PROJECT_NAME


class CacheOwnershipError(RuntimeError):
    """Raised when the application cannot prove that a cache is its own."""


class ManagedCache:
    """Own one runtime cache root and remove it only after a successful upload."""

    OWNER_MARKER: ClassVar[str] = ".cache-owner"
    _ENVIRONMENT_PATHS: ClassVar[dict[str, Path]] = {
        "HF_HOME": Path("huggingface"),
        "HF_HUB_CACHE": Path("huggingface/hub"),
        "HF_DATASETS_CACHE": Path("huggingface/datasets"),
        "HF_ASSETS_CACHE": Path("huggingface/assets"),
        "HF_XET_CACHE": Path("huggingface/xet"),
        "TRANSFORMERS_CACHE": Path("huggingface/transformers"),
        "TORCH_HOME": Path("torch"),
    }

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser()

    @property
    def root(self) -> Path:
        return self._root

    def prepare(self, environment: MutableMapping[str, str] | None = None) -> None:
        """Claim the root and route model/data caches beneath it."""
        root = self._absolute_root()
        self._validate_location(root)
        self._claim(root)
        target_environment = os.environ if environment is None else environment
        for variable, relative_path in self._ENVIRONMENT_PATHS.items():
            target_environment[variable] = str(root / relative_path)

    def cleanup(self) -> None:
        """Delete this exact root after verifying its ownership marker."""
        root = self._absolute_root()
        if not root.exists():
            return
        self._validate_location(root)
        self._assert_owned(root)
        shutil.rmtree(root)

    def _absolute_root(self) -> Path:
        if self._root.is_symlink():
            raise CacheOwnershipError("refusing to manage a symlinked cache root")
        return self._root.resolve()

    @staticmethod
    def _validate_location(root: Path) -> None:
        if root in {Path("/"), Path.home().resolve()}:
            raise CacheOwnershipError("refusing to manage a broad cache root")

    def _claim(self, root: Path) -> None:
        self._ensure_directory(root)
        marker = root / self.OWNER_MARKER
        if marker.exists():
            self._assert_owned(root)
            return
        self._require_empty(root)
        marker.write_text(f"{PROJECT_NAME}\n", encoding="utf-8")

    @staticmethod
    def _ensure_directory(root: Path) -> None:
        if not root.exists():
            root.mkdir(parents=True)
            return
        if not root.is_dir():
            raise CacheOwnershipError(f"cache root is not a directory: {root}")

    @staticmethod
    def _require_empty(root: Path) -> None:
        if any(root.iterdir()):
            raise CacheOwnershipError(f"cache root is not application-owned: {root}")

    @staticmethod
    def _assert_owned(root: Path) -> None:
        marker = root / ManagedCache.OWNER_MARKER
        if not marker.is_file() or marker.read_text(encoding="utf-8") != f"{PROJECT_NAME}\n":
            raise CacheOwnershipError(f"cache root is not application-owned: {root}")
