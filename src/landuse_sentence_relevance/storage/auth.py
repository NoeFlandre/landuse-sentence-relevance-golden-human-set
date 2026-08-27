from __future__ import annotations

import os
import shutil
from collections.abc import MutableMapping
from pathlib import Path


class HuggingFaceAuthError(RuntimeError):
    """Raised when the persistent Hugging Face auth location is unsafe to use."""


class HuggingFaceAuth:
    """Keep Hugging Face credentials persistent while model/data caches remain disposable."""

    _AUTH_FILES = ("token", "stored_tokens")

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser()

    @property
    def root(self) -> Path:
        return self._root

    def prepare(
        self,
        environment: MutableMapping[str, str] | None = None,
        legacy_home: Path | None = None,
        token: str | None = None,
    ) -> bool:
        """Prepare auth variables and migrate an older application-cache login once."""
        root = self._absolute_root()
        self._validate_location(root)
        self._ensure_directory(root)
        root.chmod(0o700)
        if legacy_home is not None:
            self._migrate(legacy_home, root)

        target_environment = os.environ if environment is None else environment
        target_environment["HF_HOME"] = str(root)
        target_environment["HF_TOKEN_PATH"] = str(root / "token")
        target_environment["HF_STORED_TOKENS_PATH"] = str(root / "stored_tokens")
        return bool(token or target_environment.get("HF_TOKEN")) or self._has_saved_token(root)

    def _absolute_root(self) -> Path:
        if self._root.is_symlink():
            raise HuggingFaceAuthError("refusing to manage a symlinked auth root")
        return self._root.resolve()

    @staticmethod
    def _validate_location(root: Path) -> None:
        if root in {Path("/"), Path.home().resolve()}:
            raise HuggingFaceAuthError("refusing to manage a broad auth root")

    @staticmethod
    def _ensure_directory(root: Path) -> None:
        if root.exists() and not root.is_dir():
            raise HuggingFaceAuthError(f"auth root is not a directory: {root}")
        root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _migrate(cls, legacy_home: Path, destination_root: Path) -> None:
        legacy_root = legacy_home.expanduser()
        if legacy_root.is_symlink():
            return
        if not legacy_root.is_dir():
            return
        legacy_root = legacy_root.resolve()
        for filename in cls._AUTH_FILES:
            cls._migrate_file(legacy_root / filename, destination_root / filename)

    @staticmethod
    def _migrate_file(source: Path, destination: Path) -> None:
        if source.is_symlink():
            return
        if not source.is_file():
            return
        if destination.exists():
            return
        shutil.copyfile(source, destination)
        destination.chmod(0o600)

    @staticmethod
    def _has_saved_token(root: Path) -> bool:
        token_path = root / "token"
        return not token_path.is_symlink() and token_path.is_file() and token_path.stat().st_size > 0
