from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from landuse_sentence_relevance.storage.auth import HuggingFaceAuth, HuggingFaceAuthError


def test_prepare_routes_hugging_face_auth_to_a_persistent_project_directory(tmp_path: Path) -> None:
    root = tmp_path / "config" / "huggingface"
    environment: dict[str, str] = {}

    authenticated = HuggingFaceAuth(root).prepare(environment)

    assert authenticated is False
    assert root.stat().st_mode & 0o777 == 0o700
    assert environment == {
        "HF_HOME": str(root),
        "HF_TOKEN_PATH": str(root / "token"),
        "HF_STORED_TOKENS_PATH": str(root / "stored_tokens"),
    }


def test_prepare_migrates_an_existing_runtime_login_without_deleting_it(tmp_path: Path) -> None:
    persistent_root = tmp_path / "persistent-auth"
    legacy_root = tmp_path / "runtime-cache" / "huggingface"
    legacy_root.mkdir(parents=True)
    (legacy_root / "token").write_text("secret-token\n", encoding="utf-8")
    (legacy_root / "stored_tokens").write_text("stored-secret\n", encoding="utf-8")
    environment: dict[str, str] = {}

    authenticated = HuggingFaceAuth(persistent_root).prepare(environment, legacy_root)

    assert authenticated is True
    assert (persistent_root / "token").read_text(encoding="utf-8") == "secret-token\n"
    assert (persistent_root / "stored_tokens").read_text(encoding="utf-8") == "stored-secret\n"
    assert (persistent_root / "token").stat().st_mode & 0o777 == 0o600
    assert (persistent_root / "stored_tokens").stat().st_mode & 0o777 == 0o600
    assert (legacy_root / "token").exists()
    assert (legacy_root / "stored_tokens").exists()


def test_prepare_never_replaces_a_persistent_login_with_a_legacy_login(tmp_path: Path) -> None:
    persistent_root = tmp_path / "persistent-auth"
    legacy_root = tmp_path / "runtime-cache" / "huggingface"
    legacy_root.mkdir(parents=True)
    persistent_root.mkdir(parents=True)
    (persistent_root / "token").write_text("new-token\n", encoding="utf-8")
    (legacy_root / "token").write_text("old-token\n", encoding="utf-8")

    HuggingFaceAuth(persistent_root).prepare({}, legacy_root)

    assert (persistent_root / "token").read_text(encoding="utf-8") == "new-token\n"


def test_prepare_recognizes_an_explicit_environment_token(tmp_path: Path) -> None:
    environment = {"HF_TOKEN": "provided-token"}

    authenticated = HuggingFaceAuth(tmp_path / "persistent-auth").prepare(environment)

    assert authenticated is True
    assert environment["HF_TOKEN"] == "provided-token"


def test_prepare_rejects_a_symlinked_auth_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(HuggingFaceAuthError) as error:
        HuggingFaceAuth(link).prepare({})

    assert str(error.value) == "refusing to manage a symlinked auth root"


def test_prepare_rejects_broad_auth_locations() -> None:
    for root in (Path("/"), Path.home().resolve()):
        with pytest.raises(HuggingFaceAuthError) as error:
            HuggingFaceAuth(root).prepare({})
        assert str(error.value) == "refusing to manage a broad auth root"


def test_prepare_rejects_a_file_as_the_auth_root(tmp_path: Path) -> None:
    root = tmp_path / "auth-file"
    root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(HuggingFaceAuthError) as error:
        HuggingFaceAuth(root).prepare({})

    assert str(error.value) == f"auth root is not a directory: {root}"


def test_prepare_ignores_a_missing_legacy_home(tmp_path: Path) -> None:
    root = tmp_path / "persistent-auth"

    assert HuggingFaceAuth(root).prepare({}, tmp_path / "missing-runtime-auth") is False


def test_prepare_ignores_a_symlinked_legacy_home(tmp_path: Path) -> None:
    target = tmp_path / "legacy-target"
    target.mkdir()
    legacy = tmp_path / "legacy-link"
    legacy.symlink_to(target, target_is_directory=True)

    assert HuggingFaceAuth(tmp_path / "persistent-auth").prepare({}, legacy) is False


def test_prepare_rejects_empty_saved_tokens(tmp_path: Path) -> None:
    root = tmp_path / "persistent-auth"
    root.mkdir()
    (root / "token").write_text("", encoding="utf-8")

    assert HuggingFaceAuth(root).prepare({}) is False


def test_prepare_accepts_a_one_byte_saved_token(tmp_path: Path) -> None:
    root = tmp_path / "persistent-auth"
    root.mkdir()
    (root / "token").write_text("x", encoding="utf-8")

    assert HuggingFaceAuth(root).prepare({}) is True


def test_has_saved_token_uses_the_hugging_face_token_filename() -> None:
    requested_names: list[str] = []

    class FakeTokenPath:
        def is_symlink(self) -> bool:
            return False

        def is_file(self) -> bool:
            return True

        def stat(self) -> SimpleNamespace:
            return SimpleNamespace(st_size=1)

    class FakeRoot:
        def __truediv__(self, name: str) -> FakeTokenPath:
            requested_names.append(name)
            return FakeTokenPath()

    assert HuggingFaceAuth._has_saved_token(cast(Path, FakeRoot())) is True
    assert requested_names == ["token"]
