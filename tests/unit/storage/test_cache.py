from pathlib import Path

import pytest

from landuse_sentence_relevance.storage.cache import CacheOwnershipError, ManagedCache


def test_prepare_marks_the_root_and_routes_hugging_face_caches_inside_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "nested" / "runtime-cache"
    environment: dict[str, str] = {}

    ManagedCache(root).prepare(environment)

    assert (root / ManagedCache.OWNER_MARKER).read_text(encoding="utf-8") == (
        "landuse-sentence-relevance-golden-human-set\n"
    )
    assert environment == {
        "HF_HOME": str(root / "huggingface"),
        "HF_HUB_CACHE": str(root / "huggingface" / "hub"),
        "HF_DATASETS_CACHE": str(root / "huggingface" / "datasets"),
        "HF_ASSETS_CACHE": str(root / "huggingface" / "assets"),
        "HF_XET_CACHE": str(root / "huggingface" / "xet"),
        "TRANSFORMERS_CACHE": str(root / "huggingface" / "transformers"),
        "TORCH_HOME": str(root / "torch"),
    }


def test_cleanup_removes_only_the_marked_application_root(tmp_path: Path) -> None:
    root = tmp_path / "runtime-cache"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")
    cache = ManagedCache(root)
    cache.prepare({})
    (root / "model.bin").write_bytes(b"weights")

    cache.cleanup()

    assert not root.exists()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_prepare_reuses_a_marked_root_for_resumable_annotation(tmp_path: Path) -> None:
    root = tmp_path / "runtime-cache"
    cache = ManagedCache(root)
    cache.prepare({})
    (root / "model.bin").write_bytes(b"weights")

    cache.prepare({})

    assert (root / "model.bin").read_bytes() == b"weights"


def test_prepare_refuses_a_foreign_nonempty_root(tmp_path: Path) -> None:
    root = tmp_path / "foreign-cache"
    root.mkdir()
    (root / "foreign.bin").write_bytes(b"foreign")

    with pytest.raises(CacheOwnershipError, match=f"cache root is not application-owned: {root}"):
        ManagedCache(root).prepare({})

    assert not (root / ManagedCache.OWNER_MARKER).exists()


def test_prepare_refuses_a_file_as_the_cache_root(tmp_path: Path) -> None:
    root = tmp_path / "cache-file"
    root.write_bytes(b"not a directory")

    with pytest.raises(CacheOwnershipError, match=f"cache root is not a directory: {root}"):
        ManagedCache(root).prepare({})


def test_cleanup_refuses_an_unmarked_root(tmp_path: Path) -> None:
    root = tmp_path / "foreign-cache"
    root.mkdir()
    (root / "keep.bin").write_bytes(b"foreign")

    with pytest.raises(CacheOwnershipError, match=f"cache root is not application-owned: {root}"):
        ManagedCache(root).cleanup()

    assert (root / "keep.bin").exists()


def test_prepare_and_cleanup_validate_the_resolved_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runtime-cache"
    cache = ManagedCache(root)
    observed = []
    original_validate_location = ManagedCache._validate_location

    def spy_validate_location(location: Path) -> None:
        observed.append(location)
        original_validate_location(location)

    monkeypatch.setattr(
        ManagedCache,
        "_validate_location",
        staticmethod(spy_validate_location),
    )
    cache.prepare({})
    assert observed == [root.resolve()]

    observed.clear()
    cache.cleanup()
    assert observed == [root.resolve()]


def test_managed_cache_rejects_a_symlinked_root_with_a_specific_error(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(CacheOwnershipError) as error:
        ManagedCache(link).prepare({})

    assert str(error.value) == "refusing to manage a symlinked cache root"


def test_managed_cache_rejects_broad_locations_with_a_specific_error() -> None:
    for root in (Path("/"), Path.home().resolve()):
        with pytest.raises(CacheOwnershipError) as error:
            ManagedCache._validate_location(root)

        assert str(error.value) == "refusing to manage a broad cache root"


def test_prepare_writes_the_owner_marker_as_utf8(tmp_path: Path, monkeypatch) -> None:
    encodings = []
    original_write_text = Path.write_text

    def spy_write_text(self, *args, **kwargs):
        encodings.append(kwargs.get("encoding"))
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", spy_write_text)
    ManagedCache(tmp_path / "runtime-cache").prepare({})

    assert encodings == ["utf-8"]


def test_prepare_reads_and_validates_an_existing_owner_marker_as_utf8(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runtime-cache"
    root.mkdir()
    (root / ManagedCache.OWNER_MARKER).write_text("foreign\n", encoding="utf-8")
    encodings = []
    original_read_text = Path.read_text

    def spy_read_text(self, *args, **kwargs):
        encodings.append(kwargs.get("encoding"))
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy_read_text)

    with pytest.raises(CacheOwnershipError, match=f"cache root is not application-owned: {root}"):
        ManagedCache(root).prepare({})

    assert encodings == ["utf-8"]
