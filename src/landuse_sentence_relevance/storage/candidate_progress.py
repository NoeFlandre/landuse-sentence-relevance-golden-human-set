from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from landuse_sentence_relevance.domain.models import Candidate
from landuse_sentence_relevance.storage.atomic import TextWriter, atomic_write

_IGNORED_METADATA_KEYS = {
    "sentence_splitter": frozenset({"batch_size", "workers"}),
    "sampling": frozenset({"website_max_text_characters"}),
}


class CandidateProgressStore:
    """Persist compact candidate checkpoints while a pool is being built."""

    def __init__(self, path: Path) -> None:
        self._path = path.expanduser()

    def load(self, expected_metadata: Mapping[str, Any]) -> tuple[Candidate, ...] | None:
        if not self._path.exists():
            return None
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        saved_metadata = payload.get("metadata")
        if not isinstance(saved_metadata, Mapping) or not _metadata_matches(
            saved_metadata, expected_metadata
        ):
            raise ValueError("candidate progress metadata does not match the current configuration")
        return tuple(Candidate.from_dict(dict(row)) for row in payload["candidates"])

    def save(self, candidates: Iterable[Candidate], metadata: Mapping[str, Any]) -> None:
        payload = {
            "metadata": dict(metadata),
            "candidates": [candidate.to_dict() for candidate in candidates],
        }

        def write_payload(handle: TextWriter) -> None:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")

        atomic_write(self._path, write_payload)


def _semantic_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _without_keys(value, _IGNORED_METADATA_KEYS[key]) if key in _IGNORED_METADATA_KEYS else value
        for key, value in metadata.items()
    }


def _metadata_matches(saved: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    saved_semantic = _semantic_metadata(saved)
    expected_semantic = _semantic_metadata(expected)
    return saved_semantic == expected_semantic or _is_remote_sample_expansion(
        saved_semantic, expected_semantic
    )


def _is_remote_sample_expansion(
    saved: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    counts = _remote_sample_counts(saved, expected)
    if counts is None:
        return False
    saved_count, expected_count = counts
    return saved_count < expected_count and _without_remote_sample_count(
        saved
    ) == _without_remote_sample_count(expected)


def _remote_sample_counts(
    saved: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> tuple[int, int] | None:
    saved_count = _sample_count(saved)
    expected_count = _sample_count(expected)
    if saved_count is None or expected_count is None:
        return None
    return saved_count, expected_count


def _sample_count(metadata: Mapping[str, Any]) -> int | None:
    sampling = metadata.get("sampling")
    if not isinstance(sampling, Mapping):
        return None
    count = sampling.get("remote_file_sample_count")
    return count if isinstance(count, int) else None


def _without_remote_sample_count(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    sampling = result.get("sampling")
    if isinstance(sampling, Mapping):
        sampling_without_count = dict(sampling)
        sampling_without_count.pop("remote_file_sample_count", None)
        result["sampling"] = sampling_without_count
    return result


def _without_keys(value: Any, ignored_keys: frozenset[str]) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {key: item for key, item in value.items() if key not in ignored_keys}
