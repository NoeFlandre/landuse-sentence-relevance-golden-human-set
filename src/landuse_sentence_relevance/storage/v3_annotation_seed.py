"""Atomic persistence for the deterministic V3 annotation seed."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from landuse_sentence_relevance.domain.v3_annotation import V3AnnotationSeed
from landuse_sentence_relevance.storage.atomic import TextWriter, atomic_write


class V3AnnotationSeedStore:
    """Persist a V3 seed only when its immutable run metadata still matches."""

    def __init__(self, path: Path) -> None:
        self._path = path.expanduser()

    def load(self, expected_metadata: Mapping[str, Any]) -> V3AnnotationSeed | None:
        if not self._path.exists():
            return None
        payload = json.loads(self._path.read_bytes().decode())
        if payload.get("metadata") != dict(expected_metadata):
            raise ValueError(
                "V3 annotation seed metadata does not match the current configuration; "
                "the benchmark SHA-256 may have changed"
            )
        state = payload.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("V3 annotation seed state must be an object")
        return V3AnnotationSeed.from_dict(dict(state))

    def save(self, seed: V3AnnotationSeed, metadata: Mapping[str, Any]) -> None:
        payload = {"metadata": dict(metadata), "state": seed.to_dict()}

        def write_payload(handle: TextWriter) -> None:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")

        atomic_write(self._path, write_payload)
