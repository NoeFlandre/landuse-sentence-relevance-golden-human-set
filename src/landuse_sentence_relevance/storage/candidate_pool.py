from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from landuse_sentence_relevance.domain.models import Candidate
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.storage.atomic import TextWriter, atomic_write


class CandidatePoolStore:
    """Persist the deterministic, one-candidate-per-cell annotation bank."""

    def __init__(self, path: Path) -> None:
        self._path = path.expanduser()

    def load(self, expected_metadata: Mapping[str, Any]) -> FinalizedCandidatePool | None:
        if not self._path.exists():
            return None
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata")
        if metadata != dict(expected_metadata):
            raise ValueError("candidate pool metadata does not match the current configuration")
        candidates = tuple(Candidate.from_dict(dict(row)) for row in payload["candidates"])
        cells = tuple(str(cell) for cell in payload["cells"])
        return FinalizedCandidatePool(candidates=candidates, cells=cells)

    def save(self, pool: FinalizedCandidatePool, metadata: Mapping[str, Any]) -> None:
        payload = {
            "metadata": dict(metadata),
            "candidates": [candidate.to_dict() for candidate in pool.candidates],
            "cells": list(pool.cells),
        }

        def write_payload(handle: TextWriter) -> None:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")

        atomic_write(self._path, write_payload)

    def load_or_build(
        self,
        metadata: Mapping[str, Any],
        builder: Callable[[], FinalizedCandidatePool],
    ) -> FinalizedCandidatePool:
        cached = self.load(metadata)
        if cached is not None:
            return cached
        built = builder()
        self.save(built, metadata)
        return built
