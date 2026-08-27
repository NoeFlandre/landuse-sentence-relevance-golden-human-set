from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from landuse_sentence_relevance.domain.models import Annotation
from landuse_sentence_relevance.storage.atomic import TextWriter, atomic_write


class AnnotationStore:
    """Persist only records that the annotator has explicitly labeled."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def record(self, annotation: Annotation) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            self._write_annotation(handle, annotation)

    def save(self, annotations: Iterable[Annotation]) -> None:
        """Replace the persisted annotation set with the supplied records."""

        def write_annotations(handle: TextWriter) -> None:
            for annotation in annotations:
                self._write_annotation(handle, annotation)

        atomic_write(self._path, write_annotations)

    @staticmethod
    def _write_annotation(handle: TextWriter, annotation: Annotation) -> None:
        handle.write(json.dumps(annotation.to_dict(), ensure_ascii=False, sort_keys=True))
        handle.write("\n")

    def load(self) -> dict[str, Annotation]:
        if not self._path.exists():
            return {}
        annotations: dict[str, Annotation] = {}
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                annotation = Annotation.from_dict(json.loads(line))
                annotations[annotation.candidate.candidate_id] = annotation
        return annotations
