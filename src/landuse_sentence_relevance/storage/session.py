from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from tempfile import mkstemp

from landuse_sentence_relevance.domain.models import Annotation
from landuse_sentence_relevance.storage.atomic import TextWriter, atomic_write, make_directory, sync_directory

logger = logging.getLogger(__name__)


def _add_annotation(annotations: dict[str, Annotation], line: bytes) -> None:
    if line.strip():
        annotation = Annotation.from_dict(json.loads(line.decode()))
        annotations[annotation.candidate.candidate_id] = annotation


def _tail_text(tail: bytes) -> str:
    try:
        return tail.decode()
    except UnicodeDecodeError as error:
        if error.reason != "unexpected end of data":
            raise
        return tail[: error.start].decode()


def _is_torn_object(tail: bytes) -> bool:
    text = _tail_text(tail)
    if not text.lstrip().startswith("{"):
        return False
    try:
        json.loads(text)
    except json.JSONDecodeError as error:
        return error.pos == len(text) or error.msg == "Unterminated string starting at"
    return False


class AnnotationStore:
    """Persist only records that the annotator has explicitly labeled."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def record(self, annotation: Annotation) -> None:
        _, data = self._load()
        make_directory(self._path.parent)
        with self._path.open("a", encoding="utf-8") as handle:
            if data and not data.endswith(b"\n"):
                handle.write("\n")
            self._write_annotation(handle, annotation)
            handle.flush()
            os.fsync(handle.fileno())
        sync_directory(self._path.parent)

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
        """Load records, preserving an unambiguously torn final fragment before recovery."""
        annotations, _ = self._load()
        return annotations

    def _load(self) -> tuple[dict[str, Annotation], bytes]:
        try:
            data = self._path.read_bytes()
        except FileNotFoundError:
            return {}, b""
        prefix, separator, tail = data.rpartition(b"\n")
        prefix += separator
        annotations: dict[str, Annotation] = {}
        for line in prefix.split(b"\n"):
            _add_annotation(annotations, line)
        try:
            _add_annotation(annotations, tail)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not _is_torn_object(tail):
                raise
            self._recover(data, prefix)
            return self._load()
        return annotations, data

    def _recover(self, original: bytes, prefix: bytes) -> None:
        descriptor, recovery_path = mkstemp(
            dir=self._path.parent, prefix=f".{self._path.name}.", suffix=".recovery"
        )
        try:
            handle = os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            raise
        with handle:
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
        sync_directory(self._path.parent)
        atomic_write(self._path, lambda handle: handle.write(prefix.decode()))
        logger.warning(
            "Recovered torn annotation tail in %s; original bytes preserved in %s", self._path, recovery_path
        )
