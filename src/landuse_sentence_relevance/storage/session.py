from __future__ import annotations

import json
import logging
import os
import re
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


def _tail_text(tail: bytes, error: UnicodeDecodeError) -> str:
    if error.reason != "unexpected end of data":
        raise error
    prefix = tail[: error.start].decode()
    if _is_unfinished_string(prefix):
        return prefix
    raise error


def _is_unfinished_string(text: str) -> bool:
    """Check whether a non-ASCII code point can continue the open string."""
    try:
        json.loads(text)
    except json.JSONDecodeError as error:
        if error.msg != "Unterminated string starting at":
            return False
        try:
            json.loads(text[error.pos :] + '�"')
        except json.JSONDecodeError:
            return False
        return True
    return False


def _has_number_completion(prefix: str, suffix: str) -> bool:
    number = re.search(r"[-+0-9.eE]+$", prefix)
    if number is None:
        return False
    try:
        json.loads(number.group() + suffix + "0")
    except json.JSONDecodeError:
        return False
    return True


def _is_torn_token(text: str, error: json.JSONDecodeError) -> bool:
    token = text[error.pos :]
    if error.msg == "Expecting value":
        return token in {"n", "nu", "nul", "t", "tr", "tru", "f", "fa", "fal", "fals", "-"}
    return (
        error.msg == "Expecting ',' delimiter"
        and token in {".", "e", "E", "e+", "e-", "E+", "E-"}
        and _has_number_completion(text[: error.pos], token)
    )


def _is_torn_object(tail: bytes, error: json.JSONDecodeError | UnicodeDecodeError) -> bool:
    if isinstance(error, UnicodeDecodeError):
        return _tail_text(tail, error).lstrip().startswith("{")
    text = error.doc
    return text.lstrip().startswith("{") and (
        error.pos == len(text)
        or error.msg == "Unterminated string starting at"
        or _is_torn_token(text, error)
    )


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
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if not _is_torn_object(tail, error):
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
