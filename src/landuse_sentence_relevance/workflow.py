from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import RLock

from landuse_sentence_relevance.domain.models import Annotation, Candidate, Label
from landuse_sentence_relevance.domain.profile import QuotaKey
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.domain.selection import select_final_annotations
from landuse_sentence_relevance.domain.v3_annotation import V3AnnotationSeed
from landuse_sentence_relevance.domain.v3_progress import (
    V3AnnotationProgress,
    V3SourceProgress,
    inspect_v3_session,
    next_v3_candidate,
)
from landuse_sentence_relevance.storage.publisher import DatasetPublicationError, DatasetPublisher
from landuse_sentence_relevance.storage.session import AnnotationStore

logger = logging.getLogger(__name__)


class UnknownCandidateError(ValueError):
    """Raised when a UI action does not target the current candidate."""


class UnknownAnnotationError(ValueError):
    """Raised when a UI action does not target a saved annotation."""


class WorkflowCompleteError(ValueError):
    """Raised when an annotation is submitted after the final upload."""


@dataclass(frozen=True, slots=True)
class WorkflowState:
    current_candidate: Candidate | None
    labeled_count: int
    yes_count: int
    no_count: int
    final_ready: bool
    published: bool
    annotations: tuple[Annotation, ...] = ()


@dataclass(frozen=True, slots=True)
class V3WorkflowState:
    """UI state for V3, where seeded V2 rows count but remain read-only."""

    current_candidate: Candidate | None
    progress: V3AnnotationProgress
    final_ready: bool
    published: bool = False
    annotations: tuple[Annotation, ...] = ()
    workflow_version: str = "v3"

    @property
    def total_count(self) -> int:
        return self.progress.total_count

    @property
    def seeded_count(self) -> int:
        return self.progress.seeded_count

    @property
    def fresh_labeled_count(self) -> int:
        return self.progress.fresh_labeled_count

    @property
    def labeled_count(self) -> int:
        return self.progress.labeled_count

    @property
    def yes_count(self) -> int:
        return self.progress.yes_count

    @property
    def no_count(self) -> int:
        return self.progress.no_count

    @property
    def target_yes_count(self) -> int:
        return self.progress.target_yes_count

    @property
    def target_no_count(self) -> int:
        return self.progress.target_no_count

    @property
    def remaining_count(self) -> int:
        return self.progress.remaining_count

    @property
    def source_progress(self) -> tuple[V3SourceProgress, ...]:
        """Return progress in the seed's deterministic source order."""
        return self.progress.source_progress

    @property
    def remaining_quotas(self) -> Mapping[QuotaKey, int]:
        return self.progress.remaining_quotas


class AnnotationWorkflow:
    def __init__(
        self,
        pool: FinalizedCandidatePool,
        store: AnnotationStore,
        publisher: DatasetPublisher,
        defer_publish: bool = False,
    ) -> None:
        self._pool = pool
        self._store = store
        self._publisher = publisher
        self._defer_publish = defer_publish
        self._lock = RLock()
        self._publish_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="landuse-publisher",
        )
        self._publish_future: Future[None] | None = None
        self._publish_requested = False
        self._closed = False
        self._annotations = store.load()
        self._published = False
        logger.info("Loaded %d saved annotations; resuming the annotation session", len(self._annotations))
        self._published = self.publish_if_ready()

    def current_candidate(self) -> Candidate | None:
        with self._lock:
            return self._current_candidate_unlocked()

    def annotate(self, candidate_id: str, label: Label) -> WorkflowState:
        with self._lock:
            if self._published:
                raise WorkflowCompleteError("the final dataset has already been uploaded")
            candidate = self._require_current_candidate(candidate_id)
            annotation = Annotation(candidate=candidate, label=Label(label))
            self._store.record(annotation)
            self._annotations[candidate_id] = annotation
            self._published = False
        if not self._defer_publish:
            self.publish_if_ready()
        state = self.state()
        logger.info(
            "Annotation saved: %d labeled (Yes=%d, No=%d)",
            state.labeled_count,
            state.yes_count,
            state.no_count,
        )
        return state

    def change_label(self, candidate_id: str, label: Label) -> WorkflowState:
        def edit(annotations: dict[str, Annotation]) -> None:
            annotation = self._saved_annotation(candidate_id)
            annotations[candidate_id] = replace(annotation, label=Label(label))

        return self._commit_review_edit(edit)

    def remove_annotation(self, candidate_id: str) -> WorkflowState:
        def edit(annotations: dict[str, Annotation]) -> None:
            self._saved_annotation(candidate_id)
            del annotations[candidate_id]

        return self._commit_review_edit(edit)

    def _commit_review_edit(self, edit: Callable[[dict[str, Annotation]], None]) -> WorkflowState:
        with self._lock:
            edit(self._annotations)
            self._store.save(self._annotations.values())
            self._published = False
        if not self._defer_publish:
            self.publish_if_ready()
        return self.state()

    def state(self) -> WorkflowState:
        with self._lock:
            annotations = tuple(self._annotations.values())
            selected = select_final_annotations(annotations)
            yes_count = sum(annotation.label is Label.YES for annotation in annotations)
            no_count = sum(annotation.label is Label.NO for annotation in annotations)
            return WorkflowState(
                current_candidate=self._current_candidate_unlocked(selected is not None),
                labeled_count=len(annotations),
                yes_count=yes_count,
                no_count=no_count,
                final_ready=selected is not None,
                published=self._published,
                annotations=annotations,
            )

    def publish_if_ready(self) -> bool:
        with self._lock:
            snapshot = tuple(self._annotations.values())
        try:
            published = self._publisher.publish_if_ready(snapshot)
        except DatasetPublicationError:
            logger.exception("Dataset upload failed; local annotations remain saved")
            return False
        with self._lock:
            if snapshot != tuple(self._annotations.values()):
                return False
            self._published = published or self._published
            return self._published

    def schedule_publish(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("workflow is closed")
            self._publish_requested = True
            if self._publish_future is None or self._publish_future.done():
                self._publish_future = self._publish_executor.submit(self._drain_publish_requests)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._publish_executor.shutdown(wait=True)

    def _drain_publish_requests(self) -> None:
        while True:
            with self._lock:
                if not self._publish_requested:
                    self._publish_future = None
                    return
                self._publish_requested = False
            self.publish_if_ready()

    def _require_current_candidate(self, candidate_id: str) -> Candidate:
        candidate = self._current_candidate_unlocked()
        if candidate is None or candidate.candidate_id != candidate_id:
            raise UnknownCandidateError("the candidate is unknown or is not the current candidate")
        return candidate

    def _current_candidate_unlocked(self, final_ready: bool | None = None) -> Candidate | None:
        if self._published or final_ready:
            return None
        if final_ready is None and select_final_annotations(self._annotations.values()) is not None:
            return None
        return self._pool.next_unannotated(set(self._annotations))

    def _saved_annotation(self, candidate_id: str) -> Annotation:
        annotation = self._annotations.get(candidate_id)
        if annotation is None:
            raise UnknownAnnotationError("the annotation is unknown")
        return annotation


class V3AnnotationWorkflow:
    """Resume fresh V3 labels over an immutable, seeded V2/V3 state."""

    def __init__(self, seed: V3AnnotationSeed, store: AnnotationStore) -> None:
        self._seed = seed
        self._store = store
        self._lock = RLock()
        self._closed = False
        self._annotations = store.load()
        self._seeded_ids = {row.candidate.candidate_id for row in seed.seeded_rows}
        inspect_v3_session(seed, self._annotations)
        logger.info(
            "Loaded %d fresh V3 annotations; %d frozen V2 rows count toward progress",
            len(self._annotations),
            seed.seeded_row_count,
        )

    def current_candidate(self) -> Candidate | None:
        with self._lock:
            return next_v3_candidate(self._seed, self._annotations)

    def annotate(self, candidate_id: str, label: Label) -> V3WorkflowState:
        with self._lock:
            self._require_open()
            candidate = self._require_current_candidate(candidate_id)
            updated = dict(self._annotations)
            updated[candidate_id] = Annotation(candidate=candidate, label=Label(label))
            self._save_and_replace(updated)
            state = self._state_unlocked()
        logger.info(
            "V3 annotation saved: %d/%d labeled (Yes=%d, No=%d)",
            state.labeled_count,
            state.total_count,
            state.yes_count,
            state.no_count,
        )
        return state

    def change_label(self, candidate_id: str, label: Label) -> V3WorkflowState:
        with self._lock:
            self._require_open()
            annotation = self._saved_annotation(candidate_id)
            updated = dict(self._annotations)
            updated[candidate_id] = replace(annotation, label=Label(label))
            self._save_and_replace(updated)
            return self._state_unlocked()

    def remove_annotation(self, candidate_id: str) -> V3WorkflowState:
        with self._lock:
            self._require_open()
            self._saved_annotation(candidate_id)
            updated = dict(self._annotations)
            del updated[candidate_id]
            self._save_and_replace(updated)
            return self._state_unlocked()

    def state(self) -> V3WorkflowState:
        with self._lock:
            return self._state_unlocked()

    def schedule_publish(self) -> None:
        """Keep the shared web contract while intentionally never publishing V3."""

        with self._lock:
            self._require_open()

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def _state_unlocked(self) -> V3WorkflowState:
        snapshot = inspect_v3_session(self._seed, self._annotations)
        return V3WorkflowState(
            current_candidate=snapshot.current_candidate,
            progress=snapshot.progress,
            final_ready=snapshot.current_candidate is None,
            annotations=snapshot.annotations,
        )

    def _save_and_replace(self, annotations: dict[str, Annotation]) -> None:
        self._store.save(tuple(annotations.values()))
        self._annotations = annotations

    def _require_current_candidate(self, candidate_id: str) -> Candidate:
        candidate = next_v3_candidate(self._seed, self._annotations)
        if candidate is None or candidate.candidate_id != candidate_id:
            raise UnknownCandidateError("the candidate is unknown or is not the current V3 candidate")
        return candidate

    def _saved_annotation(self, candidate_id: str) -> Annotation:
        if candidate_id in self._seeded_ids:
            raise UnknownAnnotationError("seeded V2 rows are immutable and cannot be edited")
        annotation = self._annotations.get(candidate_id)
        if annotation is None:
            raise UnknownAnnotationError("the annotation is unknown")
        return annotation

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("workflow is closed")
