from __future__ import annotations

from dataclasses import dataclass

from landuse_sentence_relevance.domain.models import Annotation, Candidate, Label
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.domain.selection import select_final_annotations
from landuse_sentence_relevance.storage.publisher import DatasetPublisher
from landuse_sentence_relevance.storage.session import AnnotationStore


class UnknownCandidateError(ValueError):
    """Raised when a UI action does not target the current candidate."""


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


class AnnotationWorkflow:
    def __init__(
        self,
        pool: FinalizedCandidatePool,
        store: AnnotationStore,
        publisher: DatasetPublisher,
    ) -> None:
        self._pool = pool
        self._store = store
        self._publisher = publisher
        self._annotations = store.load()
        self._published = self._publisher.publish_if_ready(self._annotations.values())

    def current_candidate(self) -> Candidate | None:
        if self._published:
            return None
        return self._pool.next_unannotated(set(self._annotations))

    def annotate(self, candidate_id: str, label: Label) -> WorkflowState:
        if self._published:
            raise WorkflowCompleteError("the final dataset has already been uploaded")
        candidate = self.current_candidate()
        if candidate is None or candidate.candidate_id != candidate_id:
            raise UnknownCandidateError("the candidate is unknown or is not the current candidate")
        annotation = Annotation(candidate=candidate, label=Label(label))
        self._store.record(annotation)
        self._annotations[candidate_id] = annotation
        self._published = self._publisher.publish_if_ready(self._annotations.values()) or self._published
        return self.state()

    def state(self) -> WorkflowState:
        annotations = tuple(self._annotations.values())
        selected = select_final_annotations(annotations)
        yes_count = sum(annotation.label is Label.YES for annotation in annotations)
        no_count = sum(annotation.label is Label.NO for annotation in annotations)
        return WorkflowState(
            current_candidate=self.current_candidate(),
            labeled_count=len(annotations),
            yes_count=yes_count,
            no_count=no_count,
            final_ready=selected is not None,
            published=self._published,
        )
