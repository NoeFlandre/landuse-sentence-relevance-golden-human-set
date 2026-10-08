"""Recording workflow shared by the web request tests."""

from dataclasses import dataclass, field

from landuse_sentence_relevance.domain.models import Annotation, Candidate, Label
from landuse_sentence_relevance.workflow import (
    UnknownAnnotationError,
    UnknownCandidateError,
    WorkflowClosedError,
    WorkflowState,
)


@dataclass
class FakeWorkflow:
    candidate: Candidate
    calls: list[tuple[str, Label]]
    annotations: list[Annotation] = field(default_factory=list)
    schedule_calls: int = 0
    close_calls: int = 0
    closed: bool = False

    def current_candidate(self) -> Candidate:
        return self.candidate

    def state(self) -> WorkflowState:
        return WorkflowState(
            current_candidate=self.candidate,
            labeled_count=len(self.annotations),
            yes_count=sum(annotation.label is Label.YES for annotation in self.annotations),
            no_count=sum(annotation.label is Label.NO for annotation in self.annotations),
            final_ready=False,
            published=False,
            annotations=tuple(self.annotations),
        )

    def annotate(self, candidate_id: str, label: Label) -> WorkflowState:
        if candidate_id != self.candidate.candidate_id:
            raise UnknownCandidateError("unknown candidate")
        self.calls.append((candidate_id, label))
        self.annotations.append(Annotation(candidate=self.candidate, label=label))
        return self.state()

    def change_label(self, candidate_id: str, label: Label) -> WorkflowState:
        for index, annotation in enumerate(self.annotations):
            if annotation.candidate.candidate_id == candidate_id:
                self.annotations[index] = Annotation(annotation.candidate, label)
                return self.state()
        raise UnknownAnnotationError("the annotation is unknown")

    def remove_annotation(self, candidate_id: str) -> WorkflowState:
        remaining = [
            annotation for annotation in self.annotations if annotation.candidate.candidate_id != candidate_id
        ]
        if len(remaining) == len(self.annotations):
            raise UnknownAnnotationError("the annotation is unknown")
        self.annotations = remaining
        return self.state()

    def schedule_publish(self) -> None:
        if self.closed:
            raise WorkflowClosedError("workflow is closed")
        self.schedule_calls += 1

    def close(self) -> None:
        self.close_calls += 1
