"""Pure progress accounting for the resumable V3 annotation session."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from landuse_sentence_relevance.domain.models import Annotation, Candidate, Label, Source
from landuse_sentence_relevance.domain.profile import QuotaKey
from landuse_sentence_relevance.domain.v3_annotation import V3AnnotationSeed


class V3AnnotationProgressError(ValueError):
    """Raised when a persisted V3 session is not compatible with its seed."""


@dataclass(frozen=True, slots=True)
class V3SourceProgress:
    """Progress for one V3 source, including its target and remaining quotas."""

    source: Source
    total_count: int
    labeled_count: int
    yes_count: int
    no_count: int
    target_yes_count: int
    target_no_count: int
    remaining_yes_count: int
    remaining_no_count: int

    @property
    def target_count(self) -> int:
        return self.target_yes_count + self.target_no_count

    @property
    def remaining_count(self) -> int:
        return max(0, self.total_count - self.labeled_count)


@dataclass(frozen=True, slots=True)
class V3AnnotationProgress:
    """Immutable totals and source breakdown for one V3 session snapshot."""

    total_count: int
    seeded_count: int
    fresh_labeled_count: int
    labeled_count: int
    yes_count: int
    no_count: int
    target_yes_count: int
    target_no_count: int
    source_progress: tuple[V3SourceProgress, ...]
    remaining_quotas: Mapping[QuotaKey, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "remaining_quotas", MappingProxyType(dict(self.remaining_quotas)))

    @property
    def target_count(self) -> int:
        return self.target_yes_count + self.target_no_count

    @property
    def remaining_count(self) -> int:
        return max(0, self.total_count - self.labeled_count)


@dataclass(frozen=True, slots=True)
class V3AnnotationSnapshot:
    """One validated view of a resumable V3 session."""

    annotations: tuple[Annotation, ...]
    progress: V3AnnotationProgress
    current_candidate: Candidate | None


def ordered_v3_annotations(
    seed: V3AnnotationSeed,
    annotations: Mapping[str, Annotation],
) -> tuple[Annotation, ...]:
    """Validate and return only fresh labels in deterministic seed order."""

    return _validated_annotations(seed, annotations)


def inspect_v3_session(
    seed: V3AnnotationSeed,
    annotations: Mapping[str, Annotation],
) -> V3AnnotationSnapshot:
    """Validate a session once and derive all UI-facing state from that view."""

    fresh = _validated_annotations(seed, annotations)
    all_annotations = (*seed.seeded_annotations, *fresh)
    counts_by_source_label = _label_counts_by_source(all_annotations)
    progress = _build_progress(seed, fresh, all_annotations, counts_by_source_label)
    labeled_ids = {annotation.candidate.candidate_id for annotation in fresh}
    pending = _next_pending_candidate(
        seed,
        labeled_ids,
        _short_sources(progress.remaining_quotas),
    )
    current_candidate = (
        pending if pending is not None else _next_candidate(seed, labeled_ids, progress.remaining_quotas)
    )
    return V3AnnotationSnapshot(
        annotations=fresh,
        progress=progress,
        current_candidate=current_candidate,
    )


def next_v3_candidate(
    seed: V3AnnotationSeed,
    annotations: Mapping[str, Annotation],
) -> Candidate | None:
    """Return the next candidate worth a human's time, never a seeded V2 row.

    Seed rows come first, then the reserve. A reserve candidate is only offered
    while its source still has an unfilled quota, because a fresh row records the
    label its slot hoped for rather than the one a human gives: a "yes" slot
    answered "no" leaves that quota open and needs another candidate.
    """

    fresh = _validated_annotations(seed, annotations)
    labeled_ids = {annotation.candidate.candidate_id for annotation in fresh}
    all_annotations = (*seed.seeded_annotations, *fresh)
    counts_by_source_label = _label_counts_by_source(all_annotations)
    remaining_quotas = _remaining_quotas(seed, counts_by_source_label)
    pending = _next_pending_candidate(seed, labeled_ids, _short_sources(remaining_quotas))
    if pending is not None:
        return pending
    return _next_candidate(seed, labeled_ids, remaining_quotas)


def _next_candidate(
    seed: V3AnnotationSeed,
    labeled_ids: set[str],
    remaining_quotas: Mapping[QuotaKey, int],
) -> Candidate | None:
    short = _short_sources(remaining_quotas)
    return _next_reserve_candidate(seed, labeled_ids, short)


def _short_sources(remaining_quotas: Mapping[QuotaKey, int]) -> set[Source]:
    return {source for (source, _label), missing in remaining_quotas.items() if missing > 0}


def _next_reserve_candidate(
    seed: V3AnnotationSeed,
    labeled_ids: set[str],
    short_sources: set[Source],
) -> Candidate | None:
    if not short_sources:
        return None
    return next(
        (
            candidate
            for candidate in seed.reserve_candidates
            if candidate.candidate_id not in labeled_ids and candidate.source in short_sources
        ),
        None,
    )


def _next_pending_candidate(
    seed: V3AnnotationSeed,
    labeled_ids: set[str],
    short_sources: set[Source],
) -> Candidate | None:
    return next(
        (
            row.candidate
            for row in seed.pending_rows
            if row.candidate.candidate_id not in labeled_ids and row.candidate.source in short_sources
        ),
        None,
    )


def summarize_v3_progress(
    seed: V3AnnotationSeed,
    annotations: Mapping[str, Annotation],
) -> V3AnnotationProgress:
    """Count frozen V2 and fresh labels without reading or writing any files."""

    return inspect_v3_session(seed, annotations).progress


def _build_progress(
    seed: V3AnnotationSeed,
    fresh: tuple[Annotation, ...],
    all_annotations: tuple[Annotation, ...],
    counts_by_source_label: Counter[tuple[Source, Label]],
) -> V3AnnotationProgress:
    source_progress = _source_progress_by_source(seed, counts_by_source_label)
    return V3AnnotationProgress(
        total_count=seed.total_rows,
        seeded_count=seed.seeded_row_count,
        fresh_labeled_count=len(fresh),
        labeled_count=len(all_annotations),
        yes_count=_label_count(all_annotations, Label.YES),
        no_count=_label_count(all_annotations, Label.NO),
        target_yes_count=_target_count(seed, Label.YES),
        target_no_count=_target_count(seed, Label.NO),
        source_progress=source_progress,
        remaining_quotas=_remaining_quotas(seed, counts_by_source_label),
    )


def _session_candidates(
    seed: V3AnnotationSeed,
) -> tuple[Mapping[str, Candidate], frozenset[str]]:
    return seed.offerable_by_id, seed.seeded_candidate_ids


def _validated_annotations(
    seed: V3AnnotationSeed,
    annotations: Mapping[str, Annotation],
) -> tuple[Annotation, ...]:
    pending, seeded_ids = _session_candidates(seed)
    _validate_session(annotations, pending, seeded_ids)
    return _ordered_pending_annotations(seed, annotations)


def _validate_session(
    annotations: Mapping[str, Annotation],
    pending: Mapping[str, Candidate],
    seeded_ids: Collection[str],
) -> None:
    for candidate_id, annotation in annotations.items():
        _validate_session_row(candidate_id, annotation, pending, seeded_ids)


def _ordered_pending_annotations(
    seed: V3AnnotationSeed,
    annotations: Mapping[str, Annotation],
) -> tuple[Annotation, ...]:
    return tuple(
        annotations[candidate.candidate_id]
        for candidate in seed.offerable_candidates
        if candidate.candidate_id in annotations
    )


def _label_counts_by_source(
    annotations: tuple[Annotation, ...],
) -> Counter[tuple[Source, Label]]:
    return Counter((annotation.candidate.source, annotation.label) for annotation in annotations)


def _source_progress_by_source(
    seed: V3AnnotationSeed,
    counts: Counter[tuple[Source, Label]],
) -> tuple[V3SourceProgress, ...]:
    return tuple(_source_progress(seed, source, counts) for source in seed.quotas.sources)


def _remaining_quotas(
    seed: V3AnnotationSeed,
    counts: Counter[tuple[Source, Label]],
) -> dict[QuotaKey, int]:
    return {
        (source, label): max(0, seed.quotas.required(source, label) - counts[(source, label)])
        for source in seed.quotas.sources
        for label in seed.quotas.labels
    }


def _label_count(annotations: tuple[Annotation, ...], label: Label) -> int:
    return sum(annotation.label is label for annotation in annotations)


def _target_count(seed: V3AnnotationSeed, label: Label) -> int:
    return sum(seed.quotas.required(source, label) for source in seed.quotas.sources)


def _source_progress(
    seed: V3AnnotationSeed,
    source: Source,
    counts: Counter[tuple[Source, Label]],
) -> V3SourceProgress:
    total_count = seed.rows_by_source[source]
    labeled_count = sum(counts[(source, label)] for label in (Label.YES, Label.NO))
    target_yes_count = seed.quotas.required(source, Label.YES)
    target_no_count = seed.quotas.required(source, Label.NO)
    return V3SourceProgress(
        source=source,
        total_count=total_count,
        labeled_count=labeled_count,
        yes_count=counts[(source, Label.YES)],
        no_count=counts[(source, Label.NO)],
        target_yes_count=target_yes_count,
        target_no_count=target_no_count,
        remaining_yes_count=max(0, target_yes_count - counts[(source, Label.YES)]),
        remaining_no_count=max(0, target_no_count - counts[(source, Label.NO)]),
    )


def _validate_session_row(
    candidate_id: str,
    annotation: Annotation,
    pending: Mapping[str, Candidate],
    seeded_ids: Collection[str],
) -> None:
    if candidate_id != annotation.candidate.candidate_id:
        raise V3AnnotationProgressError("V3 session key does not match its candidate ID")
    if candidate_id in seeded_ids:
        raise V3AnnotationProgressError("seeded V2 rows are immutable and cannot be reannotated")
    expected = pending.get(candidate_id)
    if expected is None:
        raise V3AnnotationProgressError("V3 session contains an unknown candidate")
    if annotation.candidate != expected:
        raise V3AnnotationProgressError("V3 session candidate content does not match the seed")
