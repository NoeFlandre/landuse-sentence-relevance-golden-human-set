from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from heapq import nsmallest
from types import MappingProxyType
from typing import Any, Literal, cast

from landuse_sentence_relevance.domain.models import Annotation, Candidate, Label, Source
from landuse_sentence_relevance.domain.profile import V3_QUOTAS, QuotaKey, SourceLabelQuotas
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.domain.seeding import SeedPlan

V3SeedOrigin = Literal["v2", "v3"]


class V3AnnotationSeedError(ValueError):
    """Raised when a V3 annotation seed cannot satisfy its immutable contracts."""


@dataclass(frozen=True, slots=True)
class V3SelectionMetadata:
    """Deterministic provenance for one selected V3 seed row."""

    seed: str
    rank: str
    slot_index: int

    def __post_init__(self) -> None:
        _validate_selection_seed(self.seed)
        _validate_selection_rank(self.rank)
        _validate_selection_slot(self.slot_index)

    def to_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "seed": self.seed, "slot_index": self.slot_index}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V3SelectionMetadata:
        return cls(
            seed=str(payload["seed"]), rank=str(payload["rank"]), slot_index=int(payload["slot_index"])
        )


@dataclass(frozen=True, slots=True)
class V3SeedRow:
    """One final V3 row, either a labeled frozen V2 row or an unlabeled V3 slot."""

    candidate: Candidate
    quota_source: Source
    quota_label: Label
    origin: V3SeedOrigin
    annotation: Annotation | None
    selection: V3SelectionMetadata

    def __post_init__(self) -> None:
        _validate_seed_row_source(self)
        if self.origin not in ("v2", "v3"):
            raise V3AnnotationSeedError(f"unknown V3 seed origin: {self.origin!r}")
        if self.origin == "v2":
            _validate_frozen_seed_row(self)
            return
        _validate_fresh_seed_row(self)

    @property
    def quota_key(self) -> QuotaKey:
        return self.quota_source, self.quota_label

    @property
    def is_seeded(self) -> bool:
        return self.origin == "v2"

    @property
    def is_pending(self) -> bool:
        return self.origin == "v3"

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation": self.annotation.to_dict() if self.annotation is not None else None,
            "candidate": self.candidate.to_dict(),
            "origin": self.origin,
            "quota_slot": {"label": self.quota_label.value, "source": self.quota_source.value},
            "selection": self.selection.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V3SeedRow:
        annotation_payload = payload.get("annotation")
        quota_slot = payload["quota_slot"]
        if not isinstance(quota_slot, Mapping):
            raise ValueError("V3 seed row quota_slot must be an object")
        return cls(
            candidate=Candidate.from_dict(dict(payload["candidate"])),
            quota_source=Source(str(quota_slot["source"])),
            quota_label=Label(str(quota_slot["label"])),
            origin=cast(V3SeedOrigin, str(payload["origin"])),
            annotation=(
                Annotation.from_dict(dict(annotation_payload))
                if isinstance(annotation_payload, Mapping)
                else None
            ),
            selection=V3SelectionMetadata.from_dict(dict(payload["selection"])),
        )


@dataclass(frozen=True, slots=True)
class V3AnnotationSeed:
    """The deterministic 300-row V3 state before any fresh human annotation."""

    rows: tuple[V3SeedRow, ...]
    excluded_v2_rows: tuple[Annotation, ...]
    reserved_v2_cells: frozenset[str]
    quotas: SourceLabelQuotas
    benchmark_sha256: str
    seed: str
    excluded_v2_reasons: Mapping[str, str] = field(default_factory=dict)
    reserve_candidates: tuple[Candidate, ...] = ()
    """Spare pool candidates offered when a fresh label leaves its quota short.

    A fresh row records the label a slot hoped for, not the one a human gives.
    When those disagree the quota stays open, so the oversized pool keeps a
    deterministic remainder here to offer next. They are not part of the 300-row
    contract and carry no quota slot until a label decides which one they fill.
    """

    _seeded_rows: tuple[V3SeedRow, ...] = field(init=False, repr=False, compare=False)
    _pending_rows: tuple[V3SeedRow, ...] = field(init=False, repr=False, compare=False)
    _offerable_candidates: tuple[Candidate, ...] = field(init=False, repr=False, compare=False)
    _offerable_by_id: Mapping[str, Candidate] = field(init=False, repr=False, compare=False)
    _seeded_annotations: tuple[Annotation, ...] = field(init=False, repr=False, compare=False)
    _seeded_candidate_ids: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_seed_identity(self.seed, self.benchmark_sha256)
        object.__setattr__(self, "excluded_v2_reasons", MappingProxyType(dict(self.excluded_v2_reasons)))
        _cache_seed_indexes(self)
        _validate_state(self)

    @property
    def total_rows(self) -> int:
        return len(self.rows)

    @property
    def seeded_rows(self) -> tuple[V3SeedRow, ...]:
        return self._seeded_rows

    @property
    def pending_rows(self) -> tuple[V3SeedRow, ...]:
        return self._pending_rows

    @property
    def pending_candidates(self) -> tuple[Candidate, ...]:
        return tuple(row.candidate for row in self.pending_rows)

    @property
    def offerable_candidates(self) -> tuple[Candidate, ...]:
        """Every candidate a human may still be shown, seed rows before reserve."""

        return self._offerable_candidates

    @property
    def offerable_by_id(self) -> Mapping[str, Candidate]:
        """Immutable lookup of every candidate that may be annotated."""

        return self._offerable_by_id

    @property
    def seeded_candidate_ids(self) -> frozenset[str]:
        """Candidate IDs belonging to immutable V2 rows."""

        return self._seeded_candidate_ids

    @property
    def seeded_annotations(self) -> tuple[Annotation, ...]:
        return self._seeded_annotations

    @property
    def seeded_row_count(self) -> int:
        return len(self.seeded_rows)

    @property
    def pending_row_count(self) -> int:
        return len(self.pending_rows)

    @property
    def rows_by_source(self) -> dict[Source, int]:
        counts = Counter(row.candidate.source for row in self.rows)
        return {source: counts[source] for source in self.quotas.sources}

    @property
    def quota_counts(self) -> dict[QuotaKey, int]:
        counts = Counter(row.quota_key for row in self.rows)
        return {key: counts[key] for key in _quota_keys(self.quotas)}

    @property
    def pending_counts(self) -> dict[QuotaKey, int]:
        counts = Counter(row.quota_key for row in self.pending_rows)
        return {key: counts[key] for key in _quota_keys(self.quotas) if counts[key]}

    @property
    def seeded_label_counts(self) -> dict[Label, int]:
        counts = Counter(annotation.label for annotation in self.seeded_annotations)
        return {label: counts[label] for label in self.quotas.labels}

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_sha256": self.benchmark_sha256,
            "excluded_v2_rows": [annotation.to_dict() for annotation in self.excluded_v2_rows],
            "excluded_v2_reasons": dict(self.excluded_v2_reasons),
            "reserve_candidates": [candidate.to_dict() for candidate in self.reserve_candidates],
            "quotas": _quotas_to_dict(self.quotas),
            "reserved_v2_cells": sorted(self.reserved_v2_cells),
            "rows": [row.to_dict() for row in self.rows],
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V3AnnotationSeed:
        return cls(
            quotas=_quotas_from_dict(payload),
            rows=_seed_rows_from_dict(payload),
            excluded_v2_rows=_excluded_rows_from_dict(payload),
            reserved_v2_cells=_reserved_cells_from_dict(payload),
            benchmark_sha256=str(payload["benchmark_sha256"]),
            seed=str(payload["seed"]),
            excluded_v2_reasons=dict(payload.get("excluded_v2_reasons", {})),
            reserve_candidates=tuple(
                Candidate.from_dict(dict(row)) for row in payload.get("reserve_candidates", ())
            ),
        )


def _cache_seed_indexes(state: V3AnnotationSeed) -> None:
    seeded_rows = _seeded_rows_for(state.rows)
    pending_rows = _pending_rows_for(state.rows)
    offerable_candidates = _offerable_candidates_for(pending_rows, state.reserve_candidates)
    object.__setattr__(state, "_seeded_rows", seeded_rows)
    object.__setattr__(state, "_pending_rows", pending_rows)
    object.__setattr__(state, "_offerable_candidates", offerable_candidates)
    object.__setattr__(state, "_offerable_by_id", _candidate_index(offerable_candidates))
    object.__setattr__(state, "_seeded_annotations", _seeded_annotations_for(seeded_rows))
    object.__setattr__(state, "_seeded_candidate_ids", _seeded_ids_for(seeded_rows))


def _seeded_rows_for(rows: tuple[V3SeedRow, ...]) -> tuple[V3SeedRow, ...]:
    return tuple(row for row in rows if row.is_seeded)


def _pending_rows_for(rows: tuple[V3SeedRow, ...]) -> tuple[V3SeedRow, ...]:
    return tuple(row for row in rows if row.is_pending)


def _offerable_candidates_for(
    pending_rows: tuple[V3SeedRow, ...], reserve_candidates: tuple[Candidate, ...]
) -> tuple[Candidate, ...]:
    return tuple(row.candidate for row in pending_rows) + reserve_candidates


def _candidate_index(candidates: tuple[Candidate, ...]) -> Mapping[str, Candidate]:
    return MappingProxyType({candidate.candidate_id: candidate for candidate in candidates})


def _seeded_annotations_for(rows: tuple[V3SeedRow, ...]) -> tuple[Annotation, ...]:
    return tuple(row.annotation for row in rows if row.annotation is not None)


def _seeded_ids_for(rows: tuple[V3SeedRow, ...]) -> frozenset[str]:
    return frozenset(row.candidate.candidate_id for row in rows)


def select_v3_annotation_seed(
    seed_plan: SeedPlan,
    pool: FinalizedCandidatePool,
    *,
    quotas: SourceLabelQuotas = V3_QUOTAS,
    seed: str,
    benchmark_sha256: str,
) -> V3AnnotationSeed:
    """Fill every quota slot from frozen V2 rows and fresh pool candidates."""

    _validate_seed_plan(seed_plan, quotas)
    _validate_pool(pool, seed_plan, quotas)
    seeded_rows = _seeded_rows(seed_plan, quotas, seed)
    pending_rows = _pending_rows(seed_plan, pool, quotas, seed)
    return V3AnnotationSeed(
        rows=(*seeded_rows, *pending_rows),
        reserve_candidates=_reserve_candidates(pool, (*seeded_rows, *pending_rows), seed),
        excluded_v2_rows=seed_plan.excluded,
        reserved_v2_cells=seed_plan.reserved_cells,
        quotas=quotas,
        benchmark_sha256=benchmark_sha256,
        seed=seed,
        excluded_v2_reasons=_excluded_v2_reasons(seed_plan),
    )


def _reserve_candidates(
    pool: FinalizedCandidatePool,
    rows: tuple[V3SeedRow, ...],
    seed: str,
) -> tuple[Candidate, ...]:
    """Hold back every pool candidate the 300 rows did not use, in hash-rank order.

    These are what a slot falls back to when a human's label leaves its quota
    short, and excluding the cells the rows already occupy keeps the set globally
    H3-unique. Ranking by seeded hash rather than pool order does two things: the
    reserve no longer changes when the pool is permuted, and an annotator is not
    walked through one region at a time, since a candidate id begins with its
    region.
    """

    used_ids = {row.candidate.candidate_id for row in rows}
    used_cells = {row.candidate.h3_cell for row in rows}
    spare = _unused_candidates(pool.candidates, used_ids, used_cells)
    return _ranked_candidates(spare, f"{seed}:reserve")


def _unused_candidates(
    candidates: tuple[Candidate, ...],
    used_ids: set[str],
    used_cells: set[str],
) -> tuple[Candidate, ...]:
    return tuple(
        candidate
        for candidate in candidates
        if candidate.candidate_id not in used_ids and candidate.h3_cell not in used_cells
    )


def _ranked_candidates(candidates: tuple[Candidate, ...], seed: str) -> tuple[Candidate, ...]:
    def key(candidate: Candidate) -> tuple[str, str]:
        return _rank(seed, candidate.candidate_id), candidate.candidate_id

    return tuple(sorted(candidates, key=key))


def _seeded_rows(
    seed_plan: SeedPlan,
    quotas: SourceLabelQuotas,
    seed: str,
) -> tuple[V3SeedRow, ...]:
    rows: list[V3SeedRow] = []
    for source, label in _quota_keys(quotas):
        rows.extend(_seeded_quota_rows(seed_plan, source, label, seed))
    return tuple(rows)


def _seeded_quota_rows(
    seed_plan: SeedPlan,
    source: Source,
    label: Label,
    seed: str,
) -> tuple[V3SeedRow, ...]:
    annotations = _reused_annotations(seed_plan, source, label)
    return tuple(
        V3SeedRow(
            candidate=annotation.candidate,
            quota_source=source,
            quota_label=label,
            origin="v2",
            annotation=annotation,
            selection=V3SelectionMetadata(
                seed=seed,
                rank=_rank(f"{seed}:v2", annotation.candidate.candidate_id),
                slot_index=index,
            ),
        )
        for index, annotation in enumerate(annotations)
    )


def _reused_annotations(seed_plan: SeedPlan, source: Source, label: Label) -> tuple[Annotation, ...]:
    return tuple(
        sorted(
            (
                annotation
                for annotation in seed_plan.reused
                if annotation.candidate.source is source and annotation.label is label
            ),
            key=lambda item: item.candidate.candidate_id,
        )
    )


def _pending_rows(
    seed_plan: SeedPlan,
    pool: FinalizedCandidatePool,
    quotas: SourceLabelQuotas,
    seed: str,
) -> tuple[V3SeedRow, ...]:
    used_ids = {annotation.candidate.candidate_id for annotation in (*seed_plan.reused, *seed_plan.excluded)}
    rows: list[V3SeedRow] = []
    for source, label in _quota_keys(quotas):
        rows.extend(_pending_quota_rows(seed_plan, pool, source, label, seed, used_ids))
    return tuple(rows)


def _pending_quota_rows(
    seed_plan: SeedPlan,
    pool: FinalizedCandidatePool,
    source: Source,
    label: Label,
    seed: str,
    used_ids: set[str],
) -> tuple[V3SeedRow, ...]:
    required = seed_plan.remaining.get((source, label), 0)
    ranked = _ranked_fresh_candidates(pool, source, label, seed, used_ids, limit=required)
    if len(ranked) < required:
        raise V3AnnotationSeedError(
            f"V3 {source.value}/{label.value} quota needs {required} fresh rows; "
            f"only {len(ranked)} remain globally unique"
        )
    rows: list[V3SeedRow] = []
    for index, candidate in enumerate(ranked[:required]):
        rank = _rank(f"{seed}:{source.value}:{label.value}", candidate.candidate_id)
        rows.append(
            V3SeedRow(
                candidate=candidate,
                quota_source=source,
                quota_label=label,
                origin="v3",
                annotation=None,
                selection=V3SelectionMetadata(seed=seed, rank=rank, slot_index=index),
            )
        )
        used_ids.add(candidate.candidate_id)
    return tuple(rows)


def _ranked_fresh_candidates(
    pool: FinalizedCandidatePool,
    source: Source,
    label: Label,
    seed: str,
    used_ids: set[str],
    *,
    limit: int,
) -> tuple[Candidate, ...]:
    candidates = (
        candidate for candidate in pool.candidates if _is_fresh_for_quota(candidate, source, used_ids)
    )
    return tuple(
        nsmallest(
            limit,
            candidates,
            key=lambda candidate: (
                _rank(f"{seed}:{source.value}:{label.value}", candidate.candidate_id),
                candidate.candidate_id,
            ),
        )
    )


def _is_fresh_for_quota(
    candidate: Candidate,
    source: Source,
    used_ids: set[str],
) -> bool:
    return candidate.source is source and candidate.candidate_id not in used_ids


def _validate_seed_plan(seed_plan: SeedPlan, quotas: SourceLabelQuotas) -> None:
    existing = (*seed_plan.reused, *seed_plan.excluded)
    quota_keys = _quota_keys(quotas)
    _require_unique_v2_rows(existing)
    _require_reserved_v2_cells(existing, seed_plan.reserved_cells)
    kept_counts = Counter((annotation.candidate.source, annotation.label) for annotation in seed_plan.reused)
    _require_known_quota_keys(kept_counts, quota_keys)
    _require_quota_limits(kept_counts, quotas)
    _require_matching_remaining(seed_plan.remaining, _expected_remaining(kept_counts, quotas, quota_keys))


def _require_unique_v2_rows(existing: tuple[Annotation, ...]) -> None:
    ids = [annotation.candidate.candidate_id for annotation in existing]
    cells = [annotation.candidate.h3_cell for annotation in existing]
    if len(set(ids)) != len(ids) or len(set(cells)) != len(cells):
        raise V3AnnotationSeedError("V2 seed rows must have unique candidate IDs and H3 cells")


def _require_reserved_v2_cells(
    existing: tuple[Annotation, ...],
    reserved_cells: frozenset[str],
) -> None:
    cells = {annotation.candidate.h3_cell for annotation in existing}
    if cells != set(reserved_cells):
        raise V3AnnotationSeedError("V2 reserved H3 cells must match every V2 seed row")


def _require_known_quota_keys(
    kept_counts: Mapping[QuotaKey, int],
    quota_keys: tuple[QuotaKey, ...],
) -> None:
    if any(key not in quota_keys for key in kept_counts):
        raise V3AnnotationSeedError("V2 seed contains a source/label pair outside the V3 quota matrix")


def _require_quota_limits(
    kept_counts: Mapping[QuotaKey, int],
    quotas: SourceLabelQuotas,
) -> None:
    if any(kept_counts[key] > quotas.required(*key) for key in kept_counts):
        raise V3AnnotationSeedError("V2 reused rows exceed a V3 quota")


def _expected_remaining(
    kept_counts: Mapping[QuotaKey, int],
    quotas: SourceLabelQuotas,
    quota_keys: tuple[QuotaKey, ...],
) -> dict[QuotaKey, int]:
    return {
        key: quotas.required(*key) - kept_counts[key]
        for key in quota_keys
        if quotas.required(*key) > kept_counts[key]
    }


def _require_matching_remaining(
    actual: Mapping[QuotaKey, int],
    expected: Mapping[QuotaKey, int],
) -> None:
    if dict(actual) != dict(expected):
        raise V3AnnotationSeedError("V2 seed shortfall does not match the quota matrix")


def _validate_pool(
    pool: FinalizedCandidatePool,
    seed_plan: SeedPlan,
    quotas: SourceLabelQuotas,
) -> None:
    _require_pool_sources(pool, quotas)
    _require_pool_cells(pool, seed_plan.reserved_cells)
    _require_pool_candidate_ids(pool, seed_plan)


def _require_pool_sources(pool: FinalizedCandidatePool, quotas: SourceLabelQuotas) -> None:
    expected_sources = set(quotas.sources)
    unexpected = {candidate.source for candidate in pool.candidates} - expected_sources
    if unexpected:
        values = sorted(source.value for source in unexpected)
        raise V3AnnotationSeedError(f"candidate pool contains sources outside the V3 quota matrix: {values}")


def _require_pool_cells(pool: FinalizedCandidatePool, reserved_cells: frozenset[str]) -> None:
    collisions = sorted(set(pool.cells) & set(reserved_cells))
    if collisions:
        raise V3AnnotationSeedError(f"candidate pool collides with V2-reserved H3 cells: {collisions[:5]}")


def _require_pool_candidate_ids(pool: FinalizedCandidatePool, seed_plan: SeedPlan) -> None:
    v2_ids = {annotation.candidate.candidate_id for annotation in (*seed_plan.reused, *seed_plan.excluded)}
    if v2_ids & {candidate.candidate_id for candidate in pool.candidates}:
        raise V3AnnotationSeedError("candidate pool reuses a V2 candidate ID")


def _validate_state(state: V3AnnotationSeed) -> None:
    _require_state_row_count(state)
    _require_unique_state_ids(state)
    _require_unique_state_cells(state)
    _require_state_quota_counts(state)
    _require_state_reserved_cells(state)
    _require_excluded_reasons(state)
    _require_pending_cells(state)
    _require_reserve_candidates(state)


def _require_state_row_count(state: V3AnnotationSeed) -> None:
    if state.total_rows != state.quotas.total:
        raise V3AnnotationSeedError(
            f"V3 seed must contain {state.quotas.total} rows; received {state.total_rows}"
        )


def _require_unique_state_ids(state: V3AnnotationSeed) -> None:
    ids = [row.candidate.candidate_id for row in state.rows]
    if len(set(ids)) != len(ids):
        raise V3AnnotationSeedError("V3 seed rows must have unique candidate IDs")


def _require_unique_state_cells(state: V3AnnotationSeed) -> None:
    cells = [row.candidate.h3_cell for row in state.rows]
    if len(set(cells)) != len(cells):
        raise V3AnnotationSeedError("V3 seed rows must have globally unique H3 cells")


def _require_state_quota_counts(state: V3AnnotationSeed) -> None:
    expected = {key: state.quotas.required(*key) for key in _quota_keys(state.quotas)}
    if state.quota_counts != expected:
        raise V3AnnotationSeedError("V3 source-by-label quota slots are not satisfied")


def _require_state_reserved_cells(state: V3AnnotationSeed) -> None:
    if set(state.reserved_v2_cells) != _v2_cells(state):
        raise V3AnnotationSeedError("V3 state does not preserve every V2 H3 cell")


def _require_excluded_reasons(state: V3AnnotationSeed) -> None:
    excluded_ids = {annotation.candidate.candidate_id for annotation in state.excluded_v2_rows}
    if set(state.excluded_v2_reasons) != excluded_ids or any(
        not reason.strip() for reason in state.excluded_v2_reasons.values()
    ):
        raise V3AnnotationSeedError("every excluded V2 row must have a non-empty reason")


def _require_pending_cells(state: V3AnnotationSeed) -> None:
    pending_cells = {row.candidate.h3_cell for row in state.pending_rows}
    if pending_cells & state.reserved_v2_cells:
        raise V3AnnotationSeedError("pending V3 rows collide with a V2-reserved H3 cell")


def _require_reserve_candidates(state: V3AnnotationSeed) -> None:
    _require_unique_reserve_ids(state.reserve_candidates)
    _require_unique_reserve_cells(state.reserve_candidates)
    _require_reserve_not_in_rows(state)
    _require_reserve_not_in_v2_cells(state)
    _require_reserve_sources(state)


def _require_unique_reserve_ids(candidates: tuple[Candidate, ...]) -> None:
    ids = [candidate.candidate_id for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise V3AnnotationSeedError("V3 reserve candidates must have unique candidate IDs")


def _require_unique_reserve_cells(candidates: tuple[Candidate, ...]) -> None:
    cells = [candidate.h3_cell for candidate in candidates]
    if len(set(cells)) != len(cells):
        raise V3AnnotationSeedError("V3 reserve candidates must have unique H3 cells")


def _require_reserve_not_in_rows(state: V3AnnotationSeed) -> None:
    _require_reserve_ids_not_in_rows(state)
    _require_reserve_cells_not_in_rows(state)


def _require_reserve_ids_not_in_rows(state: V3AnnotationSeed) -> None:
    row_ids = {row.candidate.candidate_id for row in state.rows}
    reserve_ids = {candidate.candidate_id for candidate in state.reserve_candidates}
    if reserve_ids & row_ids:
        raise V3AnnotationSeedError("V3 reserve candidates must not reuse a seed candidate ID")


def _require_reserve_cells_not_in_rows(state: V3AnnotationSeed) -> None:
    row_cells = {row.candidate.h3_cell for row in state.rows}
    reserve_cells = {candidate.h3_cell for candidate in state.reserve_candidates}
    if reserve_cells & row_cells:
        raise V3AnnotationSeedError("V3 reserve candidates must not reuse a seed H3 cell")


def _require_reserve_not_in_v2_cells(state: V3AnnotationSeed) -> None:
    reserve_cells = {candidate.h3_cell for candidate in state.reserve_candidates}
    if reserve_cells & state.reserved_v2_cells:
        raise V3AnnotationSeedError("V3 reserve candidates must not use a V2-reserved H3 cell")


def _require_reserve_sources(state: V3AnnotationSeed) -> None:
    expected_sources = set(state.quotas.sources)
    unexpected_sources = {candidate.source for candidate in state.reserve_candidates} - expected_sources
    if unexpected_sources:
        raise V3AnnotationSeedError("V3 reserve candidates contain sources outside the V3 quota matrix")


def _v2_cells(state: V3AnnotationSeed) -> set[str]:
    cells = {row.candidate.h3_cell for row in state.seeded_rows}
    cells.update(annotation.candidate.h3_cell for annotation in state.excluded_v2_rows)
    return cells


def _quota_keys(quotas: SourceLabelQuotas) -> tuple[QuotaKey, ...]:
    return tuple((source, label) for source in quotas.sources for label in quotas.labels)


def _quotas_to_dict(quotas: SourceLabelQuotas) -> dict[str, dict[str, int]]:
    return {
        source.value: {label.value: quotas.required(source, label) for label in quotas.labels}
        for source in quotas.sources
    }


def _excluded_v2_reasons(seed_plan: SeedPlan) -> dict[str, str]:
    return {
        annotation.candidate.candidate_id: (
            "Excluded as a deterministic V3 quota surplus; the frozen V2 row remains reserved "
            "but is not part of the selected V3 quota."
        )
        for annotation in seed_plan.excluded
    }


def _rank(seed: str, candidate_id: str) -> str:
    return hashlib.sha256(f"{seed}:{candidate_id}".encode()).hexdigest()


def _validate_selection_seed(seed: str) -> None:
    if not isinstance(seed, str) or not seed.strip():
        raise V3AnnotationSeedError("V3 selection seed must be a non-empty string")


def _validate_selection_rank(rank: str) -> None:
    if not _is_sha256_digest(rank):
        raise V3AnnotationSeedError("V3 selection rank must be a SHA-256 digest")


def _validate_selection_slot(slot_index: int) -> None:
    if not isinstance(slot_index, int) or isinstance(slot_index, bool) or slot_index < 0:
        raise V3AnnotationSeedError("V3 selection slot index must be a non-negative integer")


def _validate_seed_row_source(row: V3SeedRow) -> None:
    if row.candidate.source is not row.quota_source:
        raise V3AnnotationSeedError("V3 seed row candidate source does not match quota source")


def _validate_frozen_seed_row(row: V3SeedRow) -> None:
    if row.annotation is None or row.annotation.candidate != row.candidate:
        raise V3AnnotationSeedError("V2 seed row must carry its frozen annotation")
    if row.annotation.label is not row.quota_label:
        raise V3AnnotationSeedError("V2 seed row annotation quota must match its quota slot")


def _validate_fresh_seed_row(row: V3SeedRow) -> None:
    if row.annotation is not None:
        raise V3AnnotationSeedError("fresh V3 seed row must not carry a human label")


def _validate_seed_identity(seed: str, benchmark_sha256: str) -> None:
    if not isinstance(seed, str) or not seed.strip():
        raise V3AnnotationSeedError("V3 seed must be non-empty")
    if not _is_sha256_digest(benchmark_sha256):
        raise V3AnnotationSeedError("benchmark SHA-256 must be a 64-character hexadecimal digest")


def _is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _seed_rows_from_dict(payload: Mapping[str, Any]) -> tuple[V3SeedRow, ...]:
    return tuple(V3SeedRow.from_dict(dict(row)) for row in payload["rows"])


def _excluded_rows_from_dict(payload: Mapping[str, Any]) -> tuple[Annotation, ...]:
    return tuple(Annotation.from_dict(dict(row)) for row in payload["excluded_v2_rows"])


def _reserved_cells_from_dict(payload: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(str(cell) for cell in payload["reserved_v2_cells"])


def _quotas_from_dict(payload: Mapping[str, Any]) -> SourceLabelQuotas:
    quotas = payload["quotas"]
    if not isinstance(quotas, Mapping):
        raise ValueError("V3 seed quotas must be an object")
    counts: dict[QuotaKey, int] = {}
    for source_name, labels in quotas.items():
        if not isinstance(labels, Mapping):
            raise ValueError("V3 seed quota source must be an object")
        source = Source(str(source_name))
        for label_name, count in labels.items():
            counts[(source, Label(str(label_name)))] = int(count)
    return SourceLabelQuotas(counts)
