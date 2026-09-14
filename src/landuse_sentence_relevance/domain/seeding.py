from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from landuse_sentence_relevance.domain.models import Annotation
from landuse_sentence_relevance.domain.profile import QuotaKey, SourceLabelQuotas


def _rank(seed: str, candidate_id: str) -> str:
    return hashlib.sha256(f"{seed}:{candidate_id}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SeedPlan:
    """Which already-labeled rows a new dataset keeps, and what is left to label."""

    reused: tuple[Annotation, ...]
    excluded: tuple[Annotation, ...]
    remaining: Mapping[QuotaKey, int]
    reserved_cells: frozenset[str]


def _group(existing: tuple[Annotation, ...]) -> dict[QuotaKey, list[Annotation]]:
    grouped: dict[QuotaKey, list[Annotation]] = {}
    for row in existing:
        grouped.setdefault((row.candidate.source, row.label), []).append(row)
    return grouped


def _ordered(rows: list[Annotation], seed: str) -> list[Annotation]:
    return sorted(rows, key=lambda row: (_rank(seed, row.candidate.candidate_id), row.candidate.candidate_id))


def plan_seed(
    existing: Iterable[Annotation],
    quotas: SourceLabelQuotas,
    seed: str,
) -> SeedPlan:
    """Reuse what the quotas still need, drop the surplus, and report the shortfall.

    Every cell the existing rows occupy is reserved, including the cells of
    excluded rows, so newly built candidates never reuse one of them.
    """

    rows = tuple(existing)
    grouped = _group(rows)
    reused: list[Annotation] = []
    excluded: list[Annotation] = []
    for key, group in grouped.items():
        source, label = key
        keep = min(len(group), quotas.required(source, label))
        ordered = _ordered(group, seed)
        reused.extend(ordered[:keep])
        excluded.extend(ordered[keep:])
    remaining = _remaining(quotas, grouped)
    return SeedPlan(
        reused=tuple(_ordered(reused, seed)),
        excluded=tuple(_ordered(excluded, seed)),
        remaining=remaining,
        reserved_cells=frozenset(row.candidate.h3_cell for row in rows),
    )


def _remaining(
    quotas: SourceLabelQuotas,
    grouped: Mapping[QuotaKey, list[Annotation]],
) -> dict[QuotaKey, int]:
    shortfall: dict[QuotaKey, int] = {}
    for source in quotas.sources:
        for label in quotas.labels:
            missing = quotas.required(source, label) - len(grouped.get((source, label), ()))
            if missing > 0:
                shortfall[(source, label)] = missing
    return shortfall


def remaining_total(plan: SeedPlan) -> int:
    """Return how many rows still need a human label."""

    return sum(plan.remaining.values())


def seeded_counts(plan: SeedPlan) -> dict[QuotaKey, int]:
    """Count the reused rows for each source and label."""

    counts: dict[QuotaKey, int] = {}
    for row in plan.reused:
        key = (row.candidate.source, row.label)
        counts[key] = counts.get(key, 0) + 1
    return counts
