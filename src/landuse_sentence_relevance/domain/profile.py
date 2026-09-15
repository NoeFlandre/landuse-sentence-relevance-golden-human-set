from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from landuse_sentence_relevance.domain.models import Label, Source

_SOURCE_ORDER = tuple(Source)
_LABEL_ORDER = tuple(Label)

QuotaKey = tuple[Source, Label]


@dataclass(frozen=True, slots=True)
class SourceLabelQuotas:
    """How many rows a finished dataset needs for each source and label.

    A marginal contract cannot express "fifty yes and fifty no *within* each
    source", so V3 states the whole matrix. Pairs the dataset does not want are
    simply absent, and reported as zero.
    """

    counts: Mapping[QuotaKey, int]
    h3_resolution: int = 3
    rows_per_cell: int = 1

    def __post_init__(self) -> None:
        if not self.counts:
            raise ValueError("quotas must require at least one source and label")
        if any(count < 0 for count in self.counts.values()):
            raise ValueError("a quota must not be negative")
        if self.h3_resolution != 3:
            raise ValueError("h3_resolution must be 3 for the project sampling grid")
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))

    def required(self, source: Source, label: Label) -> int:
        return self.counts.get((source, label), 0)

    def for_source(self, source: Source) -> int:
        return sum(self.required(source, label) for label in self.labels)

    @property
    def sources(self) -> tuple[Source, ...]:
        present = {source for source, _ in self.counts}
        return tuple(source for source in _SOURCE_ORDER if source in present)

    @property
    def labels(self) -> tuple[Label, ...]:
        present = {label for _, label in self.counts}
        return tuple(label for label in _LABEL_ORDER if label in present)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def balanced_quotas(sources: Sequence[Source], rows_per_source_label: int) -> SourceLabelQuotas:
    """Require the same number of rows for every source and label combination."""

    return SourceLabelQuotas(
        {(source, label): rows_per_source_label for source in sources for label in _LABEL_ORDER}
    )


V3_SOURCES: tuple[Source, ...] = (Source.WIKIPEDIA, Source.WEBSITE, Source.DESCRIPTION)
V3_QUOTAS = balanced_quotas(V3_SOURCES, rows_per_source_label=50)
