"""Builders the test suite shares, in a module whose job is to provide them.

They used to live in `tests/unit/test_models.py`, `tests/unit/test_constraints.py` and
`tests/unit/test_v3_workflow.py`, and 25 modules imported them from there. Importing a test
module *runs* it, so collecting one file executed another's body, and renaming a helper -- or
moving a test file when the code it covers is reorganised -- broke files that had nothing to do
with the change.

Plain functions rather than fixtures: most callers want several variants inside one test
(`replace(make_candidate("a"), source=...)`), which a fixture cannot give without a factory
wrapper that reads worse than the call it replaces. Anything needing setup and teardown belongs
in a `conftest.py` fixture instead.
"""

from __future__ import annotations

from dataclasses import replace

from landuse_sentence_relevance.domain.models import Annotation, Candidate, Label, Source
from landuse_sentence_relevance.domain.profile import V3_SOURCES, balanced_quotas
from landuse_sentence_relevance.domain.v3_annotation import (
    V3AnnotationSeed,
    V3SeedRow,
    V3SelectionMetadata,
)

__all__ = ["make_annotation_seed", "make_annotations", "make_candidate", "make_v3_candidate"]


def make_candidate(candidate_id: str = "c-1") -> Candidate:
    """One valid candidate, with every provenance field populated."""
    return Candidate(
        candidate_id=candidate_id,
        sentence="A sentence about a visible landscape.",
        source=Source.WIKIPEDIA,
        source_record_id="polygon-1",
        source_field="wikipedia_section",
        h3_cell="832830fffffffff",
        h3_resolution=3,
        latitude=45.0,
        longitude=2.0,
        place_name="A place",
        region="A region",
        source_url="https://example.test/article",
    )


def make_annotations() -> list[Annotation]:
    """A complete 100-row set: 50 Wikipedia yes, 50 website no, one cell each."""
    annotations: list[Annotation] = []
    for cell_number in range(100):
        cell = f"cell-{cell_number:02d}"
        source = Source.WIKIPEDIA if cell_number < 50 else Source.WEBSITE
        candidate = replace(
            make_candidate(f"{source.value}-{cell}"),
            source=source,
            h3_cell=cell,
        )
        label = Label.YES if cell_number < 50 else Label.NO
        annotations.append(Annotation(candidate=candidate, label=label))
    return annotations


def make_v3_candidate(candidate_id: str, source: Source, cell: str) -> Candidate:
    """A candidate placed in a named source and cell, for quota and seeding tests."""
    return replace(
        make_candidate(candidate_id),
        sentence=f"Sentence for {candidate_id}.",
        source=source,
        h3_cell=cell,
    )


def make_annotation_seed() -> V3AnnotationSeed:
    """A minimal V3 seed: one row per (source, label), the first already labelled from V2."""
    quotas = balanced_quotas(V3_SOURCES, rows_per_source_label=1)
    rows: list[V3SeedRow] = []
    for index, (source, label) in enumerate(quotas.counts):
        candidate = make_v3_candidate(
            f"{source.value}-{label.value}",
            source,
            f"{source.value}-{label.value}-cell",
        )
        annotation = Annotation(candidate, label) if index == 0 else None
        rows.append(
            V3SeedRow(
                candidate=candidate,
                quota_source=source,
                quota_label=label,
                origin="v2" if annotation is not None else "v3",
                annotation=annotation,
                selection=V3SelectionMetadata(seed="workflow-test", rank="0" * 64, slot_index=index),
            )
        )
    return V3AnnotationSeed(
        rows=tuple(rows),
        excluded_v2_rows=(),
        reserved_v2_cells=frozenset({"wikipedia-yes-cell"}),
        quotas=quotas,
        benchmark_sha256="0" * 64,
        seed="workflow-test",
    )
