from __future__ import annotations

from dataclasses import replace

from tests.builders import make_candidate

from landuse_sentence_relevance.domain.models import Annotation, Label, Source
from landuse_sentence_relevance.domain.profile import SourceLabelQuotas, balanced_quotas
from landuse_sentence_relevance.domain.seeding import plan_seed

SEED = "test-seed"


def annotation(identifier: str, source: Source, label: Label, cell: str) -> Annotation:
    candidate = replace(make_candidate(identifier), source=source, h3_cell=cell)
    return Annotation(candidate=candidate, label=label)


def make_rows(source: Source, label: Label, count: int, prefix: str) -> list[Annotation]:
    return [
        annotation(f"{prefix}-{index:03d}", source, label, f"{prefix}-cell-{index:03d}")
        for index in range(count)
    ]


def test_seed_plan_reuses_every_row_a_quota_still_needs() -> None:
    existing = make_rows(Source.WIKIPEDIA, Label.YES, 3, "wy")
    quotas = SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 5})

    plan = plan_seed(existing, quotas, seed=SEED)

    assert len(plan.reused) == 3
    assert plan.excluded == ()
    assert plan.remaining == {(Source.WIKIPEDIA, Label.YES): 2}


def test_seed_plan_excludes_the_surplus_of_an_over_supplied_pair() -> None:
    existing = make_rows(Source.WIKIPEDIA, Label.NO, 4, "wn")
    quotas = SourceLabelQuotas({(Source.WIKIPEDIA, Label.NO): 3})

    plan = plan_seed(existing, quotas, seed=SEED)

    assert len(plan.reused) == 3
    assert len(plan.excluded) == 1
    assert plan.remaining == {}


def test_seed_plan_excludes_deterministically_for_a_fixed_seed() -> None:
    existing = make_rows(Source.WIKIPEDIA, Label.NO, 4, "wn")
    quotas = SourceLabelQuotas({(Source.WIKIPEDIA, Label.NO): 3})

    first = plan_seed(existing, quotas, seed=SEED)
    second = plan_seed(list(reversed(existing)), quotas, seed=SEED)

    assert first.excluded == second.excluded
    assert first.reused == second.reused


def test_seed_plan_excludes_every_row_of_a_source_the_quota_does_not_want() -> None:
    existing = make_rows(Source.WEBSITE, Label.YES, 2, "ws")
    quotas = SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 1})

    plan = plan_seed(existing, quotas, seed=SEED)

    assert plan.reused == ()
    assert len(plan.excluded) == 2
    assert plan.remaining == {(Source.WIKIPEDIA, Label.YES): 1}


def test_seed_plan_reserves_every_cell_it_touched_including_excluded_rows() -> None:
    existing = make_rows(Source.WIKIPEDIA, Label.NO, 4, "wn")
    quotas = SourceLabelQuotas({(Source.WIKIPEDIA, Label.NO): 3})

    plan = plan_seed(existing, quotas, seed=SEED)

    assert plan.reserved_cells == frozenset(row.candidate.h3_cell for row in existing)
    assert len(plan.reserved_cells) == 4


def test_seed_plan_matches_the_v3_arithmetic_of_the_released_v2_benchmark() -> None:
    existing = (
        make_rows(Source.WIKIPEDIA, Label.YES, 49, "wy")
        + make_rows(Source.WIKIPEDIA, Label.NO, 51, "wn")
        + make_rows(Source.WEBSITE, Label.YES, 31, "sy")
        + make_rows(Source.WEBSITE, Label.NO, 23, "sn")
    )
    quotas = balanced_quotas((Source.WIKIPEDIA, Source.WEBSITE, Source.DESCRIPTION), rows_per_source_label=50)

    plan = plan_seed(existing, quotas, seed=SEED)

    assert len(plan.reused) == 153
    assert len(plan.excluded) == 1
    assert plan.excluded[0].candidate.source is Source.WIKIPEDIA
    assert plan.excluded[0].label is Label.NO
    assert plan.remaining == {
        (Source.WIKIPEDIA, Label.YES): 1,
        (Source.WEBSITE, Label.YES): 19,
        (Source.WEBSITE, Label.NO): 27,
        (Source.DESCRIPTION, Label.YES): 50,
        (Source.DESCRIPTION, Label.NO): 50,
    }
    assert sum(plan.remaining.values()) == 147
    assert len(plan.reserved_cells) == 154


def test_seed_plan_selects_and_orders_by_the_seed() -> None:
    existing = make_rows(Source.WIKIPEDIA, Label.NO, 6, "wn")
    quotas = SourceLabelQuotas({(Source.WIKIPEDIA, Label.NO): 3})

    first = plan_seed(existing, quotas, seed="seed-a")
    second = plan_seed(existing, quotas, seed="seed-b")

    def identifiers(rows: tuple[Annotation, ...]) -> list[str]:
        return [row.candidate.candidate_id for row in rows]

    assert identifiers(first.reused) != identifiers(second.reused)
    assert identifiers(first.excluded) != identifiers(second.excluded)
    assert set(identifiers(first.reused)) != set(identifiers(second.reused))


def test_seed_plan_reports_no_shortfall_for_an_exactly_satisfied_pair() -> None:
    existing = make_rows(Source.WIKIPEDIA, Label.YES, 3, "wy")
    quotas = SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 3})

    plan = plan_seed(existing, quotas, seed=SEED)

    assert plan.remaining == {}
    assert len(plan.reused) == 3
    assert plan.excluded == ()
