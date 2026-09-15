from __future__ import annotations

import pytest

from landuse_sentence_relevance.domain.models import Label, Source
from landuse_sentence_relevance.domain.profile import (
    V3_QUOTAS,
    SourceLabelQuotas,
    balanced_quotas,
)


def test_balanced_quotas_require_the_same_count_for_every_source_and_label() -> None:
    quotas = balanced_quotas((Source.WIKIPEDIA, Source.WEBSITE), rows_per_source_label=25)

    assert quotas.required(Source.WIKIPEDIA, Label.YES) == 25
    assert quotas.required(Source.WEBSITE, Label.NO) == 25
    assert quotas.total == 100


def test_v3_quotas_are_three_hundred_rows_balanced_across_three_sources() -> None:
    assert V3_QUOTAS.total == 300
    assert V3_QUOTAS.sources == (Source.WIKIPEDIA, Source.WEBSITE, Source.DESCRIPTION)
    assert V3_QUOTAS.labels == (Label.YES, Label.NO)
    assert all(
        V3_QUOTAS.required(source, label) == 50 for source in V3_QUOTAS.sources for label in V3_QUOTAS.labels
    )
    assert all(V3_QUOTAS.for_source(source) == 100 for source in V3_QUOTAS.sources)
    assert V3_QUOTAS.h3_resolution == 3
    assert V3_QUOTAS.rows_per_cell == 1


def test_quotas_report_zero_for_a_pair_they_do_not_require() -> None:
    quotas = SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 3})

    assert quotas.required(Source.WEBSITE, Label.NO) == 0
    assert quotas.sources == (Source.WIKIPEDIA,)
    assert quotas.labels == (Label.YES,)
    assert quotas.total == 3


def test_quotas_reject_a_negative_requirement() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): -1})


def test_quotas_reject_an_empty_matrix() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SourceLabelQuotas({})


def test_quotas_reject_a_resolution_other_than_three() -> None:
    with pytest.raises(ValueError, match="h3_resolution"):
        SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 1}, h3_resolution=7)


def test_quotas_order_sources_and_labels_deterministically() -> None:
    quotas = SourceLabelQuotas(
        {
            (Source.DESCRIPTION, Label.NO): 1,
            (Source.WIKIPEDIA, Label.YES): 1,
            (Source.WEBSITE, Label.NO): 1,
        }
    )

    assert quotas.sources == (Source.WIKIPEDIA, Source.WEBSITE, Source.DESCRIPTION)
    assert quotas.labels == (Label.YES, Label.NO)
