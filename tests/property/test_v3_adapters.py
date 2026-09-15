from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from landuse_sentence_relevance.sources.v3 import WebsiteSentenceSource

pytestmark = pytest.mark.property


@st.composite
def sentence_lists(draw: st.DrawFn) -> list[str]:
    return draw(
        st.lists(
            st.sampled_from((" ", "\t", "alpha", " beta ", "a place")),
            min_size=1,
            max_size=8,
        )
    )


def _source(row: Mapping[str, Any]) -> WebsiteSentenceSource:
    rows: Iterable[Mapping[str, Any]] = (row,)
    return WebsiteSentenceSource(
        row_shards_loader=lambda: (rows,),
        cell_for_location=lambda latitude, longitude: "8928308280fffff",
    )


@given(sentence_lists())
def test_website_adapter_preserves_each_non_empty_upstream_sentence(texts: list[str]) -> None:
    row = {
        "polygon_id": "p1",
        "lat": 45.0,
        "lon": 2.0,
        "website_language": "eng_Latn",
        "website_language_probability": 1.0,
        "website_sentences": texts,
    }

    candidates = list(_source(row).iter_candidates())

    assert [candidate.sentence for candidate in candidates] == [
        text.strip() for text in texts if text.strip()
    ]
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
