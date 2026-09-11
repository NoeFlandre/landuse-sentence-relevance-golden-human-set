from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from landuse_sentence_relevance.domain.sentence_selection import (
    SentencePart,
    candidate_sentence_parts,
    prioritize_sentences,
)
from landuse_sentence_relevance.domain.stratification import select_spread_cells

pytestmark = pytest.mark.property

_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=40,
)
_PROPERTY_SETTINGS = settings(max_examples=40, derandomize=True, database=None, deadline=None)


@_PROPERTY_SETTINGS
@given(sentences=st.lists(_TEXT, max_size=16))
def test_candidate_parts_are_a_permutation_of_cleaned_input(sentences: list[str]) -> None:
    cleaned = tuple(sentence.strip() for sentence in sentences if sentence.strip())

    parts = candidate_sentence_parts(sentences, seed="property")

    assert len(parts) == len(cleaned)
    assert {part.index for part in parts} == set(range(len(cleaned)))
    assert tuple(part.text for part in sorted(parts, key=lambda part: part.index)) == cleaned
    assert all(part.text for part in parts)


@_PROPERTY_SETTINGS
@given(sentences=st.lists(_TEXT, max_size=16))
def test_prioritization_returns_contextual_matches_or_the_first_fallback(sentences: list[str]) -> None:
    cleaned = tuple(sentence.strip() for sentence in sentences if sentence.strip())
    accepted = tuple(
        SentencePart(index, sentence) for index, sentence in enumerate(cleaned) if sentence.endswith("!")
    )

    parts = prioritize_sentences(sentences, seed="property", accept=lambda text: text.endswith("!"))

    contextual = tuple(part for part in accepted if part.index > 0)
    if contextual:
        assert {part.index for part in parts} == {part.index for part in contextual}
        assert all(part.index > 0 for part in parts)
    else:
        assert parts == accepted[:1]


@_PROPERTY_SETTINGS
@given(
    cells=st.lists(
        st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1),
        max_size=20,
    ),
    target_count=st.integers(min_value=1, max_value=25),
)
def test_spread_selection_returns_unique_available_cells_with_a_bounded_count(
    cells: list[str], target_count: int
) -> None:
    available = tuple(sorted(set(cells)))
    centers = {cell: (0.0, float(index)) for index, cell in enumerate(available)}

    selected = select_spread_cells(
        cells,
        target_count,
        centers.__getitem__,
        seed="property",
    )

    assert len(selected) == min(target_count, len(available))
    assert len(set(selected)) == len(selected)
    assert set(selected) <= set(available)
    assert selected == select_spread_cells(cells, target_count, centers.__getitem__, seed="property")
