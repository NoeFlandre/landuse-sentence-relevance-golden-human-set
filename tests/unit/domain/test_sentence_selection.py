from landuse_sentence_relevance.domain.sentence_selection import (
    SentencePart,
    candidate_sentence_parts,
    first_prioritized_sentence,
    prioritize_sentences,
)


def test_prioritization_excludes_the_first_sentence_when_a_later_sentence_is_accepted() -> None:
    parts = prioritize_sentences(
        ["Title-like heading", "The place is surrounded by wetlands.", "It sits on a broad plain."],
        seed="v2",
    )

    assert {part.text for part in parts} == {
        "The place is surrounded by wetlands.",
        "It sits on a broad plain.",
    }
    assert all(part.index > 0 for part in parts)


def test_prioritization_is_deterministic_for_the_same_seed() -> None:
    sentences = ["Heading", "The first paragraph sentence.", "The second paragraph sentence."]

    first = prioritize_sentences(sentences, seed="other")
    second = prioritize_sentences(sentences, seed="other")

    assert first == second
    assert [part.index for part in first] == [2, 1]


def test_candidate_sentence_parts_keep_the_title_as_a_last_resort() -> None:
    parts = candidate_sentence_parts(
        ["Heading", "The first paragraph sentence.", "The second paragraph sentence."],
        seed="other",
    )

    assert [part.index for part in parts] == [2, 1, 0]


def test_candidate_sentence_parts_return_the_only_sentence() -> None:
    assert candidate_sentence_parts(["Only sentence."], seed="v2") == (
        SentencePart(index=0, text="Only sentence."),
    )


def test_prioritization_falls_back_to_the_first_accepted_sentence() -> None:
    parts = prioritize_sentences(
        ["Only available sentence.", "Non-English sentence."],
        seed="v2",
        accept=lambda sentence: sentence.startswith("Only"),
    )

    assert parts == (SentencePart(index=0, text="Only available sentence."),)


def test_prioritization_returns_no_parts_when_no_sentence_is_accepted() -> None:
    parts = prioritize_sentences(["Not available."], seed="v2", accept=lambda sentence: False)

    assert parts == ()


def test_first_prioritized_sentence_stops_after_the_first_accepted_contextual_sentence() -> None:
    seen: list[str] = []

    def accept(sentence: str) -> bool:
        seen.append(sentence)
        return sentence == "The selected paragraph sentence."

    selected = first_prioritized_sentence(
        [
            "Heading",
            "The selected paragraph sentence.",
            "Another paragraph sentence.",
        ],
        seed="v2",
        accept=accept,
    )

    assert selected == SentencePart(index=1, text="The selected paragraph sentence.")
    assert len(seen) < 3


def test_first_prioritized_sentence_uses_the_seed_for_contextual_ordering() -> None:
    selected = first_prioritized_sentence(
        ["Heading", "The first paragraph sentence.", "The second paragraph sentence."],
        seed="other",
    )

    assert selected == SentencePart(index=2, text="The second paragraph sentence.")


def test_first_prioritized_sentence_preserves_the_first_accepted_fallback() -> None:
    seen: list[str] = []

    def accept(sentence: str) -> bool:
        seen.append(sentence)
        return sentence == "Heading"

    selected = first_prioritized_sentence(
        ["Heading", "Rejected paragraph sentence."],
        seed="v2",
        accept=accept,
    )

    assert selected == SentencePart(index=0, text="Heading")
    assert seen == ["Rejected paragraph sentence.", "Heading"]


def test_first_prioritized_sentence_does_not_recheck_rejected_context() -> None:
    seen: list[str] = []

    def reject(sentence: str) -> bool:
        seen.append(sentence)
        return False

    selected = first_prioritized_sentence(
        ["Heading", "Rejected paragraph sentence."],
        seed="v2",
        accept=reject,
    )

    assert selected is None
    assert seen == ["Rejected paragraph sentence.", "Heading"]
