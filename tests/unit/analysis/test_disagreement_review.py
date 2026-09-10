from __future__ import annotations

from landuse_sentence_relevance.analysis.disagreement_review import (
    CONTEXT_COLUMNS,
    majority_label,
    minority_raters,
    review_header,
    review_rows,
)

HUMAN = {"a field": "yes", "a meeting": "no", "a road": "yes", "a treaty": "no"}
GPT = {"a field": "yes", "a meeting": "no", "a road": "no", "a treaty": "no"}
CLAUDE = {"a field": "yes", "a meeting": "yes", "a road": "yes", "a treaty": "no"}
RATERS = {"human": HUMAN, "gpt": GPT, "claude": CLAUDE}
SENTENCES = ("a field", "a meeting", "a road", "a treaty")
CONTEXT = {
    sentence: {
        "source": "wikipedia",
        "region": "fiji",
        "polygon_name": f"place for {sentence}",
        "source_url": f"https://example.invalid/{sentence.replace(' ', '-')}",
    }
    for sentence in SENTENCES
}


def test_context_columns_carry_the_provenance_a_reviewer_needs() -> None:
    assert CONTEXT_COLUMNS == ("source", "region", "polygon_name", "source_url")


def test_majority_label_returns_the_strictly_most_common_label() -> None:
    assert majority_label({"human": "no", "gpt": "no", "claude": "yes"}) == "no"
    assert majority_label({"human": "yes", "gpt": "no", "claude": "yes"}) == "yes"


def test_majority_label_is_none_when_no_label_is_strictly_most_common() -> None:
    assert majority_label({"human": "yes", "gpt": "no"}) is None


def test_majority_label_is_none_without_any_rater() -> None:
    assert majority_label({}) is None


def test_minority_raters_names_the_raters_outvoted_by_the_majority() -> None:
    assert minority_raters({"human": "no", "gpt": "no", "claude": "yes"}) == ("claude",)
    assert minority_raters({"human": "yes", "gpt": "no", "claude": "no"}) == ("human",)


def test_minority_raters_is_empty_when_the_vote_is_tied() -> None:
    assert minority_raters({"human": "yes", "gpt": "no"}) == ()


def test_minority_raters_is_empty_when_every_rater_agrees() -> None:
    assert minority_raters({"human": "no", "gpt": "no", "claude": "no"}) == ()


def test_review_header_puts_the_verdict_first_and_the_sentence_beside_the_labels() -> None:
    assert review_header(("claude", "gpt", "human"), CONTEXT_COLUMNS) == (
        "minority_rater",
        "minority_label",
        "claude",
        "gpt",
        "human",
        "sentence",
        "source",
        "region",
        "polygon_name",
        "source_url",
    )


def test_review_rows_keep_only_disagreements_grouped_by_the_outvoted_rater() -> None:
    rows = review_rows(RATERS, SENTENCES, CONTEXT)

    assert [(row["minority_rater"], row["minority_label"], row["sentence"]) for row in rows] == [
        ("claude", "yes", "a meeting"),
        ("gpt", "no", "a road"),
    ]


def test_review_rows_carry_every_rater_label_and_the_row_context() -> None:
    rows = review_rows(RATERS, SENTENCES, CONTEXT)

    assert rows[0] == {
        "minority_rater": "claude",
        "minority_label": "yes",
        "claude": "yes",
        "gpt": "no",
        "human": "no",
        "source": "wikipedia",
        "region": "fiji",
        "polygon_name": "place for a meeting",
        "source_url": "https://example.invalid/a-meeting",
        "sentence": "a meeting",
    }


def test_review_rows_follow_an_explicit_rater_order() -> None:
    rows = review_rows(RATERS, SENTENCES, CONTEXT, rater_names=("human", "gpt", "claude"))

    assert tuple(rows[0]) == review_header(("human", "gpt", "claude"), CONTEXT_COLUMNS)
    assert list(rows[0].values())[2:5] == ["no", "no", "yes"]


def test_review_rows_match_the_header_exactly() -> None:
    rows = review_rows(RATERS, SENTENCES, CONTEXT)
    header = review_header(sorted(RATERS), CONTEXT_COLUMNS)

    assert all(tuple(row) == header for row in rows)


def test_review_rows_are_empty_when_every_rater_agrees() -> None:
    unanimous = {"human": HUMAN, "copy": dict(HUMAN)}

    assert review_rows(unanimous, SENTENCES, CONTEXT) == []


def test_review_rows_join_several_outvoted_raters_sharing_one_label() -> None:
    raters = {
        "a": {"s": "yes"},
        "b": {"s": "yes"},
        "c": {"s": "yes"},
        "d": {"s": "no"},
        "e": {"s": "no"},
    }
    context = {"s": dict.fromkeys(CONTEXT_COLUMNS, "")}

    rows = review_rows(raters, ("s",), context)

    assert rows[0]["minority_rater"] == "d; e"
    assert rows[0]["minority_label"] == "no"


def test_review_rows_join_distinct_minority_labels() -> None:
    raters = {
        "a": {"s": "yes"},
        "b": {"s": "yes"},
        "c": {"s": "no"},
        "d": {"s": "maybe"},
    }
    context = {"s": dict.fromkeys(CONTEXT_COLUMNS, "")}

    rows = review_rows(raters, ("s",), context)

    assert rows[0]["minority_rater"] == "c; d"
    assert rows[0]["minority_label"] == "maybe; no"


def test_review_rows_sort_by_outvoted_rater_then_label_then_sentence() -> None:
    raters = {
        "human": {"s1": "yes", "s2": "no", "s3": "no"},
        "gpt": {"s1": "no", "s2": "yes", "s3": "yes"},
        "claude": {"s1": "no", "s2": "no", "s3": "no"},
    }
    context = {name: dict.fromkeys(CONTEXT_COLUMNS, "") for name in ("s1", "s2", "s3")}

    rows = review_rows(raters, ("s1", "s2", "s3"), context)

    assert [(row["minority_rater"], row["minority_label"], row["sentence"]) for row in rows] == [
        ("gpt", "yes", "s2"),
        ("gpt", "yes", "s3"),
        ("human", "yes", "s1"),
    ]
