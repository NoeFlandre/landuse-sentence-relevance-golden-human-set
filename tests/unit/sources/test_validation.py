import pytest

from landuse_sentence_relevance.sources.validation import (
    validate_candidate_cell_settings,
    validate_optional_limit,
    validate_positive_limit,
)


def test_candidate_cell_settings_reject_a_non_positive_count() -> None:
    with pytest.raises(ValueError, match="candidate_cell_count must be positive"):
        validate_candidate_cell_settings(0, None)


def test_positive_limit_rejects_a_non_positive_value() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        validate_positive_limit(0, "limit")


@pytest.mark.parametrize("value", [None, 1])
def test_optional_limit_accepts_none_or_a_positive_value(value: int | None) -> None:
    validate_optional_limit(value, "limit")


def test_candidate_cell_settings_report_the_exact_missing_center_error() -> None:
    with pytest.raises(ValueError) as error:
        validate_candidate_cell_settings(1, None)

    assert str(error.value) == "center_of_cell is required when candidate_cell_count is set"
