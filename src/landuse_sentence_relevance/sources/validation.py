from __future__ import annotations

from collections.abc import Callable


def validate_positive_limit(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def validate_optional_limit(value: int | None, name: str) -> None:
    if value is not None:
        validate_positive_limit(value, name)


def validate_candidate_cell_settings(
    count: int | None,
    center_of_cell: Callable[[str], tuple[float, float]] | None,
) -> None:
    if count is None:
        return
    validate_positive_limit(count, "candidate_cell_count")
    if center_of_cell is None:
        raise ValueError("center_of_cell is required when candidate_cell_count is set")
