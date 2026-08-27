from __future__ import annotations

from landuse_sentence_relevance.domain.cell_quota import CellQuota


def test_cell_quota_marks_a_cell_full_at_or_above_capacity() -> None:
    quota = CellQuota(capacity=2)
    counts = {"cell-a": 2, "cell-b": 1}

    assert quota.is_full("cell-a", counts)
    assert quota.is_full("cell-c", {"cell-c": 3})
    assert not quota.is_full("cell-b", counts)


def test_cell_quota_counts_cells_using_a_completion_threshold() -> None:
    quota = CellQuota(capacity=8, completion_threshold=2)
    counts = {"cell-a": 2, "cell-b": 8, "cell-c": 1}

    assert quota.filled_cells(("cell-a", "cell-b", "cell-c"), counts) == 2


def test_cell_quota_reaches_a_target_number_of_filled_cells() -> None:
    quota = CellQuota(capacity=2)
    cells = ("cell-a", "cell-b", "cell-c")
    counts = {"cell-a": 2, "cell-b": 1, "cell-c": 2}

    assert quota.is_reached(cells, counts, target_cells=2)
    assert not quota.is_reached(cells, counts, target_cells=3)
