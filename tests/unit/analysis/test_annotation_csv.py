from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from landuse_sentence_relevance.analysis.annotation_csv import (
    SENTENCE_COLUMN,
    read_rater_labels,
    read_sentence_context,
    read_table,
)
from landuse_sentence_relevance.analysis.interrater import InterraterDataError


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_sentence_column_is_the_shared_join_key() -> None:
    assert SENTENCE_COLUMN == "sentence"


def test_read_rater_labels_reads_sentences_and_labels_by_column_name(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "human.csv",
        'sentence,label,region\n"A field, ploughed.",yes,fiji\n"A council meeting.",no,fiji\n',
    )

    assert read_rater_labels(path, "human", "label") == {
        "A field, ploughed.": "yes",
        "A council meeting.": "no",
    }


def test_read_rater_labels_ignores_column_position(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "llm.csv", "region,llm_label,sentence\nfiji,yes,A field.\n")

    assert read_rater_labels(path, "gpt", "llm_label") == {"A field.": "yes"}


def test_read_rater_labels_rejects_a_missing_sentence_column(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "llm.csv", "text,llm_label\nA field.,yes\n")

    with pytest.raises(InterraterDataError, match=rf"^{re.escape(str(path))} is missing column 'sentence'$"):
        read_rater_labels(path, "gpt", "llm_label")


def test_read_rater_labels_rejects_a_missing_label_column(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "llm.csv", "sentence,label\nA field.,yes\n")

    with pytest.raises(InterraterDataError, match=rf"^{re.escape(str(path))} is missing column 'llm_label'$"):
        read_rater_labels(path, "gpt", "llm_label")


def test_read_rater_labels_rejects_an_empty_file(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "empty.csv", "")

    with pytest.raises(InterraterDataError, match=rf"^{re.escape(str(path))} is missing column 'sentence'$"):
        read_rater_labels(path, "gpt", "llm_label")


def test_read_rater_labels_rejects_a_row_with_a_missing_value(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "short.csv", "sentence,llm_label\nA field.\n")

    with pytest.raises(
        InterraterDataError,
        match=rf"^{re.escape(str(path))} has a row missing one of \['sentence', 'llm_label'\]$",
    ):
        read_rater_labels(path, "gpt", "llm_label")


def test_read_rater_labels_rejects_an_invalid_label(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "bad.csv", "sentence,llm_label\nA field.,perhaps\n")

    with pytest.raises(InterraterDataError, match=r"^gpt has an invalid label 'perhaps'"):
        read_rater_labels(path, "gpt", "llm_label")


def test_read_rater_labels_rejects_a_duplicated_sentence(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "dupe.csv", "sentence,llm_label\nA field.,yes\nA field.,no\n")

    with pytest.raises(InterraterDataError, match=r"^gpt has a duplicate sentence"):
        read_rater_labels(path, "gpt", "llm_label")


def test_read_rater_labels_rejects_a_header_only_file(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "header.csv", "sentence,llm_label\n")

    with pytest.raises(InterraterDataError, match=r"^gpt has no rows$"):
        read_rater_labels(path, "gpt", "llm_label")


def test_read_rater_labels_opens_the_file_as_utf8_without_newline_translation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_csv(tmp_path / "llm.csv", "sentence,llm_label\nA field.,yes\n")
    recorded: list[dict[str, Any]] = []
    original_open = Path.open

    def record_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        recorded.append(dict(kwargs))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record_open)

    assert read_rater_labels(path, "gpt", "llm_label") == {"A field.": "yes"}
    assert recorded == [{"newline": "", "encoding": "utf-8"}]


def test_read_rater_labels_keeps_a_newline_inside_a_quoted_sentence(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "wrapped.csv", 'sentence,llm_label\n"A field,\nthen a road.",yes\n')

    assert read_rater_labels(path, "gpt", "llm_label") == {"A field,\nthen a road.": "yes"}


CONTEXT_CSV = "sentence,label,source,region\nA field.,yes,wikipedia,fiji\nA road.,no,website,peru\n"


def test_read_sentence_context_keys_the_requested_columns_by_sentence(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "human.csv", CONTEXT_CSV)

    assert read_sentence_context(path, ("source", "region")) == {
        "A field.": {"source": "wikipedia", "region": "fiji"},
        "A road.": {"source": "website", "region": "peru"},
    }


def test_read_sentence_context_rejects_a_missing_context_column(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "human.csv", CONTEXT_CSV)

    with pytest.raises(InterraterDataError, match=rf"^{re.escape(str(path))} is missing column 'place'$"):
        read_sentence_context(path, ("place",))


def test_read_sentence_context_rejects_a_duplicated_sentence(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "dupe.csv", "sentence,source\nA field.,wikipedia\nA field.,website\n")

    with pytest.raises(InterraterDataError, match=rf"^{re.escape(str(path))} has a duplicate sentence: "):
        read_sentence_context(path, ("source",))


def test_read_sentence_context_rejects_an_empty_file(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "empty.csv", "")

    with pytest.raises(InterraterDataError, match=rf"^{re.escape(str(path))} is missing column 'sentence'$"):
        read_sentence_context(path, ("source",))


def test_read_sentence_context_rejects_a_file_without_rows(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "header.csv", "sentence,source\n")

    with pytest.raises(InterraterDataError, match=rf"^{re.escape(str(path))} has no rows$"):
        read_sentence_context(path, ("source",))


def test_read_table_returns_the_original_header_and_row_order(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "human.csv", CONTEXT_CSV)

    header, rows = read_table(path)

    assert header == ("sentence", "label", "source", "region")
    assert [row["sentence"] for row in rows] == ["A field.", "A road."]
    assert rows[0] == {"sentence": "A field.", "label": "yes", "source": "wikipedia", "region": "fiji"}


def test_read_table_rejects_a_file_without_the_sentence_column(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "bad.csv", "text,label\nA field.,yes\n")

    with pytest.raises(InterraterDataError, match=rf"^{re.escape(str(path))} is missing column 'sentence'$"):
        read_table(path)


def test_read_table_rejects_a_truncated_row(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "short.csv", "sentence,label\nA field.\n")

    with pytest.raises(
        InterraterDataError,
        match=rf"^{re.escape(str(path))} has a row missing one of \['sentence', 'label'\]$",
    ):
        read_table(path)


def test_read_table_opens_the_file_as_utf8_without_newline_translation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_csv(tmp_path / "human.csv", CONTEXT_CSV)
    recorded: list[dict[str, Any]] = []
    original_open = Path.open

    def record_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        recorded.append(dict(kwargs))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record_open)

    read_table(path)

    assert recorded == [{"newline": "", "encoding": "utf-8"}]


def test_read_table_rejects_a_file_without_rows(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "header.csv", "sentence,label\n")

    with pytest.raises(InterraterDataError, match=rf"^{re.escape(str(path))} has no rows$"):
        read_table(path)
