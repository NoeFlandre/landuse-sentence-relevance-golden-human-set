import logging
from collections.abc import Iterable

import pytest

from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.domain.sentence_selection import SentencePart
from landuse_sentence_relevance.sources.website import (
    WebsiteCandidateSource,
    _select_batch_parts,
    _select_discovery_rows,
)


class FakeSplitter:
    def split(self, text: str) -> Iterable[str]:
        return text.split("|")


class RecordingSplitter(FakeSplitter):
    def __init__(self) -> None:
        self.texts: list[str] = []

    def split(self, text: str) -> Iterable[str]:
        self.texts.append(text)
        return super().split(text)


class BatchRecordingSplitter(RecordingSplitter):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls: list[tuple[str, ...]] = []

    def split_many(self, texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        self.batch_calls.append(texts)
        return tuple(tuple(text.split("|")) for text in texts)


class CountingRows:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.seen = 0

    def __iter__(self):
        for row in self._rows:
            self.seen += 1
            yield row


def _unexpected_row_load() -> Iterable[dict[str, object]]:
    raise AssertionError("the bounded discovery rows should be reusable")
    yield {}


class FakeLanguageIdentifier:
    def is_english(self, text: str) -> bool:
        return not text.startswith("NONEN")


class BatchRecordingLanguageIdentifier(FakeLanguageIdentifier):
    def __init__(self) -> None:
        self.batch_calls: list[tuple[str, ...]] = []

    def is_english_many(self, texts: Iterable[str]) -> tuple[bool, ...]:
        text_items = tuple(texts)
        self.batch_calls.append(text_items)
        return tuple(self.is_english(text) for text in text_items)


class StagedLanguageIdentifier:
    def __init__(self) -> None:
        self.batch_calls: list[tuple[str, ...]] = []

    def is_english(self, text: str) -> bool:
        return text.startswith("EN")

    def is_english_many(self, texts: Iterable[str]) -> tuple[bool, ...]:
        text_items = tuple(texts)
        self.batch_calls.append(text_items)
        return tuple(self.is_english(text) for text in text_items)


def test_website_source_declares_text_and_url_fields_as_pairs() -> None:
    assert WebsiteCandidateSource._FIELD_SPECS == (
        ("website_text", "website"),
        ("contact_website_text", "contact_website"),
    )


def test_website_source_prefers_a_sentence_inside_the_text_block() -> None:
    source = WebsiteCandidateSource(
        row_loader=lambda: [
            {
                "polygon_id": "p1",
                "lat": 45.0,
                "lon": 2.0,
                "website_text": "Heading-like title.|The place contains a broad wetland.",
            }
        ],
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1",
        max_candidates_per_cell=1,
    )

    candidates = list(source.iter_candidates())

    assert [candidate.sentence for candidate in candidates] == ["The place contains a broad wetland."]


def test_website_source_can_reject_title_only_rows_for_v2() -> None:
    splitter = BatchRecordingSplitter()
    source = WebsiteCandidateSource(
        row_loader=lambda: [
            {
                "polygon_id": "title",
                "lat": 45.0,
                "lon": 2.0,
                "website_text": "Human Verification",
            },
            {
                "polygon_id": "paragraph",
                "lat": 46.0,
                "lon": 3.0,
                "website_text": "The place contains a wetland. | It lies in a broad valley.",
            },
        ],
        splitter=splitter,
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: f"cell-{latitude}",
        max_candidates_per_cell=1,
        minimum_candidate_cells=1,
        require_paragraph=True,
    )

    candidates = list(source.iter_candidates())

    assert [candidate.sentence for candidate in candidates] == ["It lies in a broad valley."]
    assert splitter.batch_calls == [("The place contains a wetland. | It lies in a broad valley.",)]


def test_website_source_uses_both_website_text_fields(caplog) -> None:
    rows = [
        {
            "polygon_id": "p1",
            "has_any_website": True,
            "lat": 45.0,
            "lon": 2.0,
            "name": "Place",
            "region": "Region",
            "website": "https://example.test",
            "contact_website": "https://contact.example.test",
            "website_text": "Website sentence.",
            "contact_website_text": "Contact sentence.",
        },
    ]

    source = WebsiteCandidateSource(
        row_loader=lambda: rows,
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1",
    )

    caplog.set_level(logging.INFO)
    candidates = list(source.iter_candidates())

    assert [candidate.source_field for candidate in candidates] == ["website_text", "contact_website_text"]
    assert [candidate.source_url for candidate in candidates] == [
        "https://example.test",
        "https://contact.example.test",
    ]
    assert all(candidate.source is Source.WEBSITE for candidate in candidates)
    assert "Website source: streamed 1 rows; yielded 2 English candidates" in caplog.text


def test_website_source_batches_texts_across_rows_when_splitter_supports_it() -> None:
    rows = [
        {"polygon_id": "p1", "lat": 45.0, "lon": 2.0, "website_text": "First."},
        {"polygon_id": "p2", "lat": 46.0, "lon": 3.0, "website_text": "Second."},
    ]
    splitter = BatchRecordingSplitter()
    source = WebsiteCandidateSource(
        row_loader=lambda: rows,
        splitter=splitter,
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: f"cell-{latitude}",
        max_candidates_per_cell=1,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["First.", "Second."]
    assert splitter.batch_calls == [("First.", "Second.")]
    assert splitter.texts == []


def test_website_source_batches_language_checks_across_rows() -> None:
    rows = [
        {
            "polygon_id": "p1",
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "Heading one.|The place contains a wetland.",
        },
        {
            "polygon_id": "p2",
            "lat": 46.0,
            "lon": 3.0,
            "website_text": "Heading two.|The place contains a forest.",
        },
    ]
    language_identifier = BatchRecordingLanguageIdentifier()
    source = WebsiteCandidateSource(
        row_loader=lambda: rows,
        splitter=BatchRecordingSplitter(),
        language_identifier=language_identifier,
        cell_for_location=lambda latitude, longitude: f"cell-{latitude}",
        max_candidates_per_cell=1,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == [
        "The place contains a wetland.",
        "The place contains a forest.",
    ]
    assert language_identifier.batch_calls


def test_batch_language_selection_stops_after_the_first_accepted_sentence() -> None:
    language_identifier = StagedLanguageIdentifier()
    first_group = tuple(SentencePart(index, f"EN first {index}") for index in range(8))
    second_group = (
        *(SentencePart(index, f"NONEN {index}") for index in range(8)),
        SentencePart(8, "EN fallback"),
    )

    selected = _select_batch_parts((first_group, second_group), language_identifier)

    assert selected == (first_group[0], second_group[-1])
    assert language_identifier.batch_calls == [
        tuple(part.text for part in first_group) + tuple(part.text for part in second_group[:8]),
        (second_group[-1].text,),
    ]


def test_website_source_does_not_split_later_fields_after_a_batch_candidate() -> None:
    rows = [
        {
            "polygon_id": "p1",
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "First.",
            "contact_website_text": "Unused contact.",
        },
        {
            "polygon_id": "p2",
            "lat": 46.0,
            "lon": 3.0,
            "website_text": "Second.",
            "contact_website_text": "Unused contact.",
        },
    ]
    splitter = BatchRecordingSplitter()
    source = WebsiteCandidateSource(
        row_loader=lambda: rows,
        splitter=splitter,
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: f"cell-{latitude}",
        max_candidates_per_cell=1,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["First.", "Second."]
    assert splitter.batch_calls == [("First.", "Second.")]


def test_website_source_batches_one_row_per_cell_and_keeps_fallback_rows() -> None:
    rows = [
        {
            "polygon_id": "p1",
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "NONEN first.",
        },
        {
            "polygon_id": "p2",
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "Fallback.",
        },
        {
            "polygon_id": "p3",
            "lat": 46.0,
            "lon": 3.0,
            "website_text": "Second.",
        },
    ]
    splitter = BatchRecordingSplitter()
    source = WebsiteCandidateSource(
        row_loader=lambda: rows,
        splitter=splitter,
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: f"cell-{latitude}",
        max_candidates_per_cell=1,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["Fallback.", "Second."]
    assert splitter.batch_calls == [("NONEN first.", "Fallback.", "Second.")]


def test_website_source_batches_all_fields_when_a_cell_allows_multiple_candidates() -> None:
    row = {
        "polygon_id": "p1",
        "lat": 45.0,
        "lon": 2.0,
        "website_text": "Website sentence.",
        "contact_website_text": "Contact sentence.",
    }
    splitter = BatchRecordingSplitter()
    source = WebsiteCandidateSource(
        row_loader=lambda: [row],
        splitter=splitter,
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1",
        max_candidates_per_cell=2,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == [
        "Website sentence.",
        "Contact sentence.",
    ]
    assert splitter.batch_calls == [("Website sentence.", "Contact sentence.")]


def test_website_source_does_not_split_more_fields_after_cell_quota() -> None:
    rows = [
        {
            "polygon_id": "p1",
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "First.",
            "contact_website_text": "Second.",
        },
    ]
    splitter = RecordingSplitter()
    source = WebsiteCandidateSource(
        row_loader=lambda: rows,
        splitter=splitter,
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1",
        max_candidates_per_cell=1,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["First."]
    assert splitter.texts == ["First."]


def test_website_source_bounds_text_before_splitting() -> None:
    text = "A" * 100
    splitter = RecordingSplitter()
    source = WebsiteCandidateSource(
        row_loader=lambda: [
            {"polygon_id": "p1", "lat": 45.0, "lon": 2.0, "website_text": text},
        ],
        splitter=splitter,
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1",
        max_text_characters=20,
    )

    list(source.iter_candidates())

    assert splitter.texts == [text[:20]]


def test_website_source_excludes_cells_from_discovery() -> None:
    rows = [
        {"polygon_id": "p1", "lat": 45.0, "lon": 2.0, "website_text": "Excluded."},
        {"polygon_id": "p2", "lat": 46.0, "lon": 3.0, "website_text": "Included."},
    ]
    source = WebsiteCandidateSource(
        row_loader=lambda: rows,
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1" if latitude == 45.0 else "cell-2",
        candidate_cell_count=1,
        center_of_cell=lambda cell: (0.0, 0.0),
        excluded_cells={"cell-1"},
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["Included."]


def test_website_source_rejects_non_english_sentences() -> None:
    rows = [
        {
            "polygon_id": "p1",
            "has_any_website": True,
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "NONEN sentence.|English sentence.",
            "contact_website_text": None,
        },
    ]
    source = WebsiteCandidateSource(
        row_loader=lambda: rows,
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1",
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["English sentence."]


def test_website_source_can_reuse_the_wikipedia_candidate_cells() -> None:
    rows = [
        {"polygon_id": "p1", "lat": 45.0, "lon": 2.0, "website_text": "Keep this."},
        {"polygon_id": "p2", "lat": 46.0, "lon": 3.0, "website_text": "Skip this."},
    ]
    source = WebsiteCandidateSource(
        row_loader=lambda: rows,
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1" if latitude == 45.0 else "cell-2",
        allowed_cells={"cell-1"},
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["Keep this."]


def test_website_source_discovers_cells_before_processing_selected_cells() -> None:
    rows = [
        {"website_text": "Missing location."},
        {"polygon_id": "p0", "lat": 44.0, "lon": 1.0},
        {"polygon_id": "p1", "lat": 45.0, "lon": 2.0, "website_text": "First."},
        {"polygon_id": "p2", "lat": 46.0, "lon": 3.0, "website_text": "Second."},
        {"polygon_id": "p3", "lat": 47.0, "lon": 4.0, "website_text": "Third."},
    ]
    calls = 0

    def row_loader() -> Iterable[dict[str, object]]:
        nonlocal calls
        calls += 1
        yield from rows

    cells = {44.0: "cell-a", 45.0: "cell-a", 46.0: "cell-b", 47.0: "cell-c"}
    source = WebsiteCandidateSource(
        row_loader=row_loader,
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: cells[latitude],
        candidate_cell_count=2,
        center_of_cell=lambda cell: {"cell-a": (0.0, 0.0), "cell-b": (0.0, 30.0), "cell-c": (0.0, 60.0)}[
            cell
        ],
        seed="test",
        max_candidates_per_cell=1,
        minimum_candidate_cells=2,
        minimum_candidates_per_cell=1,
    )

    candidates = list(source.iter_candidates())

    assert calls == 2
    assert len(candidates) == 2
    assert len({candidate.h3_cell for candidate in candidates}) == 2


def test_website_source_selects_cells_after_english_filtering() -> None:
    rows = [
        {
            "polygon_id": "p45",
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "NONEN cell 45. NONEN details.",
        },
        {
            "polygon_id": "p46",
            "lat": 46.0,
            "lon": 3.0,
            "website_text": "English cell 46. English details.",
        },
        {
            "polygon_id": "p47",
            "lat": 47.0,
            "lon": 4.0,
            "website_text": "NONEN cell 47. NONEN details.",
        },
    ]
    source = WebsiteCandidateSource(
        row_loader=_unexpected_row_load,
        row_shards_loader=lambda: (rows,),
        splitter=BatchRecordingSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: f"cell-{int(latitude)}",
        candidate_cell_count=1,
        center_of_cell=lambda cell: (0.0, float(cell.removeprefix("cell-"))),
        seed="test",
        max_candidates_per_cell=1,
        minimum_candidate_cells=1,
        minimum_candidates_per_cell=1,
        max_rows_per_cell=2,
        max_discovery_rows_per_shard=4,
        require_paragraph=True,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == [
        "English cell 46. English details."
    ]


def test_website_source_limits_discovery_prefilter_to_a_spread_oversample() -> None:
    rows = [
        {
            "polygon_id": f"p{index}",
            "lat": float(index),
            "lon": float(index),
            "website_text": f"Cell {index}. Cell {index} details.",
        }
        for index in range(10)
    ]
    splitter = BatchRecordingSplitter()
    source = WebsiteCandidateSource(
        row_loader=_unexpected_row_load,
        row_shards_loader=lambda: (rows,),
        splitter=splitter,
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: f"cell-{int(latitude)}",
        candidate_cell_count=3,
        center_of_cell=lambda cell: (float(cell.removeprefix("cell-")), 0.0),
        seed="test",
        max_candidates_per_cell=1,
        minimum_candidate_cells=3,
        minimum_candidates_per_cell=1,
        max_rows_per_cell=1,
        max_discovery_rows_per_shard=10,
        require_paragraph=True,
    )

    candidates = list(source.iter_candidates())

    assert len(candidates) == 3
    assert [len(batch) for batch in splitter.batch_calls] == [6]


def test_website_source_prefilter_stops_after_first_ranked_row_round() -> None:
    rows = [
        {
            "polygon_id": f"p{cell}-{row_index}",
            "lat": float(cell),
            "lon": float(cell),
            "website_text": (
                f"EN cell {cell} has a wetland. EN cell {cell} spans a broad valley."
                if row_index == 0
                else f"NONEN cell {cell} title. NONEN cell {cell} details."
            ),
        }
        for cell in range(4)
        for row_index in range(8)
    ]
    splitter = BatchRecordingSplitter()
    source = WebsiteCandidateSource(
        row_loader=_unexpected_row_load,
        row_shards_loader=lambda: (rows,),
        splitter=splitter,
        language_identifier=StagedLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: f"cell-{int(latitude)}",
        candidate_cell_count=2,
        center_of_cell=lambda cell: (float(cell.removeprefix("cell-")), 0.0),
        seed="test",
        max_candidates_per_cell=1,
        minimum_candidate_cells=2,
        minimum_candidates_per_cell=1,
        max_rows_per_cell=8,
        max_discovery_rows_per_shard=32,
        require_paragraph=True,
    )

    candidates = list(source.iter_candidates())

    assert len(candidates) == 2
    assert [len(batch) for batch in splitter.batch_calls] == [4]


def test_website_source_prefilter_uses_later_rows_only_for_unfilled_cells() -> None:
    rows = [
        {
            "polygon_id": f"p{cell}-{row_index}",
            "lat": float(cell),
            "lon": float(cell),
            "website_text": (
                f"NONEN cell {cell} title. NONEN cell {cell} details. NONEN cell {cell} context."
                if row_index == 0
                else f"EN cell {cell} has a wetland. EN cell {cell} spans a broad valley."
            ),
        }
        for cell in range(4)
        for row_index in range(2)
    ]
    splitter = BatchRecordingSplitter()
    source = WebsiteCandidateSource(
        row_loader=_unexpected_row_load,
        row_shards_loader=lambda: (rows,),
        splitter=splitter,
        language_identifier=StagedLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: f"cell-{int(latitude)}",
        candidate_cell_count=2,
        center_of_cell=lambda cell: (float(cell.removeprefix("cell-")), 0.0),
        seed="test",
        max_candidates_per_cell=1,
        minimum_candidate_cells=2,
        minimum_candidates_per_cell=1,
        max_rows_per_cell=2,
        max_discovery_rows_per_shard=8,
        require_paragraph=True,
    )

    candidates = list(source.iter_candidates())

    assert len(candidates) == 2
    assert [len(batch) for batch in splitter.batch_calls] == [4, 4]


def test_website_source_reuses_bounded_rows_from_cell_discovery() -> None:
    rows = [
        {"polygon_id": "p1", "lat": 45.0, "lon": 2.0, "website_text": "First."},
        {"polygon_id": "p2", "lat": 46.0, "lon": 3.0, "website_text": "Second."},
        {"polygon_id": "p3", "lat": 47.0, "lon": 4.0, "website_text": "Third."},
    ]
    calls = 0

    def row_loader() -> Iterable[dict[str, object]]:
        nonlocal calls
        calls += 1
        yield from rows

    source = WebsiteCandidateSource(
        row_loader=row_loader,
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: f"cell-{latitude}",
        candidate_cell_count=2,
        center_of_cell=lambda cell: (0.0, float(cell.removeprefix("cell-"))),
        max_candidates_per_cell=1,
        minimum_candidate_cells=2,
        minimum_candidates_per_cell=1,
        max_rows_per_cell=2,
    )

    candidates = list(source.iter_candidates())

    assert len(candidates) == 2
    assert calls == 1


def test_website_source_prefers_the_richer_discovery_row_per_cell() -> None:
    rows = [
        {
            "polygon_id": "p1",
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "NONEN heading.",
        },
        {
            "polygon_id": "p2",
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "Paragraph heading.|The place contains a wetland.",
        },
    ]
    splitter = BatchRecordingSplitter()
    source = WebsiteCandidateSource(
        row_loader=_unexpected_row_load,
        row_shards_loader=lambda: (rows,),
        splitter=splitter,
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1",
        candidate_cell_count=1,
        center_of_cell=lambda cell: (0.0, 0.0),
        max_candidates_per_cell=1,
        minimum_candidate_cells=1,
        minimum_candidates_per_cell=1,
        max_rows_per_cell=2,
        max_discovery_rows_per_shard=2,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["The place contains a wetland."]
    assert splitter.batch_calls == [
        ("Paragraph heading.|The place contains a wetland.", "NONEN heading."),
    ]


def test_website_source_reuses_fallback_discovery_rows_without_full_rescan() -> None:
    rows = [
        {
            "polygon_id": "p1",
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "NONEN heading. Another sentence.",
        },
        {
            "polygon_id": "p2",
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "Fallback paragraph.",
        },
    ]
    splitter = BatchRecordingSplitter()
    source = WebsiteCandidateSource(
        row_loader=_unexpected_row_load,
        row_shards_loader=lambda: (rows,),
        splitter=splitter,
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1",
        candidate_cell_count=1,
        center_of_cell=lambda cell: (0.0, 0.0),
        max_candidates_per_cell=1,
        minimum_candidate_cells=1,
        minimum_candidates_per_cell=1,
        max_rows_per_cell=2,
        max_discovery_rows_per_shard=2,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["Fallback paragraph."]
    assert splitter.batch_calls == [
        ("NONEN heading. Another sentence.", "Fallback paragraph."),
    ]


def test_website_source_does_not_rescan_when_discovery_rows_exhaust_cell_quota() -> None:
    rows = [
        {
            "polygon_id": "p1",
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "NONEN first.",
        },
        {
            "polygon_id": "p2",
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "NONEN second.",
        },
    ]
    source = WebsiteCandidateSource(
        row_loader=_unexpected_row_load,
        row_shards_loader=lambda: (rows,),
        splitter=BatchRecordingSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1",
        candidate_cell_count=1,
        center_of_cell=lambda cell: (0.0, 0.0),
        max_candidates_per_cell=1,
        minimum_candidate_cells=1,
        minimum_candidates_per_cell=1,
        max_rows_per_cell=2,
        max_discovery_rows_per_shard=2,
    )

    assert list(source.iter_candidates()) == []


def test_website_source_does_not_rescan_after_discovery_stream_ends() -> None:
    rows = [
        {
            "polygon_id": "p1",
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "NONEN only row.",
        }
    ]
    source = WebsiteCandidateSource(
        row_loader=_unexpected_row_load,
        row_shards_loader=lambda: (rows,),
        splitter=BatchRecordingSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1",
        candidate_cell_count=1,
        center_of_cell=lambda cell: (0.0, 0.0),
        max_candidates_per_cell=1,
        minimum_candidate_cells=1,
        minimum_candidates_per_cell=1,
        max_rows_per_cell=2,
        max_discovery_rows_per_shard=2,
    )

    assert list(source.iter_candidates()) == []


def test_discovery_rows_are_interleaved_across_cells_for_batch_progress() -> None:
    rows = (
        ("cell-1", {"polygon_id": "cell-1-first"}),
        ("cell-1", {"polygon_id": "cell-1-second"}),
        ("cell-2", {"polygon_id": "cell-2-first"}),
        ("cell-2", {"polygon_id": "cell-2-second"}),
    )

    selected = _select_discovery_rows(rows, max_rows_per_cell=2)

    assert [row["polygon_id"] for row in selected] == [
        "cell-1-first",
        "cell-2-first",
        "cell-1-second",
        "cell-2-second",
    ]


def test_website_source_bounds_each_remote_discovery_shard() -> None:
    rows = CountingRows(
        [
            {"polygon_id": "p1", "lat": 45.0, "lon": 2.0, "website_text": "First."},
            {"polygon_id": "p2", "lat": 46.0, "lon": 3.0, "website_text": "Unused."},
        ]
    )
    source = WebsiteCandidateSource(
        row_loader=_unexpected_row_load,
        row_shards_loader=lambda: (rows,),
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: f"cell-{latitude}",
        candidate_cell_count=1,
        center_of_cell=lambda cell: (0.0, 0.0),
        max_candidates_per_cell=1,
        minimum_candidate_cells=1,
        minimum_candidates_per_cell=1,
        max_rows_per_cell=1,
        max_discovery_rows_per_shard=1,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["First."]
    assert rows.seen == 1


def test_website_source_merges_discovery_shards_in_stable_order() -> None:
    shards = (
        [{"polygon_id": "p1", "lat": 45.0, "lon": 2.0, "website_text": "First."}],
        [{"polygon_id": "p2", "lat": 46.0, "lon": 3.0, "website_text": "Second."}],
    )
    source = WebsiteCandidateSource(
        row_loader=_unexpected_row_load,
        row_shards_loader=lambda: shards,
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: f"cell-{latitude}",
        candidate_cell_count=2,
        center_of_cell=lambda cell: (float(cell.removeprefix("cell-")), 0.0),
        max_candidates_per_cell=1,
        minimum_candidate_cells=2,
        minimum_candidates_per_cell=1,
        max_rows_per_cell=1,
        max_discovery_rows_per_shard=1,
        max_stream_workers=2,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["First.", "Second."]


def test_website_source_stops_after_enough_full_candidate_cells() -> None:
    def rows() -> Iterable[dict[str, object]]:
        yield {
            "polygon_id": "p1",
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "Cell one title.|Cell one.|Cell one extra.",
        }
        yield {
            "polygon_id": "p2",
            "lat": 46.0,
            "lon": 3.0,
            "website_text": "Cell two title.|Cell two.|Cell two extra.",
        }
        raise AssertionError("the website stream should stop after the cell budget is filled")

    source = WebsiteCandidateSource(
        row_loader=rows,
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1" if latitude == 45.0 else "cell-2",
        max_candidates_per_cell=8,
        minimum_candidate_cells=2,
        minimum_candidates_per_cell=2,
    )

    assert {candidate.sentence for candidate in source.iter_candidates()} == {
        "Cell one.",
        "Cell one extra.",
        "Cell two.",
        "Cell two extra.",
    }


def test_website_source_keeps_scanning_until_candidate_cell_budget_is_met() -> None:
    rows = [
        {"polygon_id": "p1", "lat": 45.0, "lon": 2.0, "website_text": "NONEN first."},
        {"polygon_id": "p2", "lat": 46.0, "lon": 3.0, "website_text": "Second."},
        {"polygon_id": "p3", "lat": 47.0, "lon": 4.0, "website_text": "Third."},
    ]
    source = WebsiteCandidateSource(
        row_loader=lambda: rows,
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: f"cell-{latitude}",
        max_candidates_per_cell=1,
        minimum_candidate_cells=2,
        max_rows_per_cell=1,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["Second.", "Third."]


def test_website_source_bounds_splitter_work_per_cell() -> None:
    def rows() -> Iterable[dict[str, object]]:
        yield {"polygon_id": "p1", "lat": 45.0, "lon": 2.0, "website_text": "First."}
        yield {"polygon_id": "p2", "lat": 45.0, "lon": 2.0, "website_text": "Second."}
        yield {"polygon_id": "p3", "lat": 45.0, "lon": 2.0, "website_text": "Skip this."}
        yield {"polygon_id": "p4", "lat": 46.0, "lon": 3.0, "website_text": "Third."}
        yield {"polygon_id": "p5", "lat": 46.0, "lon": 3.0, "website_text": "Fourth."}
        raise AssertionError("the bounded website stream should stop before the next row")

    splitter = RecordingSplitter()
    source = WebsiteCandidateSource(
        row_loader=rows,
        splitter=splitter,
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1" if latitude == 45.0 else "cell-2",
        minimum_candidate_cells=2,
        max_rows_per_cell=2,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == [
        "First.",
        "Second.",
        "Third.",
        "Fourth.",
    ]
    assert splitter.texts == ["First.", "Second.", "Third.", "Fourth."]


def test_website_source_rejects_invalid_candidate_budget() -> None:
    with pytest.raises(ValueError, match="max_candidates_per_cell must be positive"):
        WebsiteCandidateSource(
            row_loader=lambda: [],
            splitter=FakeSplitter(),
            language_identifier=FakeLanguageIdentifier(),
            cell_for_location=lambda latitude, longitude: "cell-1",
            max_candidates_per_cell=0,
        )

    with pytest.raises(ValueError, match="minimum_candidate_cells must be positive"):
        WebsiteCandidateSource(
            row_loader=lambda: [],
            splitter=FakeSplitter(),
            language_identifier=FakeLanguageIdentifier(),
            cell_for_location=lambda latitude, longitude: "cell-1",
            minimum_candidate_cells=0,
        )

    with pytest.raises(ValueError, match="center_of_cell is required"):
        WebsiteCandidateSource(
            row_loader=lambda: [],
            splitter=FakeSplitter(),
            language_identifier=FakeLanguageIdentifier(),
            cell_for_location=lambda latitude, longitude: "cell-1",
            candidate_cell_count=1,
        )

    with pytest.raises(ValueError, match="max_rows_per_cell must be positive"):
        WebsiteCandidateSource(
            row_loader=lambda: [],
            splitter=FakeSplitter(),
            language_identifier=FakeLanguageIdentifier(),
            cell_for_location=lambda latitude, longitude: "cell-1",
            max_rows_per_cell=0,
        )

    with pytest.raises(ValueError, match="max_discovery_rows_per_shard must be positive"):
        WebsiteCandidateSource(
            row_loader=lambda: [],
            splitter=FakeSplitter(),
            language_identifier=FakeLanguageIdentifier(),
            cell_for_location=lambda latitude, longitude: "cell-1",
            max_discovery_rows_per_shard=0,
        )

    with pytest.raises(ValueError, match="max_stream_workers must be positive"):
        WebsiteCandidateSource(
            row_loader=lambda: [],
            splitter=FakeSplitter(),
            language_identifier=FakeLanguageIdentifier(),
            cell_for_location=lambda latitude, longitude: "cell-1",
            max_stream_workers=0,
        )
