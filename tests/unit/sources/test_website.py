from collections.abc import Iterable

from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.sources.website import WebsiteCandidateSource


class FakeSplitter:
    def split(self, text: str) -> Iterable[str]:
        return text.split("|")


class FakeLanguageIdentifier:
    def is_english(self, text: str) -> bool:
        return not text.startswith("NONEN")


def test_website_source_declares_text_and_url_fields_as_pairs() -> None:
    assert WebsiteCandidateSource._FIELD_SPECS == (
        ("website_text", "website"),
        ("contact_website_text", "contact_website"),
    )


def test_website_source_uses_both_website_text_fields() -> None:
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

    candidates = list(source.iter_candidates())

    assert [candidate.source_field for candidate in candidates] == ["website_text", "contact_website_text"]
    assert [candidate.source_url for candidate in candidates] == [
        "https://example.test",
        "https://contact.example.test",
    ]
    assert all(candidate.source is Source.WEBSITE for candidate in candidates)


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
