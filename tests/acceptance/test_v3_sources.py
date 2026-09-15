from __future__ import annotations

import pytest

from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.sources.v3 import (
    DescriptionSentenceSource,
    WebsiteSentenceSource,
    WikipediaSentenceSource,
)

pytestmark = pytest.mark.acceptance


def _shards(*rows: dict[str, object]):
    return lambda: (iter(rows),)


def test_v3_sources_produce_the_three_english_geolocated_candidate_streams() -> None:
    def cell_for_location(latitude: float, longitude: float) -> str:
        return "8928308280fffff"

    description = DescriptionSentenceSource(
        sentence_shards_loader=_shards(
            {
                "description_identity": "d" * 64,
                "source_pbf": "occitanie-latest.osm.pbf",
                "osm_type": "node",
                "osm_id": 1,
                "language_code": "eng",
                "top_score": 0.99,
                "sentences": ["An upstream description sentence."],
            }
        ),
        geometry_shards_loader=_shards(
            {
                "source_pbf": "occitanie-latest.osm.pbf",
                "osm_type": "node",
                "osm_id": 1,
                "lat": 45.0,
                "lon": 2.0,
            }
        ),
        cell_for_location=cell_for_location,
    )
    wikipedia = WikipediaSentenceSource(
        sentence_shards_loader=_shards(
            {
                "sentence_id": "wikipedia-context",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "section_index": 1,
                "sentence_index": 1,
                "heading": "History",
                "text": "A contextual English Wikipedia sentence.",
                "page_id": 123,
            },
            {
                "sentence_id": "wikivoyage-context",
                "wikidata": "Q1",
                "project": "wikivoyage",
                "language": "en",
                "section_index": 1,
                "sentence_index": 1,
                "heading": "History",
                "text": "A travel guide sentence.",
            },
        ),
        polygon_shards_loader=_shards(
            {
                "polygon_id": "p1",
                "wikidata": "Q1",
                "has_english_wikipedia": True,
                "lat": 45.0,
                "lon": 2.0,
                "region": "Ile-de-France",
            }
        ),
        cell_for_location=cell_for_location,
    )
    website = WebsiteSentenceSource(
        row_shards_loader=_shards(
            {
                "polygon_id": "p2",
                "lat": 46.0,
                "lon": 3.0,
                "region": "Bretagne",
                "website": "https://example.test",
                "website_language": "eng_Latn",
                "website_language_probability": 0.99,
                "website_sentences": ["An upstream website sentence."],
            }
        ),
        cell_for_location=cell_for_location,
    )

    candidates = [
        *description.iter_candidates(),
        *wikipedia.iter_candidates(),
        *website.iter_candidates(),
    ]

    assert {candidate.source for candidate in candidates} == {
        Source.DESCRIPTION,
        Source.WIKIPEDIA,
        Source.WEBSITE,
    }
    assert [candidate.sentence for candidate in candidates] == [
        "An upstream description sentence.",
        "A contextual English Wikipedia sentence.",
        "An upstream website sentence.",
    ]
    assert [candidate.language for candidate in candidates] == ["en", "en", "en"]
    assert [candidate.source_url for candidate in candidates] == [
        "https://www.openstreetmap.org/node/1",
        "https://en.wikipedia.org/?curid=123",
        "https://example.test",
    ]
    assert [candidate.region for candidate in candidates] == [
        "occitanie",
        "Ile-de-France",
        "Bretagne",
    ]


def test_v3_sources_drop_the_non_english_records_of_the_same_streams() -> None:
    """The English rule must be falsifiable: same shapes, non-English metadata."""

    def cell_for_location(latitude: float, longitude: float) -> str:
        return "8928308280fffff"

    description = DescriptionSentenceSource(
        sentence_shards_loader=_shards(
            {
                "description_identity": "d" * 64,
                "source_pbf": "region.osm.pbf",
                "osm_type": "node",
                "osm_id": 1,
                "language_code": "fra",
                "top_score": 0.99,
                "sentences": ["Une phrase de description en francais."],
            }
        ),
        geometry_shards_loader=_shards(
            {
                "source_pbf": "region.osm.pbf",
                "osm_type": "node",
                "osm_id": 1,
                "lat": 45.0,
                "lon": 2.0,
            }
        ),
        cell_for_location=cell_for_location,
    )
    wikipedia = WikipediaSentenceSource(
        sentence_shards_loader=_shards(
            {
                "sentence_id": "wikipedia-french",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "fr",
                "section_index": 1,
                "sentence_index": 1,
                "heading": "Histoire",
                "text": "Une phrase contextuelle de Wikipedia en francais.",
                "page_id": 123,
            }
        ),
        polygon_shards_loader=_shards(
            {
                "polygon_id": "p1",
                "wikidata": "Q1",
                "has_english_wikipedia": True,
                "lat": 45.0,
                "lon": 2.0,
            }
        ),
        cell_for_location=cell_for_location,
    )
    website = WebsiteSentenceSource(
        row_shards_loader=_shards(
            {
                "polygon_id": "p2",
                "lat": 46.0,
                "lon": 3.0,
                "website_language": "fra_Latn",
                "website_language_probability": 0.99,
                "website_sentences": ["Une phrase de site web en francais."],
            }
        ),
        cell_for_location=cell_for_location,
    )

    assert list(description.iter_candidates()) == []
    assert list(wikipedia.iter_candidates()) == []
    assert list(website.iter_candidates()) == []
