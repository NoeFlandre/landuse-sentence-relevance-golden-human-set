from __future__ import annotations

import logging
import threading
from collections.abc import Generator, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, ClassVar, cast

import pytest

import landuse_sentence_relevance.sources.v3 as v3_module
from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.sources.v3 import (
    _SHARD_QUEUE_CAPACITY,
    DescriptionSentenceSource,
    WebsiteSentenceSource,
    WikipediaSentenceSource,
    _join_shard_rows,
    _ordered_parallel_shards,
    _parallel_shard_rows,
    _sliding_window,
)


class CountingRows:
    """A single-pass row stream that records how many rows an adapter read."""

    def __init__(self, rows: Iterable[Mapping[str, Any]]) -> None:
        self._rows = tuple(rows)
        self.seen = 0

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        for row in self._rows:
            self.seen += 1
            yield row


def _cell(latitude: float, longitude: float) -> str:
    return "8928308280fffff"


def _description_row(
    *,
    identity: str,
    osm_id: str = "1",
    source_pbf: str = "region.osm.pbf",
    language_code: str = "eng",
    top_score: float = 0.99,
    sentence: str = "The place has a contextual description.",
) -> dict[str, Any]:
    return {
        "description_identity": identity,
        "source_pbf": source_pbf,
        "osm_type": "node",
        "osm_id": osm_id,
        "language_code": language_code,
        "top_score": top_score,
        "sentences": [sentence],
    }


def _geometry_row(
    *,
    osm_id: str = "1",
    source_pbf: str = "region.osm.pbf",
    name: str = "Place",
) -> dict[str, Any]:
    """Shape a geometry row the way the recorded upstream schema publishes it.

    The pinned revision has no ``region`` column, so none is set here.
    """

    return {
        "source_pbf": source_pbf,
        "osm_type": "node",
        "osm_id": osm_id,
        "lat": 45.0,
        "lon": 2.0,
        "name": name,
        "osm_url": f"https://www.openstreetmap.org/way/{osm_id}",
    }


def _wikipedia_row(
    *,
    sentence_id: str,
    wikidata: str = "Q1",
    project: str = "wikipedia",
    language: str = "en",
    section_index: int = 3,
    sentence_index: int = 2,
    heading: str = "History",
    text: str = "The place has a contextual section sentence.",
    **overrides: Any,
) -> dict[str, Any]:
    """Shape a sentence row the way the recorded upstream schema publishes it.

    The pinned revision has no ``is_lead`` or ``is_title`` column: the lead is
    ``section_index`` 0 with an empty heading, and a section title arrives as the
    first sentence of a body section carrying the MediaWiki edit marker.
    """

    return {
        "sentence_id": sentence_id,
        "wikidata": wikidata,
        "project": project,
        "language": language,
        "section_index": section_index,
        "sentence_index": sentence_index,
        "heading": heading,
        "text": text,
        "page_id": 123,
        **overrides,
    }


def _lead_row(*, sentence_id: str, sentence_index: int = 0, **overrides: Any) -> dict[str, Any]:
    return _wikipedia_row(
        sentence_id=sentence_id,
        section_index=0,
        sentence_index=sentence_index,
        heading="",
        **overrides,
    )


def _polygon_row(
    *,
    polygon_id: str = "p1",
    wikidata: str = "Q1",
    **overrides: Any,
) -> dict[str, Any]:
    return {
        "polygon_id": polygon_id,
        "wikidata": wikidata,
        "has_english_wikipedia": True,
        "lat": 45.0,
        "lon": 2.0,
        "name": "Place",
        "region": "Region",
        **overrides,
    }


def _website_row(*, polygon_id: str = "p1", **overrides: Any) -> dict[str, Any]:
    """Shape a website polygon row the way the pinned revision publishes it."""
    return {
        "polygon_id": polygon_id,
        "lat": 45.0,
        "lon": 2.0,
        "name": "Place",
        "region": "Region",
        "website": "https://example.test",
        "website_language": "eng_Latn",
        "website_language_probability": 0.95,
        "website_sentences": ["The upstream website sentence."],
        **overrides,
    }


def _description_source(sentence_shards, geometry_shards, **options: Any) -> DescriptionSentenceSource:
    return DescriptionSentenceSource(
        sentence_shards_loader=lambda: sentence_shards,
        geometry_shards_loader=lambda: geometry_shards,
        cell_for_location=_cell,
        **options,
    )


def _wikipedia_source(sentence_shards, polygon_shards, **options: Any) -> WikipediaSentenceSource:
    return WikipediaSentenceSource(
        sentence_shards_loader=lambda: sentence_shards,
        polygon_shards_loader=lambda: polygon_shards,
        cell_for_location=_cell,
        **options,
    )


def _website_source(*shards, **options: Any) -> WebsiteSentenceSource:
    return WebsiteSentenceSource(
        row_shards_loader=lambda: shards,
        cell_for_location=_cell,
        **options,
    )


def test_description_adapter_uses_upstream_english_sentence_records() -> None:
    sentence_rows = CountingRows(
        [
            _description_row(identity="a" * 64),
            _description_row(identity="b" * 64, osm_id="2", language_code="fra"),
            _description_row(identity="c" * 64, osm_id="3", top_score=0.89),
        ]
    )
    geometry_rows = CountingRows(
        [_geometry_row(osm_id="1"), _geometry_row(osm_id="2"), _geometry_row(osm_id="3")]
    )
    source = _description_source(
        (sentence_rows,), (geometry_rows,), max_rows_per_shard=10, min_language_score=0.90
    )

    candidates = list(source.iter_candidates())

    assert [candidate.sentence for candidate in candidates] == ["The place has a contextual description."]
    assert candidates[0].source is Source.DESCRIPTION
    assert candidates[0].source_record_id == "a" * 64
    assert candidates[0].source_field == "description"


def test_description_adapter_preserves_the_upstream_geometry_provenance() -> None:
    source = _description_source(
        (iter([_description_row(identity="a" * 64, source_pbf="ile-de-france-latest.osm.pbf")]),),
        (iter([_geometry_row(name="Bois de Vincennes", source_pbf="ile-de-france-latest.osm.pbf")]),),
    )

    candidate = next(source.iter_candidates())

    assert candidate.place_name == "Bois de Vincennes"
    assert candidate.region == "ile-de-france"
    assert candidate.source_url == "https://www.openstreetmap.org/way/1"


def test_description_adapter_derives_an_osm_url_when_the_geometry_has_none() -> None:
    geometry = _geometry_row(osm_id="42")
    del geometry["osm_url"]
    source = _description_source(
        (iter([_description_row(identity="a" * 64, osm_id="42")]),),
        (iter([geometry]),),
    )

    assert next(source.iter_candidates()).source_url == "https://www.openstreetmap.org/node/42"


def test_description_adapter_joins_a_sentence_to_geometry_in_a_later_shard() -> None:
    sentence_shards = (
        CountingRows([_description_row(identity="a" * 64, osm_id="7")]),
        CountingRows([_description_row(identity="b" * 64, osm_id="8")]),
    )
    geometry_shards = (
        CountingRows([_geometry_row(osm_id="8")]),
        CountingRows([_geometry_row(osm_id="7")]),
    )

    candidates = list(_description_source(sentence_shards, geometry_shards).iter_candidates())

    assert [candidate.source_record_id for candidate in candidates] == ["a" * 64, "b" * 64]


def test_description_adapter_reads_sentence_shards_the_geometry_side_does_not_pair_with() -> None:
    sentence_shards = (
        CountingRows([_description_row(identity="a" * 64, osm_id="1")]),
        CountingRows([_description_row(identity="b" * 64, osm_id="2")]),
        CountingRows([_description_row(identity="c" * 64, osm_id="3")]),
    )
    geometry_shards = (CountingRows([_geometry_row(osm_id="3")]),)

    candidates = list(_description_source(sentence_shards, geometry_shards).iter_candidates())

    assert [candidate.source_record_id for candidate in candidates] == ["c" * 64]


def test_description_adapter_bounds_each_side_of_the_geometry_join() -> None:
    sentence_rows = CountingRows(
        [_description_row(identity="a" * 64), _description_row(identity="b" * 64, osm_id="2")]
    )
    geometry_rows = CountingRows([_geometry_row(osm_id="1"), _geometry_row(osm_id="2")])
    source = _description_source((sentence_rows,), (geometry_rows,), max_rows_per_shard=1)

    assert len(list(source.iter_candidates())) == 1
    assert sentence_rows.seen == 1
    assert geometry_rows.seen == 1


def test_description_adapter_stops_indexing_geometry_at_the_configured_join_bound() -> None:
    sentence_rows = CountingRows(
        [_description_row(identity="a" * 64, osm_id="1"), _description_row(identity="b" * 64, osm_id="2")]
    )
    geometry_shards = (
        CountingRows([_geometry_row(osm_id="1")]),
        CountingRows([_geometry_row(osm_id="2")]),
    )
    source = _description_source((sentence_rows,), geometry_shards, max_join_entries=1)

    candidates = list(source.iter_candidates())

    assert [candidate.source_record_id for candidate in candidates] == ["a" * 64]
    assert geometry_shards[1].seen == 0


def test_description_adapter_can_resolve_coordinates_from_an_upstream_bbox() -> None:
    geometry = {
        **_geometry_row(),
        "lat": None,
        "lon": None,
        "bbox_min_x": 1.0,
        "bbox_min_y": 44.0,
        "bbox_max_x": 3.0,
        "bbox_max_y": 46.0,
    }
    source = _description_source(
        (iter([_description_row(identity="a" * 64)]),),
        (iter([geometry]),),
    )

    candidate = next(source.iter_candidates())

    assert (candidate.latitude, candidate.longitude) == (45.0, 2.0)


def test_wikipedia_adapter_keeps_contextual_english_wikipedia_only() -> None:
    sentence_rows = CountingRows(
        [
            _lead_row(sentence_id="lead", text="A lead sentence."),
            _wikipedia_row(sentence_id="title", sentence_index=0, text="[ edit | edit source ]"),
            _wikipedia_row(sentence_id="good"),
            _wikipedia_row(sentence_id="voyage", project="wikivoyage", text="Travel guide text."),
            _wikipedia_row(sentence_id="french", language="fr", text="Texte francais."),
        ]
    )
    source = _wikipedia_source((sentence_rows,), (CountingRows([_polygon_row()]),), max_rows_per_shard=10)

    candidates = list(source.iter_candidates())

    assert [candidate.sentence for candidate in candidates] == [
        "The place has a contextual section sentence."
    ]
    assert candidates[0].candidate_id == "wikipedia:p1:good"
    assert candidates[0].source is Source.WIKIPEDIA
    assert candidates[0].source_url == "https://en.wikipedia.org/?curid=123"


def test_wikipedia_adapter_excludes_every_sentence_of_the_lead_section() -> None:
    sentence_rows = CountingRows(
        [
            _lead_row(sentence_id="lead-first", sentence_index=0),
            _lead_row(sentence_id="lead-later", sentence_index=4),
        ]
    )
    source = _wikipedia_source((sentence_rows,), (CountingRows([_polygon_row()]),))

    assert list(source.iter_candidates()) == []


def test_wikipedia_adapter_reads_the_lead_from_the_section_index_alone() -> None:
    sentence_rows = CountingRows(
        [_wikipedia_row(sentence_id="numbered-lead", section_index=0, heading="Overview")]
    )
    source = _wikipedia_source((sentence_rows,), (CountingRows([_polygon_row()]),))

    assert list(source.iter_candidates()) == []


def test_wikipedia_adapter_treats_an_empty_heading_as_the_lead_section() -> None:
    sentence_rows = CountingRows([_wikipedia_row(sentence_id="unnumbered-lead", section_index=7, heading="")])
    source = _wikipedia_source((sentence_rows,), (CountingRows([_polygon_row()]),))

    assert list(source.iter_candidates()) == []


@pytest.mark.parametrize(
    "heading_line",
    ["[ edit ]", "[ edit | edit source ]", "[ Edit ]", "  [ edit | edit source ]  "],
)
def test_wikipedia_adapter_excludes_the_rendered_section_heading_line(heading_line: str) -> None:
    sentence_rows = CountingRows(
        [_wikipedia_row(sentence_id="section-title", sentence_index=0, text=heading_line)]
    )
    source = _wikipedia_source((sentence_rows,), (CountingRows([_polygon_row()]),))

    assert list(source.iter_candidates()) == []


def test_wikipedia_adapter_excludes_a_heading_line_that_runs_into_body_text() -> None:
    sentence_rows = CountingRows(
        [
            _wikipedia_row(
                sentence_id="title-and-text",
                sentence_index=0,
                text="[ edit ] On 6 December 2011 a bombing struck the shrine.",
            )
        ]
    )
    source = _wikipedia_source((sentence_rows,), (CountingRows([_polygon_row()]),))

    assert list(source.iter_candidates()) == []


def test_wikipedia_adapter_keeps_a_body_sentence_that_only_mentions_editing() -> None:
    sentence_rows = CountingRows(
        [
            _wikipedia_row(
                sentence_id="mentions-editing",
                sentence_index=1,
                text="The [ edit ] marker is rendered before the section body.",
            )
        ]
    )
    source = _wikipedia_source((sentence_rows,), (CountingRows([_polygon_row()]),))

    assert [candidate.candidate_id for candidate in source.iter_candidates()] == [
        "wikipedia:p1:mentions-editing"
    ]


def test_wikipedia_adapter_keeps_zero_based_first_sentence_in_a_contextual_section() -> None:
    sentence_rows = CountingRows(
        [
            _wikipedia_row(
                sentence_id="zero-based",
                section_index=1,
                sentence_index=0,
                text="The first sentence in a contextual section.",
            )
        ]
    )
    source = _wikipedia_source((sentence_rows,), (CountingRows([_polygon_row()]),))

    assert [candidate.sentence for candidate in source.iter_candidates()] == [
        "The first sentence in a contextual section."
    ]


def test_wikipedia_adapter_builds_a_sentence_identity_from_the_upstream_indexes() -> None:
    sentence_rows = CountingRows(
        [
            {
                "document_id": "d1",
                "section_id": "s1",
                "sentence_index": 1,
                "section_index": 1,
                "heading": "History",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "text": "A contextual sentence without an upstream sentence id.",
            }
        ]
    )
    source = _wikipedia_source((sentence_rows,), (CountingRows([_polygon_row()]),))

    assert [candidate.source_record_id for candidate in source.iter_candidates()] == ["d1:s1:1"]


def test_wikipedia_adapter_preserves_the_polygon_and_article_provenance() -> None:
    sentence_rows = CountingRows([_wikipedia_row(sentence_id="good")])
    polygon_rows = CountingRows([_polygon_row(name="Vincennes", region="Ile-de-France")])
    source = _wikipedia_source((sentence_rows,), (polygon_rows,))

    candidate = next(source.iter_candidates())

    assert candidate.place_name == "Vincennes"
    assert candidate.region == "Ile-de-France"
    assert candidate.source_url == "https://en.wikipedia.org/?curid=123"


def test_wikipedia_adapter_derives_the_article_url_from_the_upstream_page_id() -> None:
    sentence_rows = CountingRows([_wikipedia_row(sentence_id="good", page_id=65772030)])
    source = _wikipedia_source((sentence_rows,), (CountingRows([_polygon_row()]),))

    assert next(source.iter_candidates()).source_url == "https://en.wikipedia.org/?curid=65772030"


def test_wikipedia_adapter_emits_no_article_url_without_a_page_id() -> None:
    row = _wikipedia_row(sentence_id="good")
    del row["page_id"]
    source = _wikipedia_source((CountingRows([row]),), (CountingRows([_polygon_row()]),))

    assert next(source.iter_candidates()).source_url is None


def test_wikipedia_adapter_skips_polygons_without_an_english_article() -> None:
    sentence_rows = CountingRows([_wikipedia_row(sentence_id="good")])
    polygon_rows = CountingRows([_polygon_row(has_english_wikipedia=False)])

    assert list(_wikipedia_source((sentence_rows,), (polygon_rows,)).iter_candidates()) == []


def test_wikipedia_adapter_joins_a_sentence_to_a_polygon_in_another_shard() -> None:
    sentence_shards = (
        CountingRows([_wikipedia_row(sentence_id="cross-shard", wikidata="Q-cross")]),
        CountingRows([]),
    )
    polygon_shards = (
        CountingRows([]),
        CountingRows([_polygon_row(polygon_id="p-cross", wikidata="Q-cross")]),
    )

    candidates = list(_wikipedia_source(sentence_shards, polygon_shards).iter_candidates())

    assert [candidate.candidate_id for candidate in candidates] == ["wikipedia:p-cross:cross-shard"]


def test_wikipedia_adapter_reads_sentence_shards_beyond_the_polygon_shard_count() -> None:
    sentence_shards = (
        CountingRows([_wikipedia_row(sentence_id="first", wikidata="Q1")]),
        CountingRows([_wikipedia_row(sentence_id="second", wikidata="Q1")]),
    )
    polygon_shards = (CountingRows([_polygon_row()]),)

    candidates = list(_wikipedia_source(sentence_shards, polygon_shards).iter_candidates())

    assert [candidate.candidate_id for candidate in candidates] == [
        "wikipedia:p1:first",
        "wikipedia:p1:second",
    ]


def test_wikipedia_adapter_keeps_the_first_polygon_for_a_repeated_wikidata_key() -> None:
    sentence_shards = (CountingRows([_wikipedia_row(sentence_id="good")]),)
    polygon_shards = (
        CountingRows([_polygon_row(polygon_id="first")]),
        CountingRows([_polygon_row(polygon_id="second")]),
    )

    candidates = list(_wikipedia_source(sentence_shards, polygon_shards).iter_candidates())

    assert [candidate.candidate_id for candidate in candidates] == ["wikipedia:first:good"]


def test_wikipedia_adapter_bounds_the_polygon_lookup_and_sentence_stream() -> None:
    sentence_rows = CountingRows(
        [
            _wikipedia_row(sentence_id="good", wikidata="Q1", text="Context."),
            _wikipedia_row(sentence_id="not-read", wikidata="Q2", text="Not read."),
        ]
    )
    polygon_rows = CountingRows(
        [_polygon_row(polygon_id="p1", wikidata="Q1"), _polygon_row(polygon_id="p2", wikidata="Q2")]
    )
    source = _wikipedia_source((sentence_rows,), (polygon_rows,), max_rows_per_shard=1)

    assert len(list(source.iter_candidates())) == 1
    assert sentence_rows.seen == 1
    assert polygon_rows.seen == 1


def test_wikipedia_adapter_applies_the_configured_text_bound() -> None:
    sentence_rows = CountingRows(
        [_wikipedia_row(sentence_id="too-long", text="This is longer than the configured bound.")]
    )
    source = _wikipedia_source((sentence_rows,), (CountingRows([_polygon_row()]),), max_text_characters=10)

    assert list(source.iter_candidates()) == []


def test_website_adapter_uses_pre_split_upstream_english_metadata() -> None:
    rows = CountingRows(
        [
            {
                "polygon_id": "p1",
                "lat": 45.0,
                "lon": 2.0,
                "name": "Place",
                "region": "Region",
                "website": "https://example.test",
                "website_language": "eng_Latn",
                "website_language_probability": 0.95,
                "website_sentences": ["The upstream website sentence."],
                "contact_website": "https://example.test/contact",
                "contact_website_language": "fra_Latn",
                "contact_website_language_probability": 0.99,
                "contact_website_sentences": ["Texte francais."],
            },
            {
                "polygon_id": "p2",
                "lat": 46.0,
                "lon": 3.0,
                "website_language": "eng_Latn",
                "website_language_probability": 0.89,
                "website_sentences": ["Below the threshold."],
            },
        ]
    )
    source = _website_source(rows, min_language_probability=0.90)

    candidates = list(source.iter_candidates())

    assert [candidate.sentence for candidate in candidates] == ["The upstream website sentence."]
    assert candidates[0].source_field == "website_sentences"
    assert candidates[0].source_url == "https://example.test"
    assert candidates[0].candidate_id == "website:p1:website_sentences:0"


def test_website_adapter_keeps_the_provenance_of_each_upstream_website_field() -> None:
    rows = CountingRows(
        [
            {
                "polygon_id": "p1",
                "lat": 45.0,
                "lon": 2.0,
                "name": "Place",
                "region": "Ile-de-France",
                "website": "https://example.test",
                "website_language": "eng_Latn",
                "website_language_probability": 0.99,
                "website_sentences": ["The main site sentence."],
                "contact_website": "https://contact.example.test",
                "contact_website_language": "eng_Latn",
                "contact_website_language_probability": 0.99,
                "contact_website_sentences": ["The contact site sentence."],
            }
        ]
    )

    candidates = list(_website_source(rows).iter_candidates())

    assert [(candidate.source_field, candidate.source_url) for candidate in candidates] == [
        ("website_sentences", "https://example.test"),
        ("contact_website_sentences", "https://contact.example.test"),
    ]
    assert {candidate.region for candidate in candidates} == {"Ile-de-France"}
    assert {candidate.place_name for candidate in candidates} == {"Place"}


def test_website_adapter_rejects_unsuccessful_upstream_sentence_records() -> None:
    rows = CountingRows(
        [
            {
                "polygon_id": "p1",
                "lat": 45.0,
                "lon": 2.0,
                "website_language": "eng_Latn",
                "website_language_probability": 0.99,
                "website_sentences": ["This record is not complete."],
                "website_sentence_status": "failed",
            }
        ]
    )

    assert list(_website_source(rows).iter_candidates()) == []


def test_website_adapter_emits_the_same_candidates_on_every_pass() -> None:
    row = {
        "polygon_id": "p1",
        "lat": 45.0,
        "lon": 2.0,
        "name": "Place",
        "region": "Region",
        "website": "https://example.test",
        "website_language": "en",
        "website_language_probability": 1.0,
        "website_sentences": ["First upstream sentence.", "Second upstream sentence."],
    }

    first = [candidate.to_dict() for candidate in _website_source((row,)).iter_candidates()]
    second = [candidate.to_dict() for candidate in _website_source((row,)).iter_candidates()]

    assert [candidate["candidate_id"] for candidate in first] == [
        "website:p1:website_sentences:0",
        "website:p1:website_sentences:1",
    ]
    assert first == second


def test_v3_adapters_reject_non_positive_shard_limits() -> None:
    with pytest.raises(ValueError, match="max_rows_per_shard must be positive"):
        WebsiteSentenceSource(
            row_shards_loader=lambda: (),
            cell_for_location=_cell,
            max_rows_per_shard=0,
        )


def test_v3_joined_adapters_reject_a_non_positive_join_bound() -> None:
    with pytest.raises(ValueError, match="max_join_entries must be positive"):
        _wikipedia_source((), (), max_join_entries=0)


def test_description_adapter_names_every_bound_it_rejects() -> None:
    shards: tuple[()] = ()
    with pytest.raises(ValueError, match=r"^max_rows_per_shard must be positive$"):
        _description_source(shards, shards, max_rows_per_shard=0)
    with pytest.raises(ValueError, match=r"^max_text_characters must be positive$"):
        _description_source(shards, shards, max_text_characters=0)
    with pytest.raises(ValueError, match=r"^max_join_entries must be positive$"):
        _description_source(shards, shards, max_join_entries=0)
    with pytest.raises(ValueError, match=r"^min_language_score must be between 0 and 1$"):
        _description_source(shards, shards, min_language_score=1.5)


def test_wikipedia_adapter_names_every_bound_it_rejects() -> None:
    shards: tuple[()] = ()
    with pytest.raises(ValueError, match=r"^max_rows_per_shard must be positive$"):
        _wikipedia_source(shards, shards, max_rows_per_shard=0)
    with pytest.raises(ValueError, match=r"^max_text_characters must be positive$"):
        _wikipedia_source(shards, shards, max_text_characters=0)
    with pytest.raises(ValueError, match=r"^max_join_entries must be positive$"):
        _wikipedia_source(shards, shards, max_join_entries=0)


def test_website_adapter_names_every_bound_it_rejects() -> None:
    with pytest.raises(ValueError, match=r"^max_text_characters must be positive$"):
        _website_source(max_text_characters=0)
    with pytest.raises(ValueError, match=r"^min_language_probability must be between 0 and 1$"):
        _website_source(min_language_probability=1.5)
    with pytest.raises(ValueError, match=r"^min_language_probability must be between 0 and 1$"):
        _website_source(min_language_probability="high")


def test_v3_adapters_accept_the_closed_probability_interval() -> None:
    assert _website_source(min_language_probability=0.0).min_language_probability == 0.0
    assert _website_source(min_language_probability=1.0).min_language_probability == 1.0


def test_description_adapter_keeps_a_score_exactly_on_the_language_threshold() -> None:
    source = _description_source(
        (iter([_description_row(identity="a" * 64, top_score=0.90)]),),
        (iter([_geometry_row()]),),
        min_language_score=0.90,
    )

    assert len(list(source.iter_candidates())) == 1


def test_description_adapter_reads_a_single_upstream_sentence_field() -> None:
    row = _description_row(identity="a" * 64)
    del row["sentences"]
    row["sentence"] = "One upstream description sentence."
    source = _description_source((iter([row]),), (iter([_geometry_row()]),))

    assert [candidate.sentence for candidate in source.iter_candidates()] == [
        "One upstream description sentence."
    ]


def test_description_adapter_joins_on_the_upstream_description_identity() -> None:
    sentence_row = {
        "description_identity": "shared-identity",
        "language_code": "eng",
        "top_score": 0.99,
        "sentences": ["A description sentence keyed only by its identity."],
    }
    geometry_row = {"description_identity": "shared-identity", "lat": 45.0, "lon": 2.0}
    source = _description_source((iter([sentence_row]),), (iter([geometry_row]),))

    assert [candidate.source_record_id for candidate in source.iter_candidates()] == ["shared-identity"]


def test_description_adapter_needs_every_part_of_the_osm_identity_to_use_it() -> None:
    sentence_row = {
        "description_identity": "shared-identity",
        "source_pbf": "region.osm.pbf",
        "language_code": "eng",
        "top_score": 0.99,
        "sentences": ["A description sentence with a partial OSM identity."],
    }
    geometry_row = {"description_identity": "shared-identity", "lat": 45.0, "lon": 2.0}
    source = _description_source((iter([sentence_row]),), (iter([geometry_row]),))

    assert [candidate.source_record_id for candidate in source.iter_candidates()] == ["shared-identity"]


def test_description_adapter_falls_back_to_the_join_key_for_a_record_without_an_identity() -> None:
    sentence_row = _description_row(identity="a" * 64)
    del sentence_row["description_identity"]
    source = _description_source((iter([sentence_row]),), (iter([_geometry_row()]),))

    assert [candidate.source_record_id for candidate in source.iter_candidates()] == ["region.osm.pbf|node|1"]


def test_description_adapter_skips_geometry_without_usable_coordinates() -> None:
    geometry = _geometry_row()
    geometry["lat"] = None
    geometry["lon"] = None
    source = _description_source((iter([_description_row(identity="a" * 64)]),), (iter([geometry]),))

    assert list(source.iter_candidates()) == []


def test_description_adapter_needs_both_corners_of_an_upstream_bbox() -> None:
    geometry = {
        **_geometry_row(),
        "lat": None,
        "lon": None,
        "bbox_min_x": 1.0,
        "bbox_min_y": 44.0,
    }
    source = _description_source((iter([_description_row(identity="a" * 64)]),), (iter([geometry]),))

    assert list(source.iter_candidates()) == []


def test_description_adapter_stops_indexing_geometry_inside_a_shard() -> None:
    sentence_rows = CountingRows(
        [
            _description_row(identity="a" * 64, osm_id="1"),
            _description_row(identity="b" * 64, osm_id="2"),
        ]
    )
    geometry_rows = CountingRows([_geometry_row(osm_id="1"), _geometry_row(osm_id="2")])
    source = _description_source((sentence_rows,), (geometry_rows,), max_join_entries=1)

    candidates = list(source.iter_candidates())

    assert [candidate.source_record_id for candidate in candidates] == ["a" * 64]


def test_description_adapter_keeps_a_sentence_exactly_on_the_text_bound() -> None:
    source = _description_source(
        (iter([_description_row(identity="a" * 64, sentence="0123456789")]),),
        (iter([_geometry_row()]),),
        max_text_characters=10,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["0123456789"]


def test_wikipedia_adapter_names_the_upstream_sentence_field_it_read() -> None:
    sentence_rows = CountingRows([_wikipedia_row(sentence_id="good")])
    source = _wikipedia_source((sentence_rows,), (CountingRows([_polygon_row()]),))

    assert next(source.iter_candidates()).source_field == "wikipedia_sentence"


def test_wikipedia_adapter_skips_a_record_without_any_sentence_identity() -> None:
    sentence_rows = CountingRows(
        [
            {
                "document_id": "d1",
                "section_id": "s1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "section_index": 1,
                "heading": "History",
                "text": "A contextual sentence with no sentence index.",
            },
            {
                "document_id": "d2",
                "sentence_index": 0,
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "section_index": 1,
                "heading": "History",
                "text": "A contextual sentence with no section id.",
            },
        ]
    )
    source = _wikipedia_source((sentence_rows,), (CountingRows([_polygon_row()]),))

    assert list(source.iter_candidates()) == []


def test_wikipedia_adapter_falls_back_to_the_wikidata_key_for_an_unnamed_polygon() -> None:
    polygon = _polygon_row()
    del polygon["polygon_id"]
    source = _wikipedia_source(
        (CountingRows([_wikipedia_row(sentence_id="good")]),), (CountingRows([polygon]),)
    )

    assert [candidate.candidate_id for candidate in source.iter_candidates()] == ["wikipedia:Q1:good"]


def test_wikipedia_adapter_skips_a_polygon_without_usable_coordinates() -> None:
    polygon = _polygon_row()
    polygon["lat"] = None
    source = _wikipedia_source(
        (CountingRows([_wikipedia_row(sentence_id="good")]),), (CountingRows([polygon]),)
    )

    assert list(source.iter_candidates()) == []


def test_website_adapter_bounds_the_rows_it_reads_from_each_shard() -> None:
    rows = CountingRows(
        [
            {
                "polygon_id": "p1",
                "lat": 45.0,
                "lon": 2.0,
                "website_language": "eng_Latn",
                "website_language_probability": 0.99,
                "website_sentences": ["The first row sentence."],
            },
            {
                "polygon_id": "p2",
                "lat": 46.0,
                "lon": 3.0,
                "website_language": "eng_Latn",
                "website_language_probability": 0.99,
                "website_sentences": ["The second row sentence."],
            },
        ]
    )
    source = _website_source(rows, max_rows_per_shard=1)

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["The first row sentence."]
    assert rows.seen == 1


def test_website_adapter_needs_both_a_polygon_identity_and_a_location() -> None:
    without_identity = {
        "lat": 45.0,
        "lon": 2.0,
        "website_language": "eng_Latn",
        "website_language_probability": 0.99,
        "website_sentences": ["No polygon identity."],
    }
    without_location = {
        "polygon_id": "p1",
        "website_language": "eng_Latn",
        "website_language_probability": 0.99,
        "website_sentences": ["No coordinates."],
    }

    assert list(_website_source((without_identity, without_location)).iter_candidates()) == []


def test_website_adapter_keeps_a_probability_exactly_on_the_threshold() -> None:
    row = {
        "polygon_id": "p1",
        "lat": 45.0,
        "lon": 2.0,
        "website_language": "eng_cyrl",
        "website_language_probability": 0.90,
        "website_sentences": ["An upstream sentence at the threshold."],
        "website_sentence_status": "success",
    }

    assert [candidate.sentence for candidate in _website_source((row,)).iter_candidates()] == [
        "An upstream sentence at the threshold."
    ]


def test_website_adapter_places_each_candidate_at_the_upstream_coordinates() -> None:
    row = {
        "polygon_id": "p1",
        "lat": 45.0,
        "lon": 2.0,
        "website_language": "en",
        "website_language_probability": 1.0,
        "website_sentences": ["An upstream sentence."],
    }
    source = WebsiteSentenceSource(
        row_shards_loader=lambda: ((row,),),
        cell_for_location=lambda latitude, longitude: f"cell:{latitude}:{longitude}",
    )

    candidate = next(source.iter_candidates())

    assert (candidate.latitude, candidate.longitude) == (45.0, 2.0)
    assert candidate.h3_cell == "cell:45.0:2.0"


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [(90.0, 180.0), (-90.0, -180.0)],
)
def test_website_adapter_accepts_the_extremes_of_the_coordinate_range(
    latitude: float, longitude: float
) -> None:
    row = {
        "polygon_id": "p1",
        "lat": latitude,
        "lon": longitude,
        "website_language": "en",
        "website_language_probability": 1.0,
        "website_sentences": ["An upstream sentence at the edge of the world."],
    }

    assert len(list(_website_source((row,)).iter_candidates())) == 1


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [(90.5, 0.0), (-90.5, 0.0), (0.0, 180.5), (0.0, -180.5), (45.0, None), (None, 2.0)],
)
def test_website_adapter_rejects_coordinates_outside_the_world(
    latitude: float | None, longitude: float | None
) -> None:
    row = {
        "polygon_id": "p1",
        "lat": latitude,
        "lon": longitude,
        "website_language": "en",
        "website_language_probability": 1.0,
        "website_sentences": ["An upstream sentence off the map."],
    }

    assert list(_website_source((row,)).iter_candidates()) == []


def test_description_adapter_ignores_an_osm_identity_missing_its_source_file() -> None:
    sentence_row = {
        "description_identity": "shared-identity",
        "osm_id": "1",
        "language_code": "eng",
        "top_score": 0.99,
        "sentences": ["A description sentence with only an OSM id."],
    }
    geometry_row = {"description_identity": "shared-identity", "lat": 45.0, "lon": 2.0}
    source = _description_source((iter([sentence_row]),), (iter([geometry_row]),))

    assert [candidate.source_record_id for candidate in source.iter_candidates()] == ["shared-identity"]


def test_description_adapter_derives_no_osm_url_from_half_an_identity() -> None:
    sentence_row = {
        "description_identity": "shared-identity",
        "language_code": "eng",
        "top_score": 0.99,
        "sentences": ["A description sentence whose geometry has no OSM id."],
    }
    geometry_row = {
        "description_identity": "shared-identity",
        "osm_type": "node",
        "lat": 45.0,
        "lon": 2.0,
    }
    source = _description_source((iter([sentence_row]),), (iter([geometry_row]),))

    assert next(source.iter_candidates()).source_url is None


def test_description_adapter_derives_a_region_from_the_upstream_source_pbf() -> None:
    source = _description_source(
        (iter([_description_row(identity="a" * 64)]),),
        (iter([_geometry_row()]),),
    )

    assert next(source.iter_candidates()).region == "region"


def test_description_adapter_derives_the_region_the_sibling_datasets_publish() -> None:
    sentence_row = _description_row(identity="a" * 64, source_pbf="afghanistan-latest.osm.pbf")
    geometry_row = _geometry_row(source_pbf="afghanistan-latest.osm.pbf")

    source = _description_source((iter([sentence_row]),), (iter([geometry_row]),))

    assert next(source.iter_candidates()).region == "afghanistan"


def test_description_adapter_emits_no_region_without_a_source_pbf() -> None:
    sentence_row = {
        "description_identity": "shared-identity",
        "language_code": "eng",
        "top_score": 0.99,
        "sentences": ["A description sentence with no source file."],
    }
    geometry_row = {"description_identity": "shared-identity", "lat": 45.0, "lon": 2.0}

    source = _description_source((iter([sentence_row]),), (iter([geometry_row]),))

    assert next(source.iter_candidates()).region is None


@pytest.mark.parametrize(
    ("source_pbf", "region"),
    [
        ("afghanistan-latest.osm.pbf", "afghanistan"),
        ("ile-de-france-latest.osm.pbf", "ile-de-france"),
        ("afghanistan.osm.pbf", "afghanistan"),
        ("afghanistan-latest", "afghanistan"),
        ("  ", None),
        (None, None),
    ],
)
def test_region_from_source_pbf_strips_only_the_extract_suffixes(
    source_pbf: str | None, region: str | None
) -> None:
    from landuse_sentence_relevance.sources.v3 import region_from_source_pbf

    assert region_from_source_pbf(source_pbf) == region


def test_the_join_reports_how_many_keys_it_indexed(caplog: pytest.LogCaptureFixture) -> None:
    source = _wikipedia_source(
        (CountingRows([_wikipedia_row(sentence_id="good")]),),
        (CountingRows([_polygon_row(), _polygon_row(polygon_id="p2", wikidata="Q2")]),),
    )

    with caplog.at_level("INFO", logger="landuse_sentence_relevance.sources.v3"):
        list(source.iter_candidates())

    messages = [record.getMessage() for record in caplog.records]
    assert "V3 join indexed 2 join keys from 1 shard(s)" in messages
    assert not any("join index is full" in message for message in messages)


def test_the_join_counts_every_shard_it_read(caplog: pytest.LogCaptureFixture) -> None:
    source = _wikipedia_source(
        (CountingRows([_wikipedia_row(sentence_id="good")]),),
        (
            CountingRows([_polygon_row()]),
            CountingRows([_polygon_row(polygon_id="p2", wikidata="Q2")]),
            CountingRows([]),
        ),
    )

    with caplog.at_level("INFO", logger="landuse_sentence_relevance.sources.v3"):
        list(source.iter_candidates())

    assert "V3 join indexed 2 join keys from 3 shard(s)" in [record.getMessage() for record in caplog.records]


def test_the_join_warns_when_the_cap_stops_it_indexing(caplog: pytest.LogCaptureFixture) -> None:
    source = _wikipedia_source(
        (CountingRows([_wikipedia_row(sentence_id="good")]),),
        (
            CountingRows([_polygon_row()]),
            CountingRows([_polygon_row(polygon_id="p2", wikidata="Q2")]),
        ),
        max_join_entries=1,
    )

    with caplog.at_level("INFO", logger="landuse_sentence_relevance.sources.v3"):
        list(source.iter_candidates())

    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert warnings == [
        "V3 join index is full at 1 keys after 1 shard(s); stopping before shard 2. "
        "Raise max_join_entries to keep indexing, or expect unmatched sentences beyond it."
    ]
    assert "V3 join indexed 1 join keys from 1 shard(s)" in [record.getMessage() for record in caplog.records]


class BlockingRows:
    """A shard that blocks until released, to prove shards are read concurrently."""

    def __init__(self, rows: Iterable[Mapping[str, Any]], barrier: Any) -> None:
        self._rows = tuple(rows)
        self._barrier = barrier

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        self._barrier.wait(timeout=10)
        yield from self._rows


def test_website_adapter_reads_shards_concurrently() -> None:
    """Each shard blocks until every other shard has started, so a sequential read deadlocks."""

    import threading

    shard_count = 4
    barrier = threading.Barrier(shard_count)
    shards = tuple(
        BlockingRows([_website_row(polygon_id=f"p{index}")], barrier) for index in range(shard_count)
    )
    source = WebsiteSentenceSource(
        row_shards_loader=lambda: shards,
        cell_for_location=_cell,
        max_stream_workers=shard_count,
    )

    candidates = list(source.iter_candidates())

    assert len(candidates) == shard_count


def test_website_adapter_yields_same_candidates_with_and_without_workers() -> None:
    rows = [_website_row(polygon_id=f"p{index}") for index in range(6)]
    shards = tuple((row,) for row in rows)

    sequential = _website_source(*shards, max_stream_workers=1)
    parallel = WebsiteSentenceSource(
        row_shards_loader=lambda: shards,
        cell_for_location=_cell,
        max_stream_workers=4,
    )

    sequential_ids = sorted(candidate.candidate_id for candidate in sequential.iter_candidates())
    parallel_ids = sorted(candidate.candidate_id for candidate in parallel.iter_candidates())
    assert sequential_ids == parallel_ids
    assert sequential_ids != []


def test_website_adapter_propagates_shard_failures_with_workers() -> None:
    def failing_rows() -> Iterator[Mapping[str, Any]]:
        yield _website_row(polygon_id="p0")
        raise RuntimeError("upstream shard failed")

    source = WebsiteSentenceSource(
        row_shards_loader=lambda: ((_website_row(polygon_id="p1"),), failing_rows()),
        cell_for_location=_cell,
        max_stream_workers=2,
    )

    with pytest.raises(RuntimeError, match="upstream shard failed"):
        list(source.iter_candidates())


def test_every_adapter_names_max_stream_workers_when_it_rejects_it() -> None:
    """The message is the only thing an operator sees: it has to name the setting to change.

    All three adapters take the same bound, so all three are checked -- a copied constructor
    that validates the wrong name would otherwise pass on the strength of its neighbours.
    """

    shards: tuple[()] = ()
    expected = r"^max_stream_workers must be positive$"
    with pytest.raises(ValueError, match=expected):
        _website_source(shards, max_stream_workers=0)
    with pytest.raises(ValueError, match=expected):
        _description_source(shards, shards, max_stream_workers=0)
    with pytest.raises(ValueError, match=expected):
        _wikipedia_source(shards, shards, max_stream_workers=0)


# --------------------------------------------------------------- concurrent shard reading


def _slow_rows(rows: Iterable[Mapping[str, Any]], delay: float = 0.0) -> Iterator[Mapping[str, Any]]:
    import time

    for row in rows:
        if delay:
            time.sleep(delay)
        yield row


class TestOrderedParallelShards:
    """`_ordered_parallel_shards` prefetches concurrently but must preserve shard order.

    The join index keeps the first value seen for a key, so its result depends on shard order.
    Prefetching that reordered shards would silently change which record wins a key -- a
    correctness bug that no amount of retrying would surface.
    """

    def test_a_single_worker_reads_sequentially(self) -> None:
        from landuse_sentence_relevance.sources.v3 import _ordered_parallel_shards

        shards = [iter([{"n": 0}]), iter([{"n": 1}]), iter([{"n": 2}])]
        built = list(_ordered_parallel_shards(shards, lambda rows: list(rows), max_workers=1))
        assert built == [[{"n": 0}], [{"n": 1}], [{"n": 2}]]

    def test_results_come_back_in_shard_order_despite_uneven_timing(self) -> None:
        """The first shard is the slowest, so arrival order and shard order disagree."""
        from landuse_sentence_relevance.sources.v3 import _ordered_parallel_shards

        delays = {0: 0.05, 1: 0.0, 2: 0.0, 3: 0.0}
        shards = [_slow_rows([{"n": index}], delays[index]) for index in range(4)]
        built = list(_ordered_parallel_shards(shards, lambda rows: list(rows), max_workers=3))
        assert built == [[{"n": index}] for index in range(4)]

    def test_every_shard_is_built_when_there_are_more_than_workers(self) -> None:
        """The sliding window must keep submitting past the initial batch."""
        from landuse_sentence_relevance.sources.v3 import _ordered_parallel_shards

        shards = [iter([{"n": index}]) for index in range(9)]
        built = list(_ordered_parallel_shards(shards, lambda rows: list(rows), max_workers=2))
        assert [rows[0]["n"] for rows in built] == list(range(9))

    def test_no_shards_yields_nothing(self) -> None:
        from landuse_sentence_relevance.sources.v3 import _ordered_parallel_shards

        assert list(_ordered_parallel_shards([], lambda rows: list(rows), max_workers=4)) == []

    def test_a_shard_that_raises_propagates_rather_than_being_dropped(self) -> None:
        from landuse_sentence_relevance.sources.v3 import _ordered_parallel_shards

        def build(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
            materialised = list(rows)
            if materialised and materialised[0]["n"] == 1:
                raise RuntimeError("shard failed")
            return materialised

        shards = [iter([{"n": index}]) for index in range(3)]
        with pytest.raises(RuntimeError, match="shard failed"):
            list(_ordered_parallel_shards(shards, build, max_workers=2))


class TestParallelShardRows:
    """`_parallel_shard_rows` interleaves shards, which is safe only because the pool ranks
    candidates by hash rather than by arrival order."""

    def test_a_single_worker_reads_shards_in_order(self) -> None:
        from landuse_sentence_relevance.sources.v3 import _parallel_shard_rows

        shards = [iter([{"n": 0}, {"n": 1}]), iter([{"n": 2}])]
        assert list(_parallel_shard_rows(shards, limit=10, max_workers=1)) == [
            {"n": 0},
            {"n": 1},
            {"n": 2},
        ]

    def test_one_shard_is_read_sequentially_whatever_the_worker_count(self) -> None:
        from landuse_sentence_relevance.sources.v3 import _parallel_shard_rows

        rows = list(_parallel_shard_rows([iter([{"n": 0}, {"n": 1}])], limit=10, max_workers=4))
        assert rows == [{"n": 0}, {"n": 1}]

    def test_every_row_arrives_when_shards_are_read_concurrently(self) -> None:
        """Order is not asserted -- interleaving is the point -- but nothing may be lost."""
        from landuse_sentence_relevance.sources.v3 import _parallel_shard_rows

        shards = [iter([{"n": index}, {"n": index + 100}]) for index in range(4)]
        rows = list(_parallel_shard_rows(shards, limit=10, max_workers=3))
        assert sorted(row["n"] for row in rows) == sorted(
            [index for index in range(4)] + [index + 100 for index in range(4)]
        )

    def test_the_limit_is_applied_per_shard(self) -> None:
        from landuse_sentence_relevance.sources.v3 import _parallel_shard_rows

        shards = [iter([{"n": index} for index in range(10)]) for _ in range(3)]
        rows = list(_parallel_shard_rows(shards, limit=2, max_workers=3))
        assert len(rows) == 6

    def test_a_failing_shard_re_raises_in_the_consumer(self) -> None:
        """A shard that raises must not be silently dropped: the pool would be short and
        nothing would say so."""
        from landuse_sentence_relevance.sources.v3 import _parallel_shard_rows

        def exploding() -> Iterator[Mapping[str, Any]]:
            yield {"n": 0}
            raise RuntimeError("upstream died")

        shards = [exploding(), iter([{"n": 1}]), iter([{"n": 2}])]
        with pytest.raises(RuntimeError, match="upstream died"):
            list(_parallel_shard_rows(shards, limit=10, max_workers=3))


# --------------------------------------------- the contracts the concurrency arguments carry


class _RecordingExecutor(ThreadPoolExecutor):
    """A pool that records how it was constructed and shut down.

    The arguments matter and are otherwise unobservable: the worker count is what bounds
    concurrent upstream reads, the thread prefix is what lets a stack dump name the shard that
    is stuck, and ``wait=True`` on shutdown is what stops a generator from returning while its
    threads are still reading. Asserting them here is how those stay true.
    """

    constructed: ClassVar[list[dict[str, Any]]] = []
    shutdowns: ClassVar[list[dict[str, Any]]] = []
    submits: ClassVar[list[str]] = []

    def __init__(self, max_workers: Any = None, thread_name_prefix: Any = "", **kwargs: Any) -> None:
        type(self).constructed.append({"max_workers": max_workers, "thread_name_prefix": thread_name_prefix})
        super().__init__(
            max_workers=max_workers if isinstance(max_workers, int) and max_workers > 0 else 2,
            thread_name_prefix=thread_name_prefix if isinstance(thread_name_prefix, str) else "",
            **kwargs,
        )

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        type(self).submits.append(getattr(fn, "__name__", type(fn).__name__))
        return super().submit(fn, *args, **kwargs)

    def shutdown(self, wait: Any = True, **kwargs: Any) -> None:
        type(self).shutdowns.append({"wait": wait})
        super().shutdown(wait=bool(wait), **kwargs)


class _RecordingQueue(Queue):  # type: ignore[type-arg]
    """A queue that records the bound it was asked for, then works regardless.

    The bound is what keeps memory flat while shards race ahead of the consumer. Recording the
    requested value rather than trusting the constructor means an unusable bound is reported by
    a failing assertion instead of by a hung test.
    """

    requested: ClassVar[list[Any]] = []

    def __init__(self, maxsize: Any = 0) -> None:
        type(self).requested.append(maxsize)
        super().__init__(maxsize=maxsize if isinstance(maxsize, int) else 0)


@pytest.fixture
def recorded_pools(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingExecutor]:
    _RecordingExecutor.constructed = []
    _RecordingExecutor.shutdowns = []
    _RecordingExecutor.submits = []
    _RecordingQueue.requested = []
    monkeypatch.setattr(v3_module, "ThreadPoolExecutor", _RecordingExecutor)
    monkeypatch.setattr(v3_module, "Queue", _RecordingQueue)
    return _RecordingExecutor


class TestShardReadingIsBounded:
    """What `_parallel_shard_rows` asks the runtime for, which its output cannot show."""

    def test_the_pool_is_sized_to_the_smaller_of_shards_and_workers(
        self, recorded_pools: type[_RecordingExecutor]
    ) -> None:
        shards = [iter([{"n": index}]) for index in range(3)]

        list(_parallel_shard_rows(shards, limit=10, max_workers=8))

        assert recorded_pools.constructed == [{"max_workers": 3, "thread_name_prefix": "v3-shard"}]

    def test_the_pool_never_exceeds_the_worker_budget(self, recorded_pools: type[_RecordingExecutor]) -> None:
        shards = [iter([{"n": index}]) for index in range(6)]

        list(_parallel_shard_rows(shards, limit=10, max_workers=2))

        assert recorded_pools.constructed == [{"max_workers": 2, "thread_name_prefix": "v3-shard"}]

    def test_the_generator_waits_for_its_threads_before_returning(
        self, recorded_pools: type[_RecordingExecutor]
    ) -> None:
        """Returning while a worker still reads would leave an upstream stream being consumed
        after the caller believes the read is over."""
        shards = [iter([{"n": index}]) for index in range(3)]

        list(_parallel_shard_rows(shards, limit=10, max_workers=3))

        assert recorded_pools.shutdowns == [{"wait": True}]

    def test_the_queue_is_bounded_to_the_configured_capacity(
        self, recorded_pools: type[_RecordingExecutor]
    ) -> None:
        shards = [iter([{"n": index}]) for index in range(3)]

        list(_parallel_shard_rows(shards, limit=10, max_workers=3))

        assert _RecordingQueue.requested == [_SHARD_QUEUE_CAPACITY]

    def test_rows_are_read_on_the_pool_threads(self) -> None:
        """The prefix is what identifies a stuck shard in a stack dump."""
        seen: list[str] = []

        def observing() -> Iterator[Mapping[str, Any]]:
            seen.append(threading.current_thread().name)
            yield {"n": 0}

        list(_parallel_shard_rows([observing(), observing()], limit=10, max_workers=2))

        assert seen and all(name.startswith("v3-shard") for name in seen), seen

    @pytest.mark.parametrize(
        ("shards", "workers", "concurrent"),
        [
            (3, 1, False),
            (1, 4, False),
            (2, 2, True),
            (3, 2, True),
        ],
    )
    def test_the_pool_is_built_only_when_there_is_something_to_overlap(
        self,
        recorded_pools: type[_RecordingExecutor],
        shards: int,
        workers: int,
        concurrent: bool,
    ) -> None:
        """One shard, or one worker, has nothing to overlap: paying for a thread pool there is
        pure cost, and the sequential path also keeps the read lazy."""
        streams = [iter([{"n": index}]) for index in range(shards)]

        rows = list(_parallel_shard_rows(streams, limit=10, max_workers=workers))

        assert len(rows) == shards
        assert bool(recorded_pools.constructed) is concurrent


class TestJoinShardRows:
    """`_join_shard_rows` feeds the join index, which depends on shard order."""

    def test_each_shard_is_bounded_and_kept_in_order_when_prefetching(
        self, recorded_pools: type[_RecordingExecutor]
    ) -> None:
        shards = [iter([{"n": index}, {"n": index + 100}, {"n": index + 200}]) for index in range(4)]

        built = [tuple(rows) for rows in _join_shard_rows(shards, max_rows_per_shard=2, max_workers=3)]

        assert built == [({"n": index}, {"n": index + 100}) for index in range(4)]
        assert recorded_pools.constructed == [{"max_workers": 3, "thread_name_prefix": "v3-join"}]

    def test_a_single_worker_stays_lazy_and_opens_no_pool(
        self, recorded_pools: type[_RecordingExecutor]
    ) -> None:
        """The sequential path must not touch the next shard before the caller asks for it."""
        opened: list[int] = []

        def tracked(index: int) -> Iterator[Mapping[str, Any]]:
            opened.append(index)
            yield {"n": index}

        shards = [tracked(0), tracked(1), tracked(2)]
        stream = _join_shard_rows(shards, max_rows_per_shard=5, max_workers=1)

        first = tuple(next(stream))

        assert first == ({"n": 0},)
        assert opened == [0]
        assert recorded_pools.constructed == []

    def test_two_workers_prefetch_rather_than_waiting_to_be_asked(
        self, recorded_pools: type[_RecordingExecutor]
    ) -> None:
        """Two is already a worker budget: the boundary between lazy and prefetching sits
        below it, not above."""
        shards = [iter([{"n": index}]) for index in range(2)]

        stream = _join_shard_rows(shards, max_rows_per_shard=5, max_workers=2)
        next(stream)

        assert recorded_pools.constructed == [{"max_workers": 2, "thread_name_prefix": "v3-join"}]
        list(stream)


class TestSlidingWindow:
    """The window is what keeps prefetching from reading the whole source into memory."""

    def test_only_the_window_is_submitted_before_the_first_result_is_taken(self) -> None:
        """A source is 386 shards; submitting them all at once would hold every shard's rows.

        Submissions are counted rather than started builds: with a bounded pool, an unbounded
        submission still shows only a couple of builds running, so counting builds would let
        the bound disappear unnoticed.
        """

        class CountingExecutor:
            def __init__(self, inner: ThreadPoolExecutor) -> None:
                self._inner = inner
                self.submits = 0

            def submit(self, fn: Any, *args: Any) -> Any:
                self.submits += 1
                return self._inner.submit(fn, *args)

        shards = iter([iter([{"n": index}]) for index in range(6)])
        with ThreadPoolExecutor(max_workers=2) as inner:
            executor = CountingExecutor(inner)
            window = _sliding_window(
                shards, lambda rows: list(rows), cast(ThreadPoolExecutor, executor), width=2
            )
            first = next(window)
            submitted_before_first_result = executor.submits
            rest = list(window)

        assert submitted_before_first_result <= 3, submitted_before_first_result
        assert [first[0]["n"], *[rows[0]["n"] for rows in rest]] == list(range(6))
        assert executor.submits == 6


class TestTheWorkerBudgetReachesTheShardReaders:
    """`max_stream_workers` is configuration; these check it arrives where it is spent."""

    def test_the_joined_adapters_read_their_join_index_with_the_configured_workers(
        self, recorded_pools: type[_RecordingExecutor]
    ) -> None:
        """The join index is the slow half of both joined adapters: a budget that stopped short
        of it would leave the run reading one shard at a time while reporting three workers."""
        sentences = (iter([_description_row(identity="a" * 64)]),)
        geometry = (iter([_geometry_row()]), iter([_geometry_row()]))

        list(_description_source(sentences, geometry, max_stream_workers=3).iter_candidates())

        assert {"max_workers": 3, "thread_name_prefix": "v3-join"} in recorded_pools.constructed

    def test_the_wikipedia_adapter_reads_its_polygon_index_with_the_configured_workers(
        self, recorded_pools: type[_RecordingExecutor]
    ) -> None:
        sentences = (iter([_wikipedia_row(sentence_id="s1")]),)
        polygons = (iter([_polygon_row()]), iter([_polygon_row()]))

        list(_wikipedia_source(sentences, polygons, max_stream_workers=3).iter_candidates())

        assert {"max_workers": 3, "thread_name_prefix": "v3-join"} in recorded_pools.constructed


class TestOrderedShardsPrefetchWithinItsBudget:
    """`_ordered_parallel_shards` decides whether to open a pool at all, and how far to run ahead."""

    def test_a_single_worker_opens_no_pool(self, recorded_pools: type[_RecordingExecutor]) -> None:
        """One worker has nothing to overlap, and the sequential path also stays lazy."""
        shards = [iter([{"n": index}]) for index in range(3)]

        built = list(_ordered_parallel_shards(shards, lambda rows: list(rows), max_workers=1))

        assert [rows[0]["n"] for rows in built] == [0, 1, 2]
        assert recorded_pools.constructed == []

    def test_only_the_worker_budget_is_submitted_before_the_first_result(
        self, recorded_pools: type[_RecordingExecutor]
    ) -> None:
        """A source is hundreds of shards; a window that ran ahead without bound would submit
        every one of them and hold their rows at once.

        Submissions are counted rather than completed builds: with a bounded pool, an unbounded
        window still shows only a couple of builds *finished*, so counting those would let the
        bound disappear unnoticed.
        """

        shards = [iter([{"n": index}]) for index in range(6)]
        stream = _ordered_parallel_shards(shards, lambda rows: list(rows), max_workers=2)

        first = next(stream)
        submitted_before_first_result = len(recorded_pools.submits)
        rest = list(stream)

        assert submitted_before_first_result <= 3, recorded_pools.submits
        assert [first[0]["n"], *[rows[0]["n"] for rows in rest]] == list(range(6))
        assert len(recorded_pools.submits) == 6


class TestShardProgressIsReported:
    """A source that streams for hours must say where it is. See #21."""

    def test_the_sequential_path_reports_each_shard_as_it_finishes(self, caplog) -> None:
        shards = [iter([{"n": 0}, {"n": 1}]), iter([{"n": 2}])]

        with caplog.at_level(logging.INFO, logger="landuse_sentence_relevance.sources.v3"):
            list(_parallel_shard_rows(shards, limit=10, max_workers=1, label="website"))

        messages = [record.getMessage() for record in caplog.records]
        assert messages == [
            "website: finished shard 1/2 (2 rows)",
            "website: finished shard 2/2 (1 rows)",
        ]

    def test_the_concurrent_path_reports_every_shard(self, caplog) -> None:
        """Order is not asserted -- shards race by design -- but none may go unreported."""
        shards = [iter([{"n": index}]) for index in range(4)]

        with caplog.at_level(logging.INFO, logger="landuse_sentence_relevance.sources.v3"):
            list(_parallel_shard_rows(shards, limit=10, max_workers=3, label="description"))

        reported = [r.getMessage() for r in caplog.records if "finished shard" in r.getMessage()]
        assert len(reported) == 4
        assert all("description" in message for message in reported)
        assert all("/4" in message for message in reported)

    def test_the_join_index_reports_its_shards_too(self, caplog) -> None:
        """The join side is the slow half of both joined adapters.

        The rows are consumed, not just collected: the sequential path hands back lazy streams so
        the join index can stop opening shards once it holds enough keys, and a shard nobody reads
        has not finished.
        """
        shards = [iter([{"n": 0}]), iter([{"n": 1}])]

        with caplog.at_level(logging.INFO, logger="landuse_sentence_relevance.sources.v3"):
            for shard in _join_shard_rows(shards, max_rows_per_shard=5, max_workers=1, label="wikipedia"):
                list(shard)

        reported = [r.getMessage() for r in caplog.records if "finished shard" in r.getMessage()]
        assert len(reported) == 2
        assert all("wikipedia" in message for message in reported)

    def test_no_row_content_reaches_the_log(self, caplog) -> None:
        secret = "a sentence that must never be logged"
        shards = [iter([{"text": secret}])]

        with caplog.at_level(logging.INFO, logger="landuse_sentence_relevance.sources.v3"):
            list(_parallel_shard_rows(shards, limit=10, max_workers=1, label="website"))

        assert all(secret not in record.getMessage() for record in caplog.records)


class TestShardLevelResume:
    """A source interrupted part way must reopen only its unread shards. See #20.

    The website source is ~386 shards read in alphabetical order. It was re-streamed from
    `afghanistan` three times in one day because each run was interrupted before `zimbabwe`, and
    each interruption threw away every shard it had processed -- one leg ran 5h12m and had not
    cleared `sri-lanka`.
    """

    def test_read_shards_are_not_reopened(self) -> None:
        opened: list[int] = []

        def shard(index: int) -> Iterator[Mapping[str, Any]]:
            opened.append(index)
            yield {"n": index}

        shards = [shard(index) for index in range(4)]
        rows = list(_parallel_shard_rows(shards, limit=10, max_workers=1, skip_shards=frozenset({0, 2})))

        assert opened == [1, 3], "a skipped shard is never opened, not merely discarded"
        assert [row["n"] for row in rows] == [1, 3]

    def test_finished_shards_are_reported_by_index(self) -> None:
        finished: list[int] = []
        shards = [iter([{"n": index}]) for index in range(3)]

        list(_parallel_shard_rows(shards, limit=10, max_workers=1, on_shard_done=finished.append))

        assert sorted(finished) == [0, 1, 2]

    def test_a_shard_abandoned_part_way_is_not_reported(self) -> None:
        """The whole point: an interrupted shard must be re-read, not skipped next time."""
        finished: list[int] = []
        shards = [iter([{"n": 0}, {"n": 1}]), iter([{"n": 2}])]

        stream = _parallel_shard_rows(shards, limit=10, max_workers=1, on_shard_done=finished.append)
        next(stream)
        cast(Generator[Any, None, None], stream).close()

        assert finished == []

    def test_skipping_preserves_the_index_of_the_shards_that_remain(self) -> None:
        """Indices name positions in the resolved shard list, so a resumed run records the same
        number for the same shard as the run before it."""
        finished: list[int] = []
        shards = [iter([{"n": index}]) for index in range(4)]

        list(
            _parallel_shard_rows(
                shards,
                limit=10,
                max_workers=1,
                skip_shards=frozenset({0, 1}),
                on_shard_done=finished.append,
            )
        )

        assert sorted(finished) == [2, 3]


class TestAdaptersCarryTheirResumeState:
    """The resume arguments are unobservable in an adapter's output, so they are asserted here.

    Each adapter passes `skip_shards` and `on_shard_done` down to its own row stream. An adapter
    that dropped either would re-read shards a run had already finished, silently, which is the
    failure #20 exists to prevent.
    """

    def test_the_description_adapter_skips_the_shards_it_already_read(self) -> None:
        opened: list[int] = []

        def shard(index: int) -> Iterator[Mapping[str, Any]]:
            opened.append(index)
            yield _description_row(identity=f"{index:064x}")

        source = _description_source(
            [shard(0), shard(1), shard(2)],
            (iter([_geometry_row()]),),
            skip_shards=frozenset({0, 2}),
        )
        list(source.iter_candidates())

        assert opened == [1]

    def test_the_wikipedia_adapter_skips_the_shards_it_already_read(self) -> None:
        opened: list[int] = []

        def shard(index: int) -> Iterator[Mapping[str, Any]]:
            opened.append(index)
            yield _wikipedia_row(sentence_id=f"s{index}")

        source = _wikipedia_source(
            [shard(0), shard(1)], (iter([_polygon_row()]),), skip_shards=frozenset({0})
        )
        list(source.iter_candidates())

        assert opened == [1]

    def test_the_website_adapter_skips_the_shards_it_already_read(self) -> None:
        opened: list[int] = []

        def shard(index: int) -> Iterator[Mapping[str, Any]]:
            opened.append(index)
            yield _website_row(polygon_id=f"p{index}")

        source = _website_source(shard(0), shard(1), skip_shards=frozenset({1}))
        list(source.iter_candidates())

        assert opened == [0]

    def test_each_adapter_reports_the_shards_it_finished(self) -> None:
        for build in (
            lambda done: _description_source(
                (iter([_description_row(identity="a" * 64)]),),
                (iter([_geometry_row()]),),
                on_shard_done=done.append,
            ),
            lambda done: _wikipedia_source(
                (iter([_wikipedia_row(sentence_id="s1")]),),
                (iter([_polygon_row()]),),
                on_shard_done=done.append,
            ),
            lambda done: _website_source(iter([_website_row()]), on_shard_done=done.append),
        ):
            finished: list[int] = []
            list(build(finished).iter_candidates())
            assert finished == [0], build


class TestEachStreamIsNamedInTheLog:
    """Five streams run; a progress line that does not say which one is moving is no use.

    The labels are unobservable in the adapters' output, so they are asserted here -- an adapter
    labelled with its neighbour's name would send an operator to the wrong stream.
    """

    def test_the_description_adapter_names_both_of_its_streams(self, caplog) -> None:
        source = _description_source(
            (iter([_description_row(identity="a" * 64)]),),
            (iter([_geometry_row()]),),
        )

        with caplog.at_level(logging.INFO, logger="landuse_sentence_relevance.sources.v3"):
            list(source.iter_candidates())

        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "description sentences" in messages
        assert "description geometry" in messages

    def test_the_wikipedia_adapter_names_both_of_its_streams(self, caplog) -> None:
        source = _wikipedia_source((iter([_wikipedia_row(sentence_id="s1")]),), (iter([_polygon_row()]),))

        with caplog.at_level(logging.INFO, logger="landuse_sentence_relevance.sources.v3"):
            list(source.iter_candidates())

        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "wikipedia sentences" in messages
        assert "wikipedia polygons" in messages

    def test_the_website_adapter_names_its_stream(self, caplog) -> None:
        with caplog.at_level(logging.INFO, logger="landuse_sentence_relevance.sources.v3"):
            list(_website_source(iter([_website_row()])).iter_candidates())

        assert "website rows" in " ".join(r.getMessage() for r in caplog.records)

    def test_an_unlabelled_stream_reports_nothing(self) -> None:
        """The label is what turns reporting on: the helpers stay silent without one, so a
        caller that does not want progress does not pay for it."""
        shards = [iter([{"n": 0}])]

        assert list(_parallel_shard_rows(shards, limit=5, max_workers=1)) == [{"n": 0}]


def test_the_concurrent_path_records_the_shards_it_finished() -> None:
    """REGRESSION: `on_shard_done` was wired into the sequential branch only.

    Every resume test used one worker, so a source running with a worker budget -- which is the
    configuration the long runs actually use -- recorded nothing and re-read every shard on the
    next attempt. The bug #20 exists to fix, reintroduced inside its own fix.
    """
    finished: list[int] = []
    shards = [iter([{"n": index}]) for index in range(4)]

    list(_parallel_shard_rows(shards, limit=10, max_workers=3, on_shard_done=finished.append))

    assert sorted(finished) == [0, 1, 2, 3]


def test_the_concurrent_path_skips_the_shards_it_already_read() -> None:
    opened: list[int] = []

    def shard(index: int) -> Iterator[Mapping[str, Any]]:
        opened.append(index)
        yield {"n": index}

    shards = [shard(index) for index in range(4)]
    rows = list(
        _parallel_shard_rows(shards, limit=10, max_workers=3, skip_shards=frozenset({1, 2}))
    )

    assert sorted(opened) == [0, 3]
    assert sorted(row["n"] for row in rows) == [0, 3]
