"""Streaming adapters for the immutable V3 sentence source records."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice, takewhile
from math import isfinite
from queue import Queue
from threading import Event
from typing import Any, Protocol

from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.sources.validation import validate_positive_limit

type Row = Mapping[str, Any]
type RowStream = Iterable[Row]
type ShardLoader = Callable[[], Iterable[RowStream]]
type CellForLocation = Callable[[float, float], str]

logger = logging.getLogger(__name__)

_ENGLISH_CODES = frozenset({"en", "eng", "eng_latn"})
_SECTION_HEADING_LINE = re.compile(r"^\[[^\]]*\bedit\b[^\]]*\]", re.IGNORECASE)
_EXTRACT_SUFFIXES = (".pbf", ".osm", "-latest")
_DEFAULT_MAX_TEXT_CHARACTERS = 400
_DEFAULT_MAX_ROWS_PER_SHARD = 20_000
_DEFAULT_MAX_JOIN_ENTRIES = 500_000
_DEFAULT_MAX_STREAM_WORKERS = 1
_SHARD_QUEUE_CAPACITY = 2_000


@dataclass(frozen=True, slots=True)
class _WebsiteField:
    sentences: str
    language: str
    probability: str
    url: str
    status: str


_WEBSITE_FIELDS = (
    _WebsiteField(
        sentences="website_sentences",
        language="website_language",
        probability="website_language_probability",
        url="website",
        status="website_sentence_status",
    ),
    _WebsiteField(
        sentences="contact_website_sentences",
        language="contact_website_language",
        probability="contact_website_language_probability",
        url="contact_website",
        status="contact_website_sentence_status",
    ),
)


@dataclass(frozen=True, slots=True)
class JoinedPlace:
    """The bounded projection of one lookup row kept in a streaming join."""

    latitude: float
    longitude: float
    record_id: str | None = None
    place_name: str | None = None
    region: str | None = None
    source_url: str | None = None


class V3CandidateSource(Protocol):
    """The narrow iterable contract consumed by the V3 reservoir builder."""

    def iter_candidates(self) -> Iterator[Candidate]: ...


@dataclass(frozen=True, slots=True)
class V3SourceAdapters:
    """The three independently streamable V3 candidate adapters."""

    description: V3CandidateSource
    wikipedia: V3CandidateSource
    website: V3CandidateSource


class DescriptionSentenceSource:
    """Adapt language-v1 description sentences and geometry into candidates.

    Sentence rows carry no coordinates, so geometry is resolved through a
    bounded streaming join: the geometry side is read once, shard by shard, and
    only a small projection of each row is retained, capped at
    ``max_join_entries`` keys. A sentence therefore still finds its geometry
    when the two rows sit in differently numbered shards, and neither upstream
    dataset is ever materialized.
    """

    def __init__(
        self,
        *,
        sentence_shards_loader: ShardLoader,
        geometry_shards_loader: ShardLoader,
        cell_for_location: CellForLocation,
        max_rows_per_shard: int = _DEFAULT_MAX_ROWS_PER_SHARD,
        min_language_score: float = 0.90,
        max_text_characters: int = _DEFAULT_MAX_TEXT_CHARACTERS,
        max_join_entries: int = _DEFAULT_MAX_JOIN_ENTRIES,
        max_stream_workers: int = _DEFAULT_MAX_STREAM_WORKERS,
    ) -> None:
        self._sentence_shards_loader = sentence_shards_loader
        self._geometry_shards_loader = geometry_shards_loader
        self._cell_for_location = cell_for_location
        validate_positive_limit(max_rows_per_shard, "max_rows_per_shard")
        self.max_rows_per_shard = max_rows_per_shard
        self.min_language_score = _validate_probability(min_language_score, "min_language_score")
        validate_positive_limit(max_text_characters, "max_text_characters")
        self.max_text_characters = max_text_characters
        validate_positive_limit(max_join_entries, "max_join_entries")
        self.max_join_entries = max_join_entries
        validate_positive_limit(max_stream_workers, "max_stream_workers")
        self.max_stream_workers = max_stream_workers

    def iter_candidates(self) -> Iterator[Candidate]:
        geometry_by_identity = _join_index(
            self._geometry_shards_loader(),
            key_for_row=_description_join_key,
            place_for_row=_description_place,
            max_rows_per_shard=self.max_rows_per_shard,
            max_entries=self.max_join_entries,
            max_workers=self.max_stream_workers,
        )
        rows = _parallel_shard_rows(
            self._sentence_shards_loader(), self.max_rows_per_shard, self.max_stream_workers
        )
        for row in rows:
            yield from self._row_candidates(row, geometry_by_identity)

    def _row_candidates(self, row: Row, geometry: Mapping[str, JoinedPlace]) -> Iterator[Candidate]:
        identity = _description_join_key(row)
        if identity is None or not _is_english_description(row, self.min_language_score):
            return
        place = geometry.get(identity)
        if place is None:
            return
        source_record_id = _optional_text(row.get("description_identity")) or identity
        yield from self._sentence_candidates(row, source_record_id, place)

    def _sentence_candidates(
        self, row: Row, source_record_id: str, place: JoinedPlace
    ) -> Iterator[Candidate]:
        for sentence_index, sentence in _upstream_sentences(row):
            cleaned = _bounded_sentence(sentence, self.max_text_characters)
            if cleaned is not None:
                yield _candidate(
                    candidate_id=f"description:{source_record_id}:{sentence_index}",
                    sentence=cleaned,
                    source=Source.DESCRIPTION,
                    source_record_id=source_record_id,
                    source_field="description",
                    place=place,
                    cell_for_location=self._cell_for_location,
                )


class WikipediaSentenceSource:
    """Adapt English Wikipedia sentence records joined to polygon metadata.

    The polygon side is joined with the same bounded streaming index the
    description source uses, so a sentence keeps its polygon regardless of the
    shard each row happens to land in.
    """

    def __init__(
        self,
        *,
        sentence_shards_loader: ShardLoader,
        polygon_shards_loader: ShardLoader,
        cell_for_location: CellForLocation,
        max_rows_per_shard: int = _DEFAULT_MAX_ROWS_PER_SHARD,
        max_text_characters: int = _DEFAULT_MAX_TEXT_CHARACTERS,
        max_join_entries: int = _DEFAULT_MAX_JOIN_ENTRIES,
        max_stream_workers: int = _DEFAULT_MAX_STREAM_WORKERS,
    ) -> None:
        self._sentence_shards_loader = sentence_shards_loader
        self._polygon_shards_loader = polygon_shards_loader
        self._cell_for_location = cell_for_location
        validate_positive_limit(max_rows_per_shard, "max_rows_per_shard")
        self.max_rows_per_shard = max_rows_per_shard
        validate_positive_limit(max_text_characters, "max_text_characters")
        self.max_text_characters = max_text_characters
        validate_positive_limit(max_join_entries, "max_join_entries")
        self.max_join_entries = max_join_entries
        validate_positive_limit(max_stream_workers, "max_stream_workers")
        self.max_stream_workers = max_stream_workers

    def iter_candidates(self) -> Iterator[Candidate]:
        polygons_by_wikidata = _join_index(
            self._polygon_shards_loader(),
            key_for_row=_wikidata_key,
            place_for_row=_wikipedia_polygon_place,
            max_rows_per_shard=self.max_rows_per_shard,
            max_entries=self.max_join_entries,
            max_workers=self.max_stream_workers,
        )
        rows = _parallel_shard_rows(
            self._sentence_shards_loader(), self.max_rows_per_shard, self.max_stream_workers
        )
        for row in rows:
            candidate = self._candidate_for(row, polygons_by_wikidata)
            if candidate is not None:
                yield candidate

    def _candidate_for(self, row: Row, polygons: Mapping[str, JoinedPlace]) -> Candidate | None:
        sentence = self._contextual_sentence(row)
        place = _joined_place(polygons, _wikidata_key(row))
        sentence_id = _wikipedia_sentence_id(row)
        if sentence is None or place is None or sentence_id is None:
            return None
        return _candidate(
            candidate_id=f"wikipedia:{place.record_id}:{sentence_id}",
            sentence=sentence,
            source=Source.WIKIPEDIA,
            source_record_id=sentence_id,
            source_field="wikipedia_sentence",
            place=place,
            cell_for_location=self._cell_for_location,
            source_url=_wikipedia_url(row),
        )

    def _contextual_sentence(self, row: Row) -> str | None:
        if not _is_contextual_english_wikipedia_sentence(row, self.max_text_characters):
            return None
        return _bounded_sentence(row.get("text"), self.max_text_characters)


class WebsiteSentenceSource:
    """Adapt pre-split website records using only upstream English metadata."""

    def __init__(
        self,
        *,
        row_shards_loader: ShardLoader,
        cell_for_location: CellForLocation,
        max_rows_per_shard: int = _DEFAULT_MAX_ROWS_PER_SHARD,
        min_language_probability: float = 0.90,
        max_text_characters: int = _DEFAULT_MAX_TEXT_CHARACTERS,
        max_stream_workers: int = _DEFAULT_MAX_STREAM_WORKERS,
    ) -> None:
        self._row_shards_loader = row_shards_loader
        self._cell_for_location = cell_for_location
        validate_positive_limit(max_rows_per_shard, "max_rows_per_shard")
        self.max_rows_per_shard = max_rows_per_shard
        self.min_language_probability = _validate_probability(
            min_language_probability, "min_language_probability"
        )
        validate_positive_limit(max_text_characters, "max_text_characters")
        self.max_text_characters = max_text_characters
        validate_positive_limit(max_stream_workers, "max_stream_workers")
        self.max_stream_workers = max_stream_workers

    def iter_candidates(self) -> Iterator[Candidate]:
        rows = _parallel_shard_rows(
            self._row_shards_loader(), self.max_rows_per_shard, self.max_stream_workers
        )
        for row in rows:
            yield from self._row_candidates(row)

    def _row_candidates(self, row: Row) -> Iterator[Candidate]:
        location = _location_from_row(row)
        polygon_id = _optional_text(row.get("polygon_id"))
        if location is None or polygon_id is None:
            return
        for field in _WEBSITE_FIELDS:
            yield from self._field_candidates(row, field, polygon_id, location)

    def _field_candidates(
        self, row: Row, field: _WebsiteField, polygon_id: str, location: tuple[float, float]
    ) -> Iterator[Candidate]:
        if not _is_eligible_website_field(row, field, self.min_language_probability):
            return
        place = JoinedPlace(
            latitude=location[0],
            longitude=location[1],
            place_name=_optional_text(row.get("name")),
            region=_optional_text(row.get("region")),
            source_url=_optional_text(row.get(field.url)),
        )
        for sentence_index, sentence in _upstream_sentences(row, field.sentences):
            cleaned = _bounded_sentence(sentence, self.max_text_characters)
            if cleaned is not None:
                yield _candidate(
                    candidate_id=f"website:{polygon_id}:{field.sentences}:{sentence_index}",
                    sentence=cleaned,
                    source=Source.WEBSITE,
                    source_record_id=polygon_id,
                    source_field=field.sentences,
                    place=place,
                    cell_for_location=self._cell_for_location,
                )


def _candidate(
    *,
    candidate_id: str,
    sentence: str,
    source: Source,
    source_record_id: str,
    source_field: str,
    place: JoinedPlace,
    cell_for_location: CellForLocation,
    source_url: str | None = None,
) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        sentence=sentence,
        source=source,
        source_record_id=source_record_id,
        source_field=source_field,
        h3_cell=cell_for_location(place.latitude, place.longitude),
        h3_resolution=3,
        latitude=place.latitude,
        longitude=place.longitude,
        place_name=place.place_name,
        region=place.region,
        source_url=source_url if source_url is not None else place.source_url,
    )


def _join_index(
    shards: Iterable[RowStream],
    *,
    key_for_row: Callable[[Row], str | None],
    place_for_row: Callable[[Row], JoinedPlace | None],
    max_rows_per_shard: int,
    max_entries: int,
    max_workers: int = _DEFAULT_MAX_STREAM_WORKERS,
) -> dict[str, JoinedPlace]:
    """Index the lookup side of a join across every shard, first key wins.

    Both axes are bounded: at most ``max_rows_per_shard`` rows are read from a
    shard, and once ``max_entries`` keys are held no further shard is opened.
    Shards may be fetched concurrently, but they are merged in shard order so
    that "first key wins" resolves to the same place whatever the network does.
    """

    index: dict[str, JoinedPlace] = {}
    shards_read = 0
    for rows in _join_shard_rows(shards, max_rows_per_shard, max_workers):
        if len(index) >= max_entries:
            _log_full_join_index(len(index), shards_read)
            break
        shards_read += 1
        _index_shard(
            index,
            rows,
            key_for_row=key_for_row,
            place_for_row=place_for_row,
            max_entries=max_entries,
        )
    logger.info("V3 join indexed %d join keys from %d shard(s)", len(index), shards_read)
    return index


def _join_shard_rows(
    shards: Iterable[RowStream],
    max_rows_per_shard: int,
    max_workers: int,
) -> Iterator[Iterable[Row]]:
    """Yield each shard's bounded rows, prefetching only when workers are enabled.

    The sequential path stays lazy so that a full join index never opens the
    next shard. Prefetching reads at most ``max_workers`` shards ahead, which is
    the cost of overlapping the network waits.
    """

    if max_workers <= 1:
        for rows in shards:
            yield _bounded_rows(rows, max_rows_per_shard)
        return
    yield from _ordered_parallel_shards(
        shards,
        lambda rows: tuple(_bounded_rows(rows, max_rows_per_shard)),
        max_workers,
    )


def _log_full_join_index(entries: int, shards_read: int) -> None:
    logger.warning(
        "V3 join index is full at %d keys after %d shard(s); stopping before shard %d. "
        "Raise max_join_entries to keep indexing, or expect unmatched sentences beyond it.",
        entries,
        shards_read,
        shards_read + 1,
    )


def _index_shard(
    index: dict[str, JoinedPlace],
    rows: Iterable[Row],
    *,
    key_for_row: Callable[[Row], str | None],
    place_for_row: Callable[[Row], JoinedPlace | None],
    max_entries: int,
) -> None:
    for row in rows:
        _add_join_entry(index, row, key_for_row=key_for_row, place_for_row=place_for_row)
        if len(index) >= max_entries:
            return


def _add_join_entry(
    index: dict[str, JoinedPlace],
    row: Row,
    *,
    key_for_row: Callable[[Row], str | None],
    place_for_row: Callable[[Row], JoinedPlace | None],
) -> None:
    key = key_for_row(row)
    if key is None or key in index:
        return
    place = place_for_row(row)
    if place is not None:
        index[key] = place


def _joined_place(index: Mapping[str, JoinedPlace], key: str | None) -> JoinedPlace | None:
    return None if key is None else index.get(key)


class _ShardFailure:
    """Carry a worker thread's exception back to the consuming generator."""

    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error


class _ShardDone:
    """Mark the end of one shard's rows on the queue."""

    __slots__ = ()


_SHARD_DONE = _ShardDone()


def _parallel_shard_rows(
    shards: Iterable[RowStream],
    limit: int,
    max_workers: int,
) -> Iterator[Row]:
    """Read shards concurrently and yield their rows as they arrive.

    The pool that consumes these rows keeps a deterministic winner per stratum by
    hash rank, not by arrival order, so interleaving shards cannot change which
    candidates survive. Memory stays bounded by a fixed queue, and a shard that
    raises re-raises here rather than being silently dropped.
    """

    streams = tuple(shards)
    if max_workers <= 1 or len(streams) <= 1:
        yield from _bounded_shard_rows(streams, limit)
        return
    # The four lines below carry `no mutate` because a mutant of any of them deadlocks the
    # suite rather than failing it: a queue that cannot accept a put, a marker the consumer does
    # not recognise, a callback that never runs, or a get that never happens all leave the
    # consumer waiting on a message no longer coming. A hung test reports as a timeout, which
    # says nothing, so the properties are pinned by assertion instead -- see
    # TestShardReadingIsBounded in tests/unit/sources/test_v3.py, which checks the queue bound,
    # the pool's size and name, and that every row of every shard arrives exactly once.
    queue: Queue[Any] = Queue(maxsize=_SHARD_QUEUE_CAPACITY)  # pragma: no mutate
    finished = Event()

    def drain(rows: RowStream) -> None:
        for row in takewhile(lambda _row: not finished.is_set(), islice(rows, limit)):
            queue.put(row)

    def announce(shard: Future[None]) -> None:
        """Report one shard's outcome from the submitting side, not from inside the worker.

        A worker that never starts -- because the pool rejected it, or it died before its first
        statement -- cannot announce itself, and a consumer waiting on a sentinel that a dead
        worker owed it waits forever. The callback runs whatever the worker did, so every shard
        is accounted for exactly once.
        """

        error = shard.exception()
        if error is not None:
            queue.put(_ShardFailure(error))
        queue.put(_SHARD_DONE)  # pragma: no mutate

    executor = ThreadPoolExecutor(
        max_workers=min(len(streams), max_workers),
        thread_name_prefix="v3-shard",
    )
    try:
        for rows in streams:
            executor.submit(drain, rows).add_done_callback(announce)  # pragma: no mutate
        yield from _consume_shard_queue(queue, len(streams))
    finally:
        finished.set()
        _drain_queue(queue)
        executor.shutdown(wait=True)


def _consume_shard_queue(queue: Queue[Any], shards: int) -> Iterator[Row]:
    """Yield rows until every shard has signalled that it finished.

    Exactly one ``_SHARD_DONE`` marker arrives per shard, so the consumer reads that many
    shards' worth of rows and stops, without joining the pool. A ``_ShardFailure`` re-raises
    here rather than in the worker thread, where it would be swallowed and the pool would
    simply come up short.
    """

    for _shard in range(shards):
        while True:
            item = queue.get()  # pragma: no mutate
            if isinstance(item, _ShardDone):
                break
            if isinstance(item, _ShardFailure):
                raise item.error
            yield item


def _drain_queue(queue: Queue[Any]) -> None:
    """Unblock workers parked on a full queue so shutdown cannot deadlock."""

    while not queue.empty():
        queue.get_nowait()


def _ordered_parallel_shards(
    shards: Iterable[RowStream],
    build: Callable[[RowStream], Any],
    max_workers: int,
) -> Iterator[Any]:
    """Prefetch shards concurrently but hand results back in shard order.

    The join index keeps the first value seen for a key, so its result depends on
    shard order. Prefetching with a bounded sliding window keeps that order while
    still overlapping the network waits.
    """

    streams = iter(shards)
    if max_workers <= 1:
        for rows in streams:
            yield build(rows)
        return
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="v3-join") as executor:
        yield from _sliding_window(streams, build, executor, max_workers)


def _sliding_window(
    streams: Iterator[RowStream],
    build: Callable[[RowStream], Any],
    executor: ThreadPoolExecutor,
    width: int,
) -> Iterator[Any]:
    """Keep ``width`` shards in flight, yielding each in submission order.

    Popping the head before submitting the next keeps the window from growing, and yielding
    ``head.result()`` rather than whichever future finished first is what preserves shard order --
    the property the join index depends on.
    """

    pending: list[Future[Any]] = [executor.submit(build, rows) for rows in islice(streams, width)]
    while pending:
        head = pending.pop(0)
        next_rows = next(streams, None)
        if next_rows is not None:
            pending.append(executor.submit(build, next_rows))
        yield head.result()


def _bounded_shard_rows(shards: Iterable[RowStream], limit: int) -> Iterator[Row]:
    for rows in shards:
        yield from _bounded_rows(rows, limit)


def _bounded_rows(rows: RowStream, limit: int) -> Iterator[Row]:
    yield from islice(rows, limit)


def _description_place(row: Row) -> JoinedPlace | None:
    identity = _description_join_key(row)
    location = _location_from_row(row)
    if identity is None or location is None:
        return None
    latitude, longitude = location
    return JoinedPlace(
        latitude=latitude,
        longitude=longitude,
        place_name=_optional_text(row.get("name")),
        region=region_from_source_pbf(row.get("source_pbf")),
        source_url=_osm_url(row),
    )


def _wikipedia_polygon_place(row: Row) -> JoinedPlace | None:
    if row.get("has_english_wikipedia") is False:
        return None
    wikidata = _wikidata_key(row)
    location = _location_from_row(row)
    if wikidata is None or location is None:
        return None
    latitude, longitude = location
    return JoinedPlace(
        latitude=latitude,
        longitude=longitude,
        record_id=_optional_text(row.get("polygon_id")) or wikidata,
        place_name=_optional_text(row.get("name")),
        region=_optional_text(row.get("region")),
    )


def _description_join_key(row: Row) -> str | None:
    source_pbf = _optional_text(row.get("source_pbf"))
    osm_type = _optional_text(row.get("osm_type"))
    osm_id = _optional_text(row.get("osm_id"))
    if source_pbf is not None and osm_type is not None and osm_id is not None:
        return f"{source_pbf}|{osm_type}|{osm_id}"
    return _optional_text(row.get("description_identity"))


def _is_english_description(row: Row, min_score: float) -> bool:
    code = _normalise_code(row.get("language_code"))
    if code not in _ENGLISH_CODES:
        return False
    score = _number(row.get("top_score"))
    return score is not None and score >= min_score


def _is_contextual_english_wikipedia_sentence(row: Row, max_text_characters: int) -> bool:
    """Keep only body sentences of an English Wikipedia article.

    The pinned revision publishes no ``is_lead`` or ``is_title`` column, so the
    rule is built on the columns it does publish, and on what they hold in the
    recorded schema fixture (``tests/fixtures/v3_upstream_schema.json``):

    * the lead is ``section_index`` 0, and every lead row there carries an empty
      ``heading`` at ``level`` 0 while no body row has an empty heading;
    * a section title reaches the sentence table as the first sentence of a body
      section, rendered with the MediaWiki edit marker (``[ edit ]`` or
      ``[ edit | edit source ]``). In the recorded shard every one of the 592
      body-section first sentences starts with that marker and the marker never
      appears anywhere else, so rejecting it drops heading lines only.

    ``sentence_index`` alone never decides this: a body sentence that is not a
    heading line is kept whatever its index.
    """

    if not _is_english_wikipedia_row(row):
        return False
    if _is_lead_section(row) or _is_section_heading_line(row):
        return False
    return _bounded_sentence(row.get("text"), max_text_characters) is not None


def _is_english_wikipedia_row(row: Row) -> bool:
    return row.get("project") == "wikipedia" and row.get("language") == "en"


def _is_lead_section(row: Row) -> bool:
    return _integer(row.get("section_index")) == 0 or _optional_text(row.get("heading")) is None


def _is_section_heading_line(row: Row) -> bool:
    if _integer(row.get("sentence_index")) != 0:
        return False
    text = _optional_text(row.get("text"))
    return text is not None and _SECTION_HEADING_LINE.match(text) is not None


def _is_eligible_website_field(row: Row, field: _WebsiteField, min_probability: float) -> bool:
    if not _is_english_code(row.get(field.language)):
        return False
    probability = _number(row.get(field.probability))
    if probability is None or probability < min_probability:
        return False
    status = row.get(field.status)
    return status is None or status == "success"


def _is_english_code(value: Any) -> bool:
    code = _normalise_code(value)
    return code in _ENGLISH_CODES or (code is not None and code.startswith("eng_"))


def _upstream_sentences(row: Row, field: str = "sentences") -> Iterator[tuple[int, str]]:
    for index, value in enumerate(_sentence_values(row, field)):
        if isinstance(value, str):
            yield index, value


def _sentence_values(row: Row, field: str) -> Iterable[Any]:
    values = row.get(field)
    if values is None and field == "sentences":
        values = row.get("sentence")
    return _sentence_sequence(values)


def _sentence_sequence(values: Any) -> Iterable[Any]:
    if isinstance(values, str):
        return (values,)
    return values if isinstance(values, Iterable) else ()


def _location_from_row(row: Row) -> tuple[float, float] | None:
    direct = _coordinate_pair(row.get("lat"), row.get("lon"))
    if direct is not None:
        return direct
    minimum = _coordinate_pair(row.get("bbox_min_y"), row.get("bbox_min_x"))
    maximum = _coordinate_pair(row.get("bbox_max_y"), row.get("bbox_max_x"))
    if minimum is None or maximum is None:
        return None
    latitude = (minimum[0] + maximum[0]) / 2
    longitude = (minimum[1] + maximum[1]) / 2
    return _coordinate_pair(latitude, longitude)


def _coordinate_pair(latitude: Any, longitude: Any) -> tuple[float, float] | None:
    latitude_number = _number(latitude)
    longitude_number = _number(longitude)
    if latitude_number is None or longitude_number is None:
        return None
    if not -90 <= latitude_number <= 90 or not -180 <= longitude_number <= 180:
        return None
    return latitude_number, longitude_number


def _osm_url(row: Row) -> str | None:
    explicit = _optional_text(row.get("osm_url"))
    if explicit is not None:
        return explicit
    osm_type = _optional_text(row.get("osm_type"))
    osm_id = _optional_text(row.get("osm_id"))
    if osm_type is None or osm_id is None:
        return None
    return f"https://www.openstreetmap.org/{osm_type}/{osm_id}"


def _wikipedia_url(row: Row) -> str | None:
    """Derive the article URL; the pinned revision publishes no URL column.

    Only ``language == "en"`` rows reach here, so the English site is the right
    host for every candidate this adapter emits.
    """

    page_id = _optional_text(row.get("page_id"))
    if page_id is None:
        return None
    return f"https://en.wikipedia.org/?curid={page_id}"


def _wikipedia_sentence_id(row: Row) -> str | None:
    explicit = _optional_text(row.get("sentence_id"))
    if explicit is not None:
        return explicit
    document_id = _optional_text(row.get("document_id"))
    section_id = _optional_text(row.get("section_id"))
    sentence_index = _integer(row.get("sentence_index"))
    if document_id is None or section_id is None or sentence_index is None:
        return None
    return f"{document_id}:{section_id}:{sentence_index}"


def _wikidata_key(row: Row) -> str | None:
    return _optional_text(row.get("wikidata"))


def _bounded_sentence(value: Any, max_characters: int) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or len(cleaned) > max_characters:
        return None
    return cleaned


def region_from_source_pbf(value: Any) -> str | None:
    """Derive the region label the sibling datasets publish from an extract name.

    The description dataset publishes no ``region`` column, but it does publish
    ``source_pbf``. In the recorded schema fixture the Wikipedia and website
    polygons of the same extract carry ``region`` "afghanistan" while their
    ``source_pbf`` is "afghanistan-latest.osm.pbf", so stripping the extract
    suffixes reproduces the published label rather than inventing one.
    """

    name = _optional_text(value)
    if name is None:
        return None
    for suffix in _EXTRACT_SUFFIXES:
        name = name.removesuffix(suffix)
    return _optional_text(name)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalise_code(value: Any) -> str | None:
    text = _optional_text(value)
    return text.lower() if text is not None else None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_probability(value: float, name: str) -> float:
    number = _number(value)
    if number is None or not 0 <= number <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return number
