"""Build website candidates from already eligible geolocated rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from landuse_sentence_relevance.domain.cell_quota import CellQuota
from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.sentence_selection import (
    first_prioritized_sentence,
    prioritize_sentences,
)
from landuse_sentence_relevance.sources.protocols import (
    BatchLanguageIdentifier,
    LanguageIdentifier,
    SentenceSplitter,
    split_sentences_many,
)
from landuse_sentence_relevance.sources.website_text import (
    WEBSITE_FIELD_SPECS,
    FieldSpec,
    Location,
    _bounded_text,
    _candidate_part_groups,
    _field_text,
    _field_url,
    _FieldWork,
    _is_contextual_text,
    _location_from_row,
    _pending_field_indexes,
    _select_batch_parts,
)

EligibleRow = tuple[Mapping[str, Any], str]


class WebsiteCandidateBuilder:
    """Turn eligible website rows into deterministic English candidates."""

    _FIELD_SPECS = WEBSITE_FIELD_SPECS

    def __init__(
        self,
        splitter: SentenceSplitter,
        language_identifier: LanguageIdentifier,
        seed: str,
        candidate_quota: CellQuota | None,
        max_text_characters: int | None,
        require_paragraph: bool,
    ) -> None:
        self._splitter = splitter
        self._language_identifier = language_identifier
        self._seed = seed
        self._candidate_quota = candidate_quota
        self._max_text_characters = max_text_characters
        self._require_paragraph = require_paragraph

    def has_eligible_text(self, row: Mapping[str, Any]) -> bool:
        return any(
            text is not None and self._is_eligible_text(_bounded_text(text, self._max_text_characters))
            for text_field, _ in self._FIELD_SPECS
            if (text := _field_text(row, text_field)) is not None
        )

    def candidates_for_row(self, row: Mapping[str, Any], cell: str) -> Iterable[Candidate]:
        location = _location_from_row(row)
        if location is None:
            return
        polygon_id, latitude, longitude = location
        for text_field, url_field in self._FIELD_SPECS:
            yield from self._field_candidates(
                row,
                polygon_id,
                latitude,
                longitude,
                cell,
                text_field,
                url_field,
            )

    def candidates_for_rows(
        self,
        rows: tuple[EligibleRow, ...],
    ) -> tuple[tuple[Candidate, ...], ...]:
        if self._candidate_quota is not None and self._candidate_quota.capacity == 1:
            return self._single_candidate_field_batch(rows)
        return self._all_field_candidates_batch(rows)

    def _all_field_candidates_batch(
        self,
        rows: tuple[EligibleRow, ...],
    ) -> tuple[tuple[Candidate, ...], ...]:
        work = self._field_work_for_rows(rows)
        sentence_groups = split_sentences_many(self._splitter, (field.text for field in work))
        return self._group_row_candidates(rows, work, sentence_groups)

    def _single_candidate_field_batch(
        self,
        rows: tuple[EligibleRow, ...],
    ) -> tuple[tuple[Candidate, ...], ...]:
        candidates: list[list[Candidate]] = [[] for _ in rows]
        pending_indexes = tuple(range(len(rows)))
        for field_spec in self._FIELD_SPECS:
            if not pending_indexes:
                break
            pending_indexes = self._field_batch_round(rows, pending_indexes, field_spec, candidates)
        return tuple(tuple(row_candidates) for row_candidates in candidates)

    def _field_batch_round(
        self,
        rows: tuple[EligibleRow, ...],
        pending_indexes: tuple[int, ...],
        field_spec: FieldSpec,
        candidates: list[list[Candidate]],
    ) -> tuple[int, ...]:
        work = self._field_batch_work(rows, pending_indexes, field_spec)
        if not work:
            return pending_indexes
        sentence_groups = split_sentences_many(self._splitter, (field.text for field in work))
        self._append_batch_field_candidates(candidates, work, sentence_groups)
        return _pending_field_indexes(pending_indexes, candidates)

    def _field_batch_work(
        self,
        rows: tuple[EligibleRow, ...],
        pending_indexes: tuple[int, ...],
        field_spec: FieldSpec,
    ) -> tuple[_FieldWork, ...]:
        pending_rows = tuple(rows[index] for index in pending_indexes)
        return self._field_work_for_rows(pending_rows, (field_spec,), pending_indexes)

    def _append_batch_field_candidates(
        self,
        candidates: list[list[Candidate]],
        work: tuple[_FieldWork, ...],
        sentence_groups: Iterable[Iterable[str]],
    ) -> None:
        language_identifier = self._language_identifier
        if isinstance(language_identifier, BatchLanguageIdentifier):
            self._append_batch_language_candidates(
                candidates,
                work,
                sentence_groups,
                language_identifier,
            )
            return
        self._append_scalar_field_candidates(candidates, work, sentence_groups)

    def _append_scalar_field_candidates(
        self,
        candidates: list[list[Candidate]],
        work: tuple[_FieldWork, ...],
        sentence_groups: Iterable[Iterable[str]],
    ) -> None:
        for field, sentences in zip(work, sentence_groups, strict=True):
            candidate = self._first_field_candidate(field, sentences)
            if candidate is not None:
                candidates[field.row_index].append(candidate)

    def _append_batch_language_candidates(
        self,
        candidates: list[list[Candidate]],
        work: tuple[_FieldWork, ...],
        sentence_groups: Iterable[Iterable[str]],
        language_identifier: BatchLanguageIdentifier,
    ) -> None:
        part_groups = _candidate_part_groups(work, sentence_groups, self._seed)
        selected_parts = _select_batch_parts(part_groups, language_identifier)
        for field, part in zip(work, selected_parts, strict=True):
            if part is not None:
                candidate = self._candidate_for_sentence(
                    part.text,
                    part.index,
                    field.polygon_id,
                    field.latitude,
                    field.longitude,
                    field.cell,
                    field.text_field,
                    field.website_url,
                    field.row,
                )
                if candidate is not None:
                    candidates[field.row_index].append(candidate)

    def _first_field_candidate(
        self,
        field: _FieldWork,
        sentences: Iterable[str],
    ) -> Candidate | None:
        part = first_prioritized_sentence(
            sentences,
            seed=f"{self._seed}:{field.polygon_id}:{field.text_field}",
            accept=self._language_identifier.is_english,
        )
        if part is None:
            return None
        return self._candidate_for_sentence(
            part.text,
            part.index,
            field.polygon_id,
            field.latitude,
            field.longitude,
            field.cell,
            field.text_field,
            field.website_url,
            field.row,
        )

    def _field_work_for_rows(
        self,
        rows: tuple[EligibleRow, ...],
        field_specs: tuple[FieldSpec, ...] | None = None,
        row_indexes: tuple[int, ...] | None = None,
    ) -> tuple[_FieldWork, ...]:
        work: list[_FieldWork] = []
        indexes = tuple(range(len(rows))) if row_indexes is None else row_indexes
        specs = self._FIELD_SPECS if field_specs is None else field_specs
        for row_index, (row, cell) in zip(indexes, rows, strict=True):
            work.extend(self._field_work(row_index, row, cell, specs))
        return tuple(work)

    def _group_row_candidates(
        self,
        rows: tuple[EligibleRow, ...],
        work: tuple[_FieldWork, ...],
        sentence_groups: Iterable[Iterable[str]],
    ) -> tuple[tuple[Candidate, ...], ...]:
        candidates: list[list[Candidate]] = [[] for _ in rows]
        for field, sentences in zip(work, sentence_groups, strict=True):
            candidates[field.row_index].extend(self._field_candidates_for_work(field, sentences))
        return tuple(tuple(row_candidates) for row_candidates in candidates)

    def _field_candidates_for_work(
        self,
        field: _FieldWork,
        sentences: Iterable[str],
    ) -> Iterable[Candidate]:
        return self._field_candidates(
            field.row,
            field.polygon_id,
            field.latitude,
            field.longitude,
            field.cell,
            field.text_field,
            field.url_field,
            text=field.text,
            sentences=sentences,
        )

    def _field_work(
        self,
        row_index: int,
        row: Mapping[str, Any],
        cell: str,
        field_specs: tuple[FieldSpec, ...] | None = None,
    ) -> tuple[_FieldWork, ...]:
        location = _location_from_row(row)
        if location is None:
            return ()
        specs = self._FIELD_SPECS if field_specs is None else field_specs
        return self._field_work_for_specs(row_index, row, cell, location, specs)

    def _field_work_for_specs(
        self,
        row_index: int,
        row: Mapping[str, Any],
        cell: str,
        location: Location,
        specs: tuple[FieldSpec, ...],
    ) -> tuple[_FieldWork, ...]:
        polygon_id, latitude, longitude = location
        work: list[_FieldWork] = []
        for text_field, url_field in specs:
            text = _field_text(row, text_field)
            if text is None:
                continue
            bounded = _bounded_text(text, self._max_text_characters)
            if not self._is_eligible_text(bounded):
                continue
            work.append(
                _FieldWork(
                    row_index=row_index,
                    row=row,
                    polygon_id=polygon_id,
                    latitude=latitude,
                    longitude=longitude,
                    cell=cell,
                    text_field=text_field,
                    url_field=url_field,
                    website_url=_field_url(row, url_field),
                    text=bounded,
                )
            )
        return tuple(work)

    def _field_candidates(
        self,
        row: Mapping[str, Any],
        polygon_id: str,
        latitude: float,
        longitude: float,
        cell: str,
        text_field: str,
        url_field: str,
        text: str | None = None,
        sentences: Iterable[str] | None = None,
    ) -> Iterable[Candidate]:
        text = _field_text(row, text_field) if text is None else text
        if text is None or not self._is_eligible_text(_bounded_text(text, self._max_text_characters)):
            return
        text = _bounded_text(text, self._max_text_characters)
        website_url = _field_url(row, url_field)
        yield from self._field_candidate_parts(
            self._sentences_for_text(text, sentences),
            seed=f"{self._seed}:{polygon_id}:{text_field}",
            polygon_id=polygon_id,
            latitude=latitude,
            longitude=longitude,
            cell=cell,
            text_field=text_field,
            website_url=website_url,
            row=row,
        )

    def _sentences_for_text(
        self,
        text: str,
        sentences: Iterable[str] | None,
    ) -> Iterable[str]:
        if sentences is not None:
            return sentences
        return self._splitter.split(text)

    def _is_eligible_text(self, text: str) -> bool:
        return not self._require_paragraph or _is_contextual_text(text)

    def _field_candidate_parts(
        self,
        sentences: Iterable[str],
        seed: str,
        polygon_id: str,
        latitude: float,
        longitude: float,
        cell: str,
        text_field: str,
        website_url: str | None,
        row: Mapping[str, Any],
    ) -> Iterable[Candidate]:
        for part in prioritize_sentences(
            sentences,
            seed=seed,
            accept=self._language_identifier.is_english,
        ):
            candidate = self._candidate_for_sentence(
                part.text,
                part.index,
                polygon_id,
                latitude,
                longitude,
                cell,
                text_field,
                website_url,
                row,
            )
            if candidate is None:
                continue
            yield candidate

    def _candidate_for_sentence(
        self,
        sentence: str,
        sentence_index: int,
        polygon_id: str,
        latitude: float,
        longitude: float,
        cell: str,
        field: str,
        website_url: str | None,
        row: Mapping[str, Any],
    ) -> Candidate | None:
        cleaned = sentence.strip()
        if not cleaned:
            return None
        return Candidate(
            candidate_id=f"website:{polygon_id}:{field}:{sentence_index}",
            sentence=cleaned,
            source=Source.WEBSITE,
            source_record_id=polygon_id,
            source_field=field,
            h3_cell=cell,
            h3_resolution=3,
            latitude=latitude,
            longitude=longitude,
            place_name=row.get("name"),
            region=row.get("region"),
            source_url=website_url,
        )
