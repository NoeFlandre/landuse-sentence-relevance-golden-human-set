from __future__ import annotations

from dataclasses import replace

import pytest
from tests.builders import make_candidate

from landuse_sentence_relevance.domain.models import Annotation, Candidate, Label, Source
from landuse_sentence_relevance.domain.profile import V3_SOURCES, balanced_quotas
from landuse_sentence_relevance.domain.v3_annotation import (
    V3AnnotationSeed,
    V3SeedRow,
    V3SelectionMetadata,
)
from landuse_sentence_relevance.domain.v3_progress import (
    V3AnnotationProgressError,
    inspect_v3_session,
    next_v3_candidate,
    ordered_v3_annotations,
    summarize_v3_progress,
)


def _candidate(candidate_id: str, source: Source, cell: str) -> Candidate:
    return replace(
        make_candidate(candidate_id),
        sentence=f"Sentence for {candidate_id}.",
        source=source,
        h3_cell=cell,
    )


def _seed() -> V3AnnotationSeed:
    quotas = balanced_quotas(V3_SOURCES, rows_per_source_label=1)
    rows: list[V3SeedRow] = []
    for index, (source, label) in enumerate(key for key in quotas.counts if key[0] in V3_SOURCES):
        candidate = _candidate(
            f"{source.value}-{label.value}",
            source,
            f"{source.value}-{label.value}-cell",
        )
        annotation = (
            Annotation(candidate, label) if source is Source.WIKIPEDIA and label is Label.YES else None
        )
        rows.append(
            V3SeedRow(
                candidate=candidate,
                quota_source=source,
                quota_label=label,
                origin="v2" if annotation is not None else "v3",
                annotation=annotation,
                selection=V3SelectionMetadata(seed="progress-test", rank="0" * 64, slot_index=index),
            )
        )
    return V3AnnotationSeed(
        rows=tuple(rows),
        excluded_v2_rows=(),
        reserved_v2_cells=frozenset({"wikipedia-yes-cell"}),
        quotas=quotas,
        benchmark_sha256="0" * 64,
        seed="progress-test",
    )


def _seed_with_reserve() -> V3AnnotationSeed:
    """One pending WIKIPEDIA/YES slot, plus a reserve candidate for the same source."""

    quotas = balanced_quotas((Source.WIKIPEDIA,), rows_per_source_label=1)
    pending = _candidate("wiki-pending", Source.WIKIPEDIA, "wiki-pending-cell")
    seeded = _candidate("wiki-seeded", Source.WIKIPEDIA, "wiki-seeded-cell")
    rows = (
        V3SeedRow(
            candidate=pending,
            quota_source=Source.WIKIPEDIA,
            quota_label=Label.YES,
            origin="v3",
            annotation=None,
            selection=V3SelectionMetadata(seed="progress-test", rank="0" * 64, slot_index=0),
        ),
        V3SeedRow(
            candidate=seeded,
            quota_source=Source.WIKIPEDIA,
            quota_label=Label.NO,
            origin="v2",
            annotation=Annotation(seeded, Label.NO),
            selection=V3SelectionMetadata(seed="progress-test", rank="1" * 64, slot_index=1),
        ),
    )
    return V3AnnotationSeed(
        rows=rows,
        excluded_v2_rows=(),
        reserved_v2_cells=frozenset({"wiki-seeded-cell"}),
        quotas=quotas,
        benchmark_sha256="0" * 64,
        seed="progress-test",
        reserve_candidates=(_candidate("wiki-reserve", Source.WIKIPEDIA, "wiki-reserve-cell"),),
    )


def test_progress_counts_seeded_and_fresh_labels_by_source() -> None:
    seed = _seed()
    fresh = {
        "wikipedia-no": Annotation(seed.rows[1].candidate, Label.NO),
        "website-yes": Annotation(seed.rows[2].candidate, Label.YES),
    }

    progress = summarize_v3_progress(seed, fresh)

    assert progress.total_count == 6
    assert progress.seeded_count == 1
    assert progress.fresh_labeled_count == 2
    assert progress.labeled_count == 3
    assert progress.yes_count == 2
    assert progress.no_count == 1
    assert progress.target_yes_count == 3
    assert progress.target_no_count == 3
    assert progress.target_count == 6
    assert progress.remaining_quotas == {
        (Source.WIKIPEDIA, Label.YES): 0,
        (Source.WIKIPEDIA, Label.NO): 0,
        (Source.WEBSITE, Label.YES): 0,
        (Source.WEBSITE, Label.NO): 1,
        (Source.DESCRIPTION, Label.YES): 1,
        (Source.DESCRIPTION, Label.NO): 1,
    }
    assert progress.source_progress[0].labeled_count == 2
    assert progress.source_progress[0].source is Source.WIKIPEDIA
    assert progress.source_progress[0].yes_count == 1
    assert progress.source_progress[0].no_count == 1
    assert progress.source_progress[0].target_yes_count == 1
    assert progress.source_progress[0].target_no_count == 1
    assert progress.source_progress[0].remaining_yes_count == 0
    assert progress.source_progress[0].remaining_no_count == 0
    assert progress.source_progress[0].remaining_count == 0
    assert progress.source_progress[1].source is Source.WEBSITE
    assert progress.source_progress[1].labeled_count == 1
    assert progress.source_progress[1].target_yes_count == 1
    assert progress.source_progress[1].target_no_count == 1
    assert progress.source_progress[1].remaining_yes_count == 0
    assert progress.source_progress[1].remaining_no_count == 1


def test_ordered_annotations_returns_fresh_labels_in_seed_order() -> None:
    seed = _seed()
    second = seed.pending_rows[1].candidate
    first = seed.pending_rows[0].candidate
    annotations = {
        second.candidate_id: Annotation(second, Label.YES),
        first.candidate_id: Annotation(first, Label.NO),
    }

    assert ordered_v3_annotations(seed, annotations) == (
        annotations[first.candidate_id],
        annotations[second.candidate_id],
    )


def test_next_candidate_skips_saved_fresh_rows_and_never_returns_seeded_v2() -> None:
    seed = _seed()
    first_pending = seed.rows[1].candidate
    saved = {first_pending.candidate_id: Annotation(first_pending, Label.NO)}

    assert next_v3_candidate(seed, saved) == seed.rows[2].candidate


def test_next_candidate_returns_the_first_pending_row_before_quota_accounting() -> None:
    seed = _seed()

    assert next_v3_candidate(seed, {}) == seed.pending_rows[0].candidate


def test_next_candidate_returns_none_after_all_fresh_rows_are_labeled() -> None:
    seed = _seed()
    saved = {row.candidate.candidate_id: Annotation(row.candidate, Label.NO) for row in seed.pending_rows}

    assert next_v3_candidate(seed, saved) is None


@pytest.mark.parametrize(
    ("candidate_id", "message"),
    [
        ("wikipedia-yes", "seeded V2 rows are immutable and cannot be reannotated"),
        ("unknown", "V3 session contains an unknown candidate"),
    ],
)
def test_progress_rejects_seeded_or_unknown_session_rows(
    candidate_id: str,
    message: str,
) -> None:
    seed = _seed()
    candidate = (
        seed.rows[0].candidate
        if candidate_id == "wikipedia-yes"
        else _candidate(candidate_id, Source.WIKIPEDIA, "unknown-cell")
    )

    with pytest.raises(V3AnnotationProgressError) as error:
        summarize_v3_progress(seed, {candidate_id: Annotation(candidate, Label.YES)})
    assert str(error.value) == message


def test_progress_rejects_a_session_key_that_does_not_match_its_candidate() -> None:
    seed = _seed()
    candidate = seed.rows[1].candidate

    with pytest.raises(V3AnnotationProgressError) as error:
        summarize_v3_progress(seed, {"wrong-key": Annotation(candidate, Label.NO)})

    assert str(error.value) == "V3 session key does not match its candidate ID"


def test_progress_rejects_a_session_row_with_changed_candidate_content() -> None:
    seed = _seed()
    candidate = replace(seed.rows[1].candidate, sentence="Tampered sentence.")

    with pytest.raises(V3AnnotationProgressError) as error:
        summarize_v3_progress(seed, {candidate.candidate_id: Annotation(candidate, Label.NO)})
    assert str(error.value) == "V3 session candidate content does not match the seed"


def test_a_fresh_row_labelled_against_its_slot_is_replaced_from_the_reserve() -> None:
    """A quota slot is filled by the label a human gives, not by the one it hoped for.

    The pool is oversized precisely so a "yes" slot answered "no" can be offered
    another candidate. Without a reserve the session simply runs out while the
    quota is still short, which is what stranded a real run at 33 yes / 67 no on
    the website source.
    """

    seed = _seed_with_reserve()
    pending = seed.pending_rows[0].candidate
    # The slot wanted YES; the human says NO, so the YES quota is still unfilled.
    annotations = {pending.candidate_id: Annotation(candidate=pending, label=Label.NO)}

    nxt = next_v3_candidate(seed, annotations)

    assert nxt is not None, "the reserve must offer another candidate for the unfilled slot"
    assert nxt.candidate_id != pending.candidate_id
    assert nxt.source is pending.source


def test_the_reserve_stops_once_every_quota_is_filled() -> None:
    seed = _seed_with_reserve()
    pending = seed.pending_rows[0].candidate
    annotations = {pending.candidate_id: Annotation(candidate=pending, label=Label.YES)}

    assert next_v3_candidate(seed, annotations) is None


def test_the_reserve_only_serves_sources_with_an_unfilled_quota() -> None:
    quotas = balanced_quotas((Source.WIKIPEDIA, Source.WEBSITE), rows_per_source_label=1)
    rows = tuple(
        V3SeedRow(
            candidate=_candidate(f"{source.value}-{label.value}", source, f"{source.value}-{label.value}"),
            quota_source=source,
            quota_label=label,
            origin="v3",
            annotation=None,
            selection=V3SelectionMetadata(seed="progress-test", rank=f"{index:064d}", slot_index=0),
        )
        for index, (source, label) in enumerate(quotas.counts)
    )
    wiki_reserve = _candidate("wiki-reserve", Source.WIKIPEDIA, "wiki-reserve-cell")
    website_reserve = _candidate("website-reserve", Source.WEBSITE, "website-reserve-cell")
    seed = V3AnnotationSeed(
        rows=rows,
        excluded_v2_rows=(),
        reserved_v2_cells=frozenset(),
        quotas=quotas,
        benchmark_sha256="0" * 64,
        seed="progress-test",
        reserve_candidates=(website_reserve, wiki_reserve),
    )
    labels = {
        Source.WIKIPEDIA: {Label.YES: Label.NO, Label.NO: Label.NO},
        Source.WEBSITE: {Label.YES: Label.YES, Label.NO: Label.NO},
    }
    annotations = {
        row.candidate.candidate_id: Annotation(row.candidate, labels[row.quota_source][row.quota_label])
        for row in rows
    }

    assert next_v3_candidate(seed, annotations) == wiki_reserve


def test_session_snapshot_reuses_one_ordered_view_for_progress_and_next_candidate() -> None:
    seed = _seed_with_reserve()
    pending = seed.pending_rows[0].candidate
    annotations = {pending.candidate_id: Annotation(candidate=pending, label=Label.NO)}

    snapshot = inspect_v3_session(seed, annotations)

    assert snapshot.annotations == (annotations[pending.candidate_id],)
    assert snapshot.progress.fresh_labeled_count == 1
    assert snapshot.progress.remaining_quotas[(Source.WIKIPEDIA, Label.YES)] == 1
    assert snapshot.current_candidate == seed.reserve_candidates[0]


def test_session_snapshot_starts_with_the_first_pending_candidate() -> None:
    seed = _seed()

    snapshot = inspect_v3_session(seed, {})

    assert snapshot.current_candidate == seed.pending_rows[0].candidate
