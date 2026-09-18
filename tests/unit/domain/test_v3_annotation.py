from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from tests.builders import make_candidate

from landuse_sentence_relevance.domain.models import Annotation, Candidate, Label, Source
from landuse_sentence_relevance.domain.profile import V3_QUOTAS, SourceLabelQuotas
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.domain.seeding import SeedPlan, plan_seed
from landuse_sentence_relevance.domain.v3_annotation import (
    V3AnnotationSeed,
    V3AnnotationSeedError,
    V3SeedOrigin,
    V3SeedRow,
    V3SelectionMetadata,
    select_v3_annotation_seed,
)
from landuse_sentence_relevance.storage.v3_candidate_pool import load_v2_seed_plan

ROOT = Path(__file__).parents[3]


def _candidate(source: Source, index: int):
    return replace(
        make_candidate(f"fresh-{source.value}-{index:03d}"),
        source=source,
        h3_cell=f"fresh-{source.value}-{index:03d}",
    )


def _pool() -> FinalizedCandidatePool:
    candidates = tuple(_candidate(source, index) for source in Source for index in range(400))
    return FinalizedCandidatePool(candidates, tuple(candidate.h3_cell for candidate in candidates))


def _test_candidate(
    candidate_id: str = "candidate", source: Source = Source.WIKIPEDIA, cell: str = "cell"
) -> Candidate:
    return replace(make_candidate(candidate_id), source=source, h3_cell=cell)


def _annotation(
    candidate_id: str = "annotation",
    source: Source = Source.WIKIPEDIA,
    label: Label = Label.YES,
    cell: str = "annotation-cell",
) -> Annotation:
    return Annotation(candidate=_test_candidate(candidate_id, source, cell), label=label)


def _selection() -> V3SelectionMetadata:
    return V3SelectionMetadata(seed="test-seed", rank="0" * 64, slot_index=0)


def _pending_row(
    candidate: Candidate | None = None,
    *,
    quota_source: Source = Source.WIKIPEDIA,
    quota_label: Label = Label.YES,
) -> V3SeedRow:
    return V3SeedRow(
        candidate=candidate or _test_candidate(),
        quota_source=quota_source,
        quota_label=quota_label,
        origin="v3",
        annotation=None,
        selection=_selection(),
    )


def _seeded_row(annotation: Annotation) -> V3SeedRow:
    return V3SeedRow(
        candidate=annotation.candidate,
        quota_source=annotation.candidate.source,
        quota_label=annotation.label,
        origin="v2",
        annotation=annotation,
        selection=_selection(),
    )


def test_v3_seed_has_exact_source_label_slots_and_unlabeled_pending_rows() -> None:
    benchmark = ROOT / "data/benchmark/v2-adjudicated.csv"
    seed_plan = load_v2_seed_plan(benchmark, quotas=V3_QUOTAS, seed="landuse-v3-test")

    state = select_v3_annotation_seed(
        seed_plan,
        _pool(),
        quotas=V3_QUOTAS,
        seed="landuse-v3-test",
        benchmark_sha256="0" * 64,
    )

    assert state.total_rows == 300
    assert state.rows_by_source == {source: 100 for source in V3_QUOTAS.sources}
    assert state.quota_counts == {
        (source, label): 50 for source in V3_QUOTAS.sources for label in (Label.YES, Label.NO)
    }
    assert state.seeded_row_count == 153
    assert state.pending_row_count == 147
    assert Counter(row.candidate.source for row in state.seeded_rows) == {
        Source.WIKIPEDIA: 99,
        Source.WEBSITE: 54,
    }
    assert Counter(row.candidate.source for row in state.pending_rows) == {
        Source.WIKIPEDIA: 1,
        Source.WEBSITE: 46,
        Source.DESCRIPTION: 100,
    }
    assert all(row.annotation is None for row in state.pending_rows)
    assert all(row.origin == "v3" for row in state.pending_rows)
    assert all(row.candidate.h3_cell not in state.reserved_v2_cells for row in state.pending_rows)
    assert state.excluded_v2_reasons == {
        annotation.candidate.candidate_id: (
            "Excluded as a deterministic V3 quota surplus; the frozen V2 row remains reserved "
            "but is not part of the selected V3 quota."
        )
        for annotation in seed_plan.excluded
    }


def test_selection_records_deterministic_provenance_for_seeded_and_pending_rows() -> None:
    quotas = SourceLabelQuotas(
        {
            (Source.WIKIPEDIA, Label.YES): 1,
            (Source.WIKIPEDIA, Label.NO): 1,
        }
    )
    seeded = _annotation("seeded", label=Label.YES, cell="seeded-cell")
    fresh = _test_candidate("fresh", source=Source.WIKIPEDIA, cell="fresh-cell")
    state = select_v3_annotation_seed(
        plan_seed((seeded,), quotas, seed="stable-seed"),
        FinalizedCandidatePool((fresh,), (fresh.h3_cell,)),
        quotas=quotas,
        seed="stable-seed",
        benchmark_sha256="0" * 64,
    )

    assert state.rows[0].selection == V3SelectionMetadata(
        seed="stable-seed",
        rank=hashlib.sha256(b"stable-seed:v2:seeded").hexdigest(),
        slot_index=0,
    )
    assert state.rows[1].selection == V3SelectionMetadata(
        seed="stable-seed",
        rank=hashlib.sha256(b"stable-seed:wikipedia:no:fresh").hexdigest(),
        slot_index=0,
    )


def test_selection_never_reuses_a_fresh_candidate_across_label_slots() -> None:
    quotas = SourceLabelQuotas(
        {
            (Source.WIKIPEDIA, Label.YES): 1,
            (Source.WIKIPEDIA, Label.NO): 1,
        }
    )
    first = _test_candidate("first", cell="first-cell")
    second = _test_candidate("second", cell="second-cell")

    state = _select_with_plan(
        plan_seed((), quotas, seed="test-seed"),
        quotas,
        FinalizedCandidatePool((first, second), (first.h3_cell, second.h3_cell)),
    )

    assert len({row.candidate.candidate_id for row in state.pending_rows}) == 2


def test_pending_slots_take_the_lowest_ranked_fresh_candidates_for_each_quota_slot() -> None:
    quotas = SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 2, (Source.WIKIPEDIA, Label.NO): 2})
    candidates = tuple(_test_candidate(f"fresh-{index}", cell=f"fresh-cell-{index}") for index in range(6))

    state = _select_with_plan(
        plan_seed((), quotas, seed="test-seed"),
        quotas,
        FinalizedCandidatePool(candidates, tuple(candidate.h3_cell for candidate in candidates)),
    )

    assert _selected_ids(state, Label.YES) == ("fresh-4", "fresh-0")
    assert _selected_ids(state, Label.NO) == ("fresh-1", "fresh-2")
    assert tuple(row.selection.slot_index for row in state.pending_rows) == (0, 1, 0, 1)


def _selected_ids(state: V3AnnotationSeed, label: Label) -> tuple[str, ...]:
    return tuple(row.candidate.candidate_id for row in state.pending_rows if row.quota_label is label)


def test_v3_seed_rejects_a_non_hex_benchmark_digest() -> None:
    benchmark = ROOT / "data/benchmark/v2-adjudicated.csv"
    seed_plan = load_v2_seed_plan(benchmark, quotas=V3_QUOTAS, seed="landuse-v3-test")

    with pytest.raises(ValueError) as error:
        select_v3_annotation_seed(
            seed_plan,
            _pool(),
            quotas=V3_QUOTAS,
            seed="landuse-v3-test",
            benchmark_sha256="z" * 64,
        )
    assert str(error.value) == "benchmark SHA-256 must be a 64-character hexadecimal digest"


@pytest.mark.parametrize(
    ("seed", "rank", "slot_index", "message"),
    [
        (" ", "0" * 64, 0, "V3 selection seed must be a non-empty string"),
        ("test", "z" * 64, 0, "V3 selection rank must be a SHA-256 digest"),
        ("test", "X" * 64, 0, "V3 selection rank must be a SHA-256 digest"),
        ("test", "0" * 64, -1, "V3 selection slot index must be a non-negative integer"),
        ("test", "0" * 64, "0", "V3 selection slot index must be a non-negative integer"),
        ("test", "0" * 64, True, "V3 selection slot index must be a non-negative integer"),
    ],
)
def test_selection_metadata_rejects_invalid_provenance(
    seed: str,
    rank: str,
    slot_index: object,
    message: str,
) -> None:
    with pytest.raises(ValueError) as error:
        V3SelectionMetadata(seed=seed, rank=rank, slot_index=cast(int, slot_index))
    assert str(error.value) == message


def test_selection_metadata_accepts_uppercase_sha256_hex_digests() -> None:
    metadata = V3SelectionMetadata(seed="test", rank="A" * 64, slot_index=0)

    assert metadata.rank == "A" * 64


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"quota_source": Source.WEBSITE},
            "V3 seed row candidate source does not match quota source",
        ),
        ({"origin": cast(V3SeedOrigin, "other")}, "unknown V3 seed origin: 'other'"),
        ({"origin": "v2", "annotation": None}, "V2 seed row must carry its frozen annotation"),
        (
            {"origin": "v2", "annotation": _annotation("other", cell="other-cell")},
            "V2 seed row must carry its frozen annotation",
        ),
        (
            {"origin": "v2", "annotation": _annotation("candidate", label=Label.NO, cell="cell")},
            "V2 seed row annotation quota must match its quota slot",
        ),
        ({"annotation": _annotation()}, "fresh V3 seed row must not carry a human label"),
    ],
)
def test_seed_row_rejects_inconsistent_origin_quota_or_annotation(
    updates: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "candidate": _test_candidate(),
        "quota_source": Source.WIKIPEDIA,
        "quota_label": Label.YES,
        "origin": "v3",
        "annotation": None,
        "selection": _selection(),
    }
    values.update(updates)

    with pytest.raises(V3AnnotationSeedError) as error:
        V3SeedRow(**values)  # type: ignore[arg-type]
    assert str(error.value) == message


def test_seed_row_from_dict_requires_an_object_quota_slot() -> None:
    payload = _pending_row().to_dict()
    payload["quota_slot"] = []

    with pytest.raises(ValueError, match="quota_slot must be an object"):
        V3SeedRow.from_dict(payload)


def test_annotation_seed_from_dict_requires_an_object_quota_matrix() -> None:
    with pytest.raises(ValueError) as error:
        V3AnnotationSeed.from_dict({"quotas": []})
    assert str(error.value) == "V3 seed quotas must be an object"


def test_annotation_seed_from_dict_requires_an_object_for_each_quota_source() -> None:
    with pytest.raises(ValueError) as error:
        V3AnnotationSeed.from_dict({"quotas": {"wikipedia": []}})
    assert str(error.value) == "V3 seed quota source must be an object"


def test_annotation_seed_rejects_an_empty_seed_name() -> None:
    with pytest.raises(V3AnnotationSeedError) as error:
        V3AnnotationSeed(
            rows=(),
            excluded_v2_rows=(),
            reserved_v2_cells=frozenset(),
            quotas=SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 0}),
            benchmark_sha256="0" * 64,
            seed=" ",
        )
    assert str(error.value) == "V3 seed must be non-empty"


def test_annotation_seed_dict_round_trip_preserves_rows_and_quota_metadata() -> None:
    seeded = _annotation("seeded", label=Label.YES, cell="seeded-cell")
    excluded = _annotation("excluded", label=Label.NO, cell="excluded-cell")
    state = _state(
        (_seeded_row(seeded),),
        SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 1}),
        excluded=(excluded,),
        reserved=frozenset({"seeded-cell", "excluded-cell"}),
        reasons={"excluded": "deterministic quota surplus"},
    )

    assert V3AnnotationSeed.from_dict(state.to_dict()) == state


def _select_with_plan(
    seed_plan: SeedPlan,
    quotas: SourceLabelQuotas,
    pool: FinalizedCandidatePool | None = None,
) -> V3AnnotationSeed:
    return select_v3_annotation_seed(
        seed_plan,
        pool or FinalizedCandidatePool((), ()),
        quotas=quotas,
        seed="test-seed",
        benchmark_sha256="0" * 64,
    )


def test_selection_rejects_a_fresh_quota_shortfall() -> None:
    quotas = SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 2})
    candidate = _test_candidate()
    pool = FinalizedCandidatePool((candidate,), (candidate.h3_cell,))

    with pytest.raises(V3AnnotationSeedError, match="quota needs 2 fresh rows"):
        _select_with_plan(plan_seed((), quotas, seed="test-seed"), quotas, pool)


@pytest.mark.parametrize(
    ("seed_plan", "message"),
    [
        (
            SeedPlan(
                reused=(_annotation("same"), _annotation("same")),
                excluded=(),
                remaining={},
                reserved_cells=frozenset({"same-cell"}),
            ),
            "V2 seed rows must have unique candidate IDs and H3 cells",
        ),
        (
            SeedPlan(
                reused=(_annotation("first", cell="same-cell"), _annotation("second", cell="same-cell")),
                excluded=(),
                remaining={},
                reserved_cells=frozenset({"same-cell"}),
            ),
            "V2 seed rows must have unique candidate IDs and H3 cells",
        ),
        (
            SeedPlan(
                reused=(_annotation("reserved"),),
                excluded=(),
                remaining={},
                reserved_cells=frozenset({"not-the-cell"}),
            ),
            "V2 reserved H3 cells must match every V2 seed row",
        ),
        (
            SeedPlan(
                reused=(_annotation("website", source=Source.WEBSITE),),
                excluded=(),
                remaining={(Source.WIKIPEDIA, Label.YES): 1},
                reserved_cells=frozenset({"annotation-cell"}),
            ),
            "V2 seed contains a source/label pair outside the V3 quota matrix",
        ),
        (
            SeedPlan(
                reused=(_annotation("one"), _annotation("two", cell="two-cell")),
                excluded=(),
                remaining={},
                reserved_cells=frozenset({"annotation-cell", "two-cell"}),
            ),
            "V2 reused rows exceed a V3 quota",
        ),
        (
            SeedPlan(
                reused=(_annotation("shortfall"),),
                excluded=(),
                remaining={},
                reserved_cells=frozenset({"annotation-cell"}),
            ),
            "V2 seed shortfall does not match the quota matrix",
        ),
    ],
)
def test_selection_rejects_an_invalid_v2_seed_plan(seed_plan: SeedPlan, message: str) -> None:
    quotas = SourceLabelQuotas(
        {
            (Source.WIKIPEDIA, Label.YES): 1,
            (Source.WIKIPEDIA, Label.NO): 1,
        }
    )

    with pytest.raises(V3AnnotationSeedError) as error:
        _select_with_plan(seed_plan, quotas)
    assert str(error.value) == message


def test_selection_rejects_candidates_outside_the_quota_sources() -> None:
    quotas = SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 1})
    candidate = _test_candidate(source=Source.WEBSITE)

    with pytest.raises(V3AnnotationSeedError) as error:
        _select_with_plan(
            plan_seed((), quotas, seed="test-seed"),
            quotas,
            FinalizedCandidatePool((candidate,), (candidate.h3_cell,)),
        )
    assert str(error.value) == "candidate pool contains sources outside the V3 quota matrix: ['website']"


def test_selection_rejects_a_candidate_pool_collision_with_a_reserved_v2_cell() -> None:
    quotas = SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 1})
    seeded = _annotation("reserved", cell="reserved-cell")
    collision = _test_candidate("fresh", cell="reserved-cell")

    with pytest.raises(V3AnnotationSeedError) as error:
        _select_with_plan(
            SeedPlan((seeded,), (), {}, frozenset({"reserved-cell"})),
            quotas,
            FinalizedCandidatePool((collision,), (collision.h3_cell,)),
        )
    assert str(error.value) == "candidate pool collides with V2-reserved H3 cells: ['reserved-cell']"


def test_selection_reports_only_the_first_five_reserved_cell_collisions() -> None:
    quotas = SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 1})
    cells = tuple(f"reserved-cell-{index}" for index in range(6))
    excluded = tuple(_annotation(f"v2-{index}", cell=cell) for index, cell in enumerate(cells))
    candidates = tuple(_test_candidate(f"fresh-{index}", cell=cell) for index, cell in enumerate(cells))

    with pytest.raises(V3AnnotationSeedError) as error:
        _select_with_plan(
            SeedPlan((), excluded, {(Source.WIKIPEDIA, Label.YES): 1}, frozenset(cells)),
            quotas,
            FinalizedCandidatePool(candidates, cells),
        )

    assert str(error.value) == (
        "candidate pool collides with V2-reserved H3 cells: "
        "['reserved-cell-0', 'reserved-cell-1', 'reserved-cell-2', 'reserved-cell-3', "
        "'reserved-cell-4']"
    )


def test_selection_rejects_a_candidate_pool_id_used_by_v2() -> None:
    quotas = SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 1})
    seeded = _annotation("reused", cell="reserved-cell")
    duplicate = _test_candidate("reused", cell="fresh-cell")

    with pytest.raises(V3AnnotationSeedError) as error:
        _select_with_plan(
            SeedPlan((seeded,), (), {}, frozenset({"reserved-cell"})),
            quotas,
            FinalizedCandidatePool((duplicate,), (duplicate.h3_cell,)),
        )
    assert str(error.value) == "candidate pool reuses a V2 candidate ID"


def _state(
    rows: tuple[V3SeedRow, ...],
    quotas: SourceLabelQuotas,
    *,
    excluded: tuple[Annotation, ...] = (),
    reserved: frozenset[str] = frozenset(),
    reasons: dict[str, str] | None = None,
) -> V3AnnotationSeed:
    return V3AnnotationSeed(
        rows=rows,
        excluded_v2_rows=excluded,
        reserved_v2_cells=reserved,
        quotas=quotas,
        benchmark_sha256="0" * 64,
        seed="test-seed",
        excluded_v2_reasons=reasons or {},
    )


def test_annotation_seed_rejects_the_wrong_total_row_count() -> None:
    with pytest.raises(V3AnnotationSeedError, match="must contain 2 rows"):
        _state((_pending_row(),), SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 2}))


@pytest.mark.parametrize("same_cell", [False, True])
def test_annotation_seed_rejects_duplicate_ids_or_h3_cells(same_cell: bool) -> None:
    quotas = SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 2})
    first = _test_candidate("first", cell="first-cell")
    second = _test_candidate(
        "first" if not same_cell else "second", cell="first-cell" if same_cell else "second-cell"
    )

    expected = (
        "V3 seed rows must have globally unique H3 cells"
        if same_cell
        else "V3 seed rows must have unique candidate IDs"
    )
    with pytest.raises(V3AnnotationSeedError) as error:
        _state((_pending_row(first), _pending_row(second)), quotas)
    assert str(error.value) == expected


def test_annotation_seed_rejects_a_quota_matrix_mismatch() -> None:
    quotas = SourceLabelQuotas(
        {
            (Source.WIKIPEDIA, Label.YES): 1,
            (Source.WIKIPEDIA, Label.NO): 1,
        }
    )

    with pytest.raises(V3AnnotationSeedError) as error:
        _state(
            (
                _pending_row(_test_candidate("yes", cell="yes-cell")),
                _pending_row(_test_candidate("also-yes", cell="also-yes-cell")),
            ),
            quotas,
        )
    assert str(error.value) == "V3 source-by-label quota slots are not satisfied"


def test_annotation_seed_rejects_a_reserved_v2_cell_mismatch() -> None:
    with pytest.raises(V3AnnotationSeedError) as error:
        _state(
            (_pending_row(),),
            SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 1}),
            reserved=frozenset({"old-cell"}),
        )
    assert str(error.value) == "V3 state does not preserve every V2 H3 cell"


@pytest.mark.parametrize("reasons", [{}, {"excluded": " "}])
def test_annotation_seed_requires_a_non_empty_reason_for_each_excluded_row(
    reasons: dict[str, str],
) -> None:
    excluded = _annotation("excluded", cell="excluded-cell")
    seeded = _annotation("seeded", cell="seeded-cell")

    with pytest.raises(V3AnnotationSeedError) as error:
        _state(
            (_seeded_row(seeded),),
            SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 1}),
            excluded=(excluded,),
            reserved=frozenset({"excluded-cell", "seeded-cell"}),
            reasons=reasons,
        )
    assert str(error.value) == "every excluded V2 row must have a non-empty reason"


def test_annotation_seed_rejects_a_pending_row_in_a_reserved_v2_cell() -> None:
    excluded = _annotation("excluded", cell="reserved-cell")
    pending = _pending_row(_test_candidate("pending", cell="reserved-cell"))

    with pytest.raises(V3AnnotationSeedError) as error:
        _state(
            (pending,),
            SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 1}),
            excluded=(excluded,),
            reserved=frozenset({"reserved-cell"}),
            reasons={"excluded": "deterministic surplus"},
        )
    assert str(error.value) == "pending V3 rows collide with a V2-reserved H3 cell"
