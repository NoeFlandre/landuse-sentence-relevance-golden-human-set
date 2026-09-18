from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from landuse_sentence_relevance.domain.models import Annotation, Label, Source
from landuse_sentence_relevance.domain.profile import SourceLabelQuotas
from landuse_sentence_relevance.domain.v3_annotation import (
    V3AnnotationSeed,
    V3SeedRow,
    V3SelectionMetadata,
)
from tests.builders import make_candidate


def _annotation(candidate_id: str, cell: str, label: Label) -> Annotation:
    candidate = replace(
        make_candidate(candidate_id),
        h3_cell=cell,
        source=Source.WIKIPEDIA,
    )
    return Annotation(candidate=candidate, label=label)


def _state() -> V3AnnotationSeed:
    reused = _annotation("seeded", "seeded-cell", Label.YES)
    excluded = _annotation("excluded", "excluded-cell", Label.YES)
    return V3AnnotationSeed(
        rows=(
            V3SeedRow(
                candidate=reused.candidate,
                quota_source=Source.WIKIPEDIA,
                quota_label=Label.YES,
                origin="v2",
                annotation=reused,
                selection=V3SelectionMetadata(seed="test", rank="0" * 64, slot_index=0),
            ),
        ),
        excluded_v2_rows=(excluded,),
        excluded_v2_reasons={excluded.candidate.candidate_id: "deterministic quota surplus"},
        reserved_v2_cells=frozenset({reused.candidate.h3_cell, excluded.candidate.h3_cell}),
        quotas=SourceLabelQuotas({(Source.WIKIPEDIA, Label.YES): 1}),
        benchmark_sha256="a" * 64,
        seed="test",
    )


def test_cli_prints_quota_evidence_without_candidate_sentences(monkeypatch, capsys, tmp_path: Path) -> None:
    import scripts.build_v3_annotation_seed as script

    settings = type("Settings", (), {"annotation_seed_path": tmp_path / "v3.json"})()
    monkeypatch.setattr(script.V3Settings, "from_env", lambda: settings)
    monkeypatch.setattr(script, "build_v3_annotation_seed", lambda actual: _state())

    assert script.main([]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload == {
        "annotation_seed_path": str(settings.annotation_seed_path),
        "benchmark_sha256": "a" * 64,
        "excluded_v2": [{"candidate_id": "excluded", "reason": "deterministic quota surplus"}],
        "pending_count": 0,
        "pending_count_by_source": {"wikipedia": 0},
        "pending_count_by_source_label": {"wikipedia/yes": 0},
        "quota_counts": {"wikipedia/yes": 1},
        "reserved_v2_cell_count": 2,
        "seeded_count": 1,
        "seeded_count_by_source": {"wikipedia": 1},
        "seeded_count_by_source_label": {"wikipedia/yes": 1},
        "total_rows": 1,
        "rows_by_source": {"wikipedia": 1},
    }
    assert "sentence" not in payload
