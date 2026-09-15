from __future__ import annotations

import json
from pathlib import Path

from landuse_sentence_relevance.bootstrap import V3CandidatePoolResult
from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.domain.profile import V3_SOURCES, balanced_quotas
from landuse_sentence_relevance.domain.seeding import plan_seed
from landuse_sentence_relevance.domain.v3_preflight import V3PreflightReport


def test_cli_prints_only_compact_v3_preflight_evidence(monkeypatch, capsys, tmp_path: Path) -> None:
    import scripts.build_v3_candidate_pool as script

    from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool

    settings = V3Settings(data_root=tmp_path, benchmark_path=tmp_path / "v2.csv")
    settings.benchmark_path.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        "seed,yes,Seed,seed-cell,1,2,wikipedia,region,https://example.test/seed\n",
        encoding="utf-8",
    )
    result = V3CandidatePoolResult(
        pool=FinalizedCandidatePool((), ()),
        preflight=V3PreflightReport(
            candidate_count_by_source={source: 400 for source in V3_SOURCES},
            candidate_cells_by_source={source: 400 for source in V3_SOURCES},
            required_new_rows_by_source={source: 2 for source in V3_SOURCES},
            remaining_by_source_label={},
            reserved_v2_cells=frozenset({"seed-cell"}),
        ),
        seed_plan=plan_seed((), balanced_quotas(V3_SOURCES, 1), seed="test"),
    )
    monkeypatch.setattr(script.V3Settings, "from_env", lambda: settings)
    monkeypatch.setattr(script, "build_v3_candidate_pool", lambda actual: result)

    assert script.main([]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload == {
        "candidate_cells_by_source": {source.value: 400 for source in V3_SOURCES},
        "candidate_count_by_source": {source.value: 400 for source in V3_SOURCES},
        "candidate_pool_path": str(settings.candidate_pool_path),
        "candidate_progress_path": str(settings.candidate_progress_path),
        "required_new_rows_by_source": {source.value: 2 for source in V3_SOURCES},
        "reserved_v2_cell_count": 1,
        "total_candidate_count": 1200,
        "total_required_new_rows": 6,
    }
    assert "sentence" not in payload
    assert Source.WIKIPEDIA.value in payload["candidate_count_by_source"]
