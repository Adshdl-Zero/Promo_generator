"""Tests for the standalone Module 8 evaluation command."""

from __future__ import annotations

import json
from pathlib import Path

from tests.cmd.run_module8_evaluation import main


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_module8_cli_reads_inputs_and_writes_all_reports(tmp_path):
    ranked_path = tmp_path / "ranked.json"
    multi_path = tmp_path / "multi.json"
    episodes_path = tmp_path / "episodes.json"
    quality_path = tmp_path / "quality.json"
    reviews_path = tmp_path / "reviews.json"
    metadata_path = tmp_path / "metadata.json"
    _write_json(multi_path, [{
        "promo_id": "MS1",
        "promo_type": "multi_segment",
        "selection_status": "selected",
        "rank": 2,
        "raw_score": 75,
        "segments": [{
            "episode_id": 1,
            "scene_ids": [1],
            "start_time": "00:00:30",
            "end_time": "00:00:55",
        }],
    }])
    _write_json(episodes_path, [{
        "episode_id": 1,
        "scenes": [{"scene_id": 1, "dialogue": [], "actions": []}],
    }])
    _write_json(quality_path, {
        "candidate_results": [
            {
                "promo_id": "P1",
                "selection_status": "selected",
                "rank": 1,
                "raw_score": 80,
                "quality_decision": "PASS_WITH_WARNINGS",
                "technical_status": "PASS",
                "segments": [{
                    "episode_id": 1,
                    "scene_ids": [1],
                    "start_time": "00:00:00",
                    "end_time": "00:00:30",
                }],
            },
            {
                "promo_id": "P2",
                "selection_status": "selected",
                "quality_decision": "FAIL",
                "technical_status": "FAIL",
                "segments": [],
            },
        ],
        "technical_validation_results": [],
    })
    _write_json(ranked_path, [{
        "promo_id": "P1",
        "promo_type": "continuous",
        "selection_status": "selected",
        "rank": 1,
        "raw_score": 80,
        "segments": [{
            "episode_id": 1,
            "scene_ids": [1],
            "start_time": "00:00:00",
            "end_time": "00:00:30",
        }],
    }, {
        "promo_id": "P2",
        "selection_status": "selected",
        "rank": 2,
        "raw_score": 70,
    }])
    _write_json(reviews_path, {"reviews": []})
    _write_json(metadata_path, {"elapsed_seconds": 1.2, "llm_requests": 3, "cache_hits": 0})

    output_paths = {
        "metrics": tmp_path / "output" / "metrics.json",
        "report": tmp_path / "output" / "report.csv",
        "errors": tmp_path / "output" / "errors.json",
        "summary": tmp_path / "output" / "summary.md",
    }
    main([
        "--ranked", str(ranked_path),
        "--multi-segment", str(multi_path),
        "--episodes", str(episodes_path),
        "--quality", str(quality_path),
        "--reviews", str(reviews_path),
        "--module6-metadata", str(metadata_path),
        "--metrics-output", str(output_paths["metrics"]),
        "--report-output", str(output_paths["report"]),
        "--errors-output", str(output_paths["errors"]),
        "--summary-output", str(output_paths["summary"]),
    ])

    for output_path in output_paths.values():
        assert output_path.exists()
    metrics = json.loads(output_paths["metrics"].read_text(encoding="utf-8"))
    assert metrics["generated_candidate_count"] == 1
    assert metrics["reviewed_promo_count"] == 0
    assert metrics["performance_metrics"]["llm_api_calls"] == 3
    report_rows = output_paths["report"].read_text(encoding="utf-8-sig").splitlines()
    assert len(report_rows) == 2
    assert "P1" in report_rows[1]
    assert "P2" not in "\n".join(report_rows)
    assert "MS1" not in "\n".join(report_rows)
