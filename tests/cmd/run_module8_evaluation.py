"""Run Module 8 system evaluation and write QA reports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.system_evaluation import (
    compute_system_evaluation,
    load_human_reviews,
    write_evaluation_outputs,
)


def _load_json(path: Path, *, default: Any = None, required: bool = False) -> Any:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input file not found: {path}")
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _load_list(path: Path, *, required: bool = False) -> list[dict[str, Any]]:
    value = _load_json(path, default=[], required=required)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return [item for item in value if isinstance(item, dict)]


def _load_object(path: Path) -> dict[str, Any]:
    value = _load_json(path, default={})
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _final_candidates(
    ranked: list[dict[str, Any]], quality_report: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return only Module 4-selected promos accepted by Module 5."""
    quality_by_id = {
        str(item.get("promo_id", "")): item
        for item in quality_report.get("candidate_results", [])
        if isinstance(item, dict)
    }
    final: list[dict[str, Any]] = []
    for candidate in ranked:
        promo_id = str(candidate.get("promo_id", ""))
        quality = quality_by_id.get(promo_id)
        if candidate.get("selection_status") != "selected" or quality is None:
            continue
        if quality.get("technical_status") not in {"PASS", "PASS_WITH_WARNINGS"}:
            continue
        if quality.get("quality_decision") not in {"PASS", "PASS_WITH_WARNINGS"}:
            continue
        # Quality rows preserve the complete ranked candidate plus Module 5 results.
        final.append(quality)
    return final


def main(argv: list[str] | None = None) -> None:
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(
        description="Module 8: Human evaluation and quality assurance."
    )
    parser.add_argument("--ranked", default="outputs/promo_ranking_and_selection/ranked_promos.json", help="Ranked promo JSON (required).")
    parser.add_argument("--multi-segment", default="outputs/candidates/multi_segment_candidates.json", help="Legacy input; only ranked and Module 5-approved promos are evaluated.")
    parser.add_argument("--quality", default="outputs/quality_evaluation_and_validation/quality_report.json", help="Module 5 quality report JSON (required to identify finalists).")
    parser.add_argument("--episodes", default="outputs/episodes/parsed_episodes.json", help="Parsed episode/source JSON (required).")
    parser.add_argument("--reviews", default="outputs/evaluation/human_reviews.json", help="Optional human review JSON.")
    parser.add_argument("--module6-metadata", default="outputs/multi_segment_promo_generation/run_metadata.json", help="Optional Module 6 run metadata JSON.")
    parser.add_argument("--metrics-output", default="outputs/evaluation/evaluation_metrics.json", help="Metrics JSON output.")
    parser.add_argument("--report-output", default="outputs/evaluation/evaluation_report.csv", help="Evaluation CSV output.")
    parser.add_argument("--errors-output", default="outputs/evaluation/error_analysis.json", help="Error analysis JSON output.")
    parser.add_argument("--summary-output", default="outputs/evaluation/evaluation_summary.md", help="Summary Markdown output.")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 20], help="Ranking cutoffs, e.g. --cutoffs 5 10 20.")
    args = parser.parse_args(argv)

    if any(cutoff < 1 for cutoff in args.cutoffs):
        parser.error("All --cutoffs values must be positive integers.")

    ranked = _load_list(Path(args.ranked), required=True)
    episodes = _load_list(Path(args.episodes), required=True)
    quality_path = Path(args.quality)
    if not quality_path.exists():
        parser.error(f"Module 5 quality report not found: {quality_path}. Run Module 5 before Module 8.")
    quality_report = _load_object(quality_path)
    if not isinstance(quality_report.get("candidate_results"), list):
        parser.error(f"Module 5 report has no candidate_results list: {quality_path}")
    reviews = load_human_reviews(args.reviews)
    module6_metadata = _load_object(Path(args.module6_metadata))

    candidates = _final_candidates(ranked, quality_report)
    metrics, error_analysis, rows = compute_system_evaluation(
        candidates,
        quality_report,
        episodes,
        reviews,
        cutoffs=args.cutoffs,
        module6_metadata=module6_metadata,
    )
    write_evaluation_outputs(
        metrics,
        error_analysis,
        rows,
        metrics_path=args.metrics_output,
        csv_path=args.report_output,
        error_path=args.errors_output,
        summary_path=args.summary_output,
    )
    print(
        f"Module 8 evaluation complete: candidates={len(candidates)}, "
        f"human_reviews={len(reviews)}. Outputs: {args.metrics_output}, "
        f"{args.report_output}, {args.errors_output}, {args.summary_output}"
    )


if __name__ == "__main__":
    main()
