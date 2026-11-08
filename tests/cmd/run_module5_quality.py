"""Module 5 CLI: quality evaluation and validation."""

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

from src.config import get_settings
from src.llm.factory import create_provider
from src.quality.evaluator import evaluate_ranked_promos, write_quality_outputs
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _load_list(path: Path, *, optional: bool = False) -> list[dict[str, Any]]:
    if optional and not path.exists():
        return []
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return value


def main() -> None:
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Module 5: Quality evaluation and validation.")
    parser.add_argument("--ranked", default="outputs/promo_ranking_and_selection/ranked_promos.json", help="Module 4 ranked promo JSON.")
    parser.add_argument("--candidates", default="outputs/candidates/promo_candidates.json", help="Original Module 3 candidate JSON.")
    parser.add_argument("--multi-segment", default="outputs/candidates/multi_segment_candidates.json", help="Optional original Module 6 candidate JSON.")
    parser.add_argument("--episodes", default="outputs/episodes/parsed_episodes.json", help="Parsed source episodes JSON.")
    parser.add_argument("--output", default="outputs/quality_evaluation_and_validation/quality_report.json", help="Quality report JSON.")
    parser.add_argument("--csv", default="outputs/quality_evaluation_and_validation/quality_report.csv", help="Quality report CSV.")
    parser.add_argument("--failures", default="outputs/quality_evaluation_and_validation/failure_analysis.json", help="Failure analysis JSON.")
    parser.add_argument("--cache", default=None, help="Override persistent narrative evaluation cache path.")
    parser.add_argument("--technical-only", action="store_true", help="Run deterministic validation without narrative LLM calls.")
    parser.add_argument("--narrative-attempts", type=int, default=None, help="Override maximum schema-correction attempts per promo.")
    parser.add_argument("--no-cache", action="store_true", help="Do not read or write cached narrative evaluations.")
    parser.add_argument("--provider", default=None, help="Optional provider override.")
    args = parser.parse_args()

    ranked = _load_list(Path(args.ranked))
    original = _load_list(Path(args.candidates), optional=True)
    multi_path = Path(args.multi_segment)
    if multi_path.exists():
        original.extend(_load_list(multi_path))
    episodes = _load_list(Path(args.episodes))
    settings = get_settings()
    if args.technical_only:
        settings.quality.maximum_narrative_evaluations = 0
    if args.narrative_attempts is not None:
        if args.narrative_attempts < 1:
            parser.error("--narrative-attempts must be at least 1.")
        settings.quality.max_narrative_attempts = args.narrative_attempts
    provider = create_provider(args.provider or settings.llm.provider)

    report, failures, records = evaluate_ranked_promos(
        ranked,
        original,
        episodes,
        settings=settings,
        provider=provider,
        cache_path=args.cache,
        use_cache=not args.no_cache,
    )
    write_quality_outputs(report, failures, records, args.output, args.csv, args.failures)
    logger.info(
        "Module 5 complete: candidates=%d narrative=%d pass=%d warnings=%d fail=%d provider=%s model=%s",
        report["candidates_evaluated"], report["narrative_candidates_evaluated"],
        report["passing_validation"], report["passing_with_warnings"], report["failing"],
        provider.provider_name, getattr(provider, "model", "unknown"),
    )
    print(
        f"Quality evaluation complete: {report['candidates_evaluated']} candidates; "
        f"PASS={report['passing_validation']}, WARN={report['passing_with_warnings']}, "
        f"FAIL={report['failing']}. Outputs: {args.output}, {args.csv}, {args.failures}"
    )


if __name__ == "__main__":
    main()
