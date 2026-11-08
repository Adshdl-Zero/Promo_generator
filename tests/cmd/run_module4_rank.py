"""Module 4 CLI: promo ranking and selection."""

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
from src.ranking.ranker import rank_candidates
from src.ranking.reports import write_ranking_outputs
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _load_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return value


def main() -> None:
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Module 4: Promo ranking and selection.")
    parser.add_argument("--input", default="outputs/candidates/promo_candidates.json", help="Module 3 candidate JSON.")
    parser.add_argument("--multi-segment", default="outputs/candidates/multi_segment_candidates.json", help="Optional Module 6 candidates to include.")
    parser.add_argument("--continuous-only", action="store_true", help="Do not include Module 6 candidates.")
    parser.add_argument("--episodes", default="outputs/episodes/parsed_episodes.json", help="Parsed source episodes JSON.")
    parser.add_argument("--analysis", default="outputs/scenes/scene_analysis.json", help="Module 2 scene analysis JSON.")
    parser.add_argument("--output", default="outputs/promo_ranking_and_selection/ranked_promos.json", help="Ranked output JSON.")
    parser.add_argument("--csv", default="outputs/promo_ranking_and_selection/ranked_promos.csv", help="Excel-compatible ranked output CSV.")
    parser.add_argument("--cache", default="outputs/promo_ranking_and_selection/ranking_score_cache.json", help="Persistent cache for missing-score LLM assessments.")
    parser.add_argument("--provider", default=None, help="Optional provider override.")
    args = parser.parse_args()

    candidates = _load_list(Path(args.input))
    if not args.continuous_only:
        multi_path = Path(args.multi_segment)
        if multi_path.exists():
            candidates.extend(_load_list(multi_path))
    episodes = _load_list(Path(args.episodes))
    analysis_path = Path(args.analysis)
    scene_analysis = _load_list(analysis_path) if analysis_path.exists() else []
    settings = get_settings()
    provider = create_provider(args.provider or settings.llm.provider)

    ranked = rank_candidates(
        candidates,
        episodes,
        scene_analysis,
        settings=settings,
        provider=provider,
        cache_path=args.cache,
    )
    write_ranking_outputs(ranked, args.output, args.csv)
    selected = sum(row.get("selection_status") == "selected" for row in ranked)
    logger.info(
        "Module 4 complete: input=%d ranked=%d selected=%d provider=%s model=%s json=%s csv=%s",
        len(candidates), len(ranked), selected, provider.provider_name,
        getattr(provider, "model", "unknown"), args.output, args.csv,
    )
    print(f"Ranked {len(ranked)} promos; selected {selected}. Outputs: {args.output}, {args.csv}")


if __name__ == "__main__":
    main()
