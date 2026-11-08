"""Module 6 CLI: multi-segment promo generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.generation.multi_segment_generator import generate_multi_segment_promos
from src.generation.multi_segment_reports import write_multi_segment_outputs
from src.llm.factory import create_provider
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _load_list(path: Path, optional: bool = False) -> list[dict[str, Any]]:
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
    parser = argparse.ArgumentParser(description="Module 6: Multi-segment promo generation.")
    parser.add_argument("--episodes", default="outputs/episodes/parsed_episodes.json", help="Parsed episode/source JSON.")
    parser.add_argument("--analysis", default="outputs/scenes/scene_analysis.json", help="Module 2 scene analysis JSON.")
    parser.add_argument("--candidates", default="outputs/candidates/promo_candidates.json", help="Module 3 continuous candidates.")
    parser.add_argument("--ranked", default="outputs/promo_ranking_and_selection/ranked_promos.json", help="Optional Module 4 ranking source.")
    parser.add_argument("--output", default="outputs/candidates/multi_segment_candidates.json", help="Multi-segment candidate JSON.")
    parser.add_argument("--csv", default="outputs/multi_segment_promo_generation/multi_segment_promos.csv", help="Excel-compatible multi-segment CSV.")
    parser.add_argument("--metadata", default="outputs/multi_segment_promo_generation/run_metadata.json", help="Run metrics/checkpoint metadata JSON.")
    parser.add_argument("--provider", default=None, help="Override the provider from .env/config.yaml (use mock for offline tests).")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh narrative evaluation calls.")
    parser.add_argument("--cache", default=None, help="Override the narrative evaluation cache path.")
    parser.add_argument("--checkpoint", default=None, help="Override the recovery checkpoint path.")
    parser.add_argument("--max-llm-calls", type=int, default=None, help="Override configured LLM evaluation call cap.")
    args = parser.parse_args()

    settings = get_settings()
    if args.max_llm_calls is not None:
        if args.max_llm_calls < 0:
            parser.error("--max-llm-calls must be zero or greater.")
        settings.multi_segment.max_llm_calls = args.max_llm_calls
    episodes = _load_list(Path(args.episodes))
    scene_analysis = _load_list(Path(args.analysis), optional=True)
    continuous = _load_list(Path(args.candidates), optional=True)
    ranked_path = Path(args.ranked)
    ranked = _load_list(ranked_path, optional=True)
    # Module 4 rows preserve the original candidate fields plus rank/scores.
    candidates_by_id = {str(item.get("promo_id")): item for item in continuous}
    for item in ranked:
        candidates_by_id.setdefault(str(item.get("promo_id")), item)

    provider = create_provider(args.provider or settings.llm.provider)
    started = time.perf_counter()
    candidates, metadata = generate_multi_segment_promos(
        episodes,
        scene_analysis,
        list(candidates_by_id.values()),
        ranked,
        settings=settings,
        provider=provider,
        cache_path=args.cache or settings.multi_segment.cache_path,
        checkpoint_path=args.checkpoint or settings.multi_segment.checkpoint_path,
        use_cache=not args.no_cache,
    )
    metadata["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    metadata["candidate_input_count"] = len(continuous)
    metadata["ranked_input_count"] = len(ranked)
    metadata["episodes_count"] = len(episodes)
    metadata["scene_analyses_count"] = len(scene_analysis)
    metadata["cache_enabled"] = not args.no_cache
    metadata["narrative_evaluation_status"] = "mock" if provider.provider_name == "mock" else "provider"
    write_multi_segment_outputs(candidates, args.output, args.csv, args.metadata, metadata)
    logger.info(
        "Module 6 complete: moments=%s combinations=%s generated=%d calls=%s cache_hits=%s provider=%s model=%s",
        metadata.get("moments_discovered", 0), metadata.get("combinations_explored", 0),
        len(candidates), metadata.get("llm_requests", 0), metadata.get("cache_hits", 0),
        provider.provider_name, getattr(provider, "model", "unknown"),
    )
    print(f"Generated {len(candidates)} multi-segment promos. JSON: {args.output}; CSV: {args.csv}")


if __name__ == "__main__":
    main()
