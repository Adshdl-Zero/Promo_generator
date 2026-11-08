"""Module 3 CLI: continuous promo candidate generation."""

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
from src.generation.candidate_evaluator import CandidateEvaluator
from src.generation.candidate_filter import select_top_candidates
from src.generation.candidate_generator import CandidateGenerator
from src.llm.factory import create_provider


def load_input(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Module 3: Continuous promo candidate generation.")
    parser.add_argument("--input", default="outputs/episodes/parsed_episodes.json", help="Path to the parsed episode JSON file.")
    parser.add_argument("--analysis", default="outputs/scenes/scene_analysis.json", help="Path to the scene analysis JSON file.")
    parser.add_argument("--output", default="outputs/candidates/promo_candidates.json", help="Path to the output candidate JSON file.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Override the configured maximum number of candidates evaluated by the LLM.")
    parser.add_argument("--allow-incomplete-analysis", action="store_true", help="Continue even if some episode scenes have no Module 2 analysis.")
    parser.add_argument("--provider", default=None, help="Override provider selection: mock, gemini, openai, or openai_compatible.")
    args = parser.parse_args()
    if args.max_candidates is not None and args.max_candidates < 1:
        parser.error("--max-candidates must be a positive integer.")

    settings = get_settings()
    provider = create_provider(args.provider or settings.llm.provider)
    evaluation_limit = args.max_candidates if args.max_candidates is not None else settings.evaluation.max_candidates
    if evaluation_limit < 1:
        parser.error("The configured evaluation.max_candidates must be a positive integer.")
    episodes = load_input(Path(args.input))
    analysis_data = load_input(Path(args.analysis)) if Path(args.analysis).exists() else []
    scene_by_episode: dict[int, list[dict[str, Any]]] = {}
    for item in analysis_data:
        scene_by_episode.setdefault(int(item.get("episode_id", 0)), []).append(item)

    expected_scene_keys = {
        (int(episode.get("episode_id", 0)), int(scene.get("scene_id", 0)))
        for episode in episodes
        for scene in episode.get("scenes", [])
    }
    analyzed_scene_keys = {
        (int(item.get("episode_id", 0)), int(item.get("scene_id", 0)))
        for item in analysis_data
    }
    missing_analyses = len(expected_scene_keys - analyzed_scene_keys)
    if missing_analyses:
        message = (
            f"Scene analysis is incomplete: {len(analyzed_scene_keys & expected_scene_keys)} of "
            f"{len(expected_scene_keys)} scenes have analysis. Run Module 2 to completion first."
        )
        if not args.allow_incomplete_analysis:
            parser.error(message + " Use --allow-incomplete-analysis only if a partial run is intentional.")
        print(f"WARNING: {message}", flush=True)

    generator = CandidateGenerator(
        min_duration=settings.candidate_generation.min_duration,
        max_duration=settings.candidate_generation.max_duration,
        preferred_methods=settings.candidate_generation.methods,
    )
    evaluator = CandidateEvaluator(
        provider,
        max_retries=settings.llm.max_retries,
        retry_backoff_seconds=settings.llm.retry_backoff_seconds,
        request_interval_seconds=settings.evaluation.request_interval_seconds,
    )

    pending: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    for episode in episodes:
        episode_id = int(episode.get("episode_id"))
        analyses = scene_by_episode.get(episode_id, [])
        for candidate in generator.generate_for_episode(episode, analyses):
            pending.append((episode, candidate, analyses))

    generated_count = len(pending)
    pending = select_top_candidates(pending, evaluation_limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_candidates = len(pending)

    results: list[dict[str, Any]] = []
    if output_path.exists():
        existing_results = load_input(output_path)
        for index, result in enumerate(existing_results[:total_candidates]):
            _, candidate, _ = pending[index]
            if (
                result.get("segments") != candidate.get("segments")
                or result.get("total_duration") != candidate.get("total_duration")
                or result.get("generation_method") != candidate.get("generation_method")
            ):
                break
            results.append(result)

    print(
        f"Ranked {generated_count} candidates and selected {total_candidates} for LLM evaluation "
        f"(limit={evaluation_limit}). Provider: '{provider.provider_name}'.",
        flush=True,
    )
    if results:
        print(f"Resuming from {len(results)} previously completed candidates.", flush=True)

    for index, (episode, candidate, analyses) in enumerate(pending[len(results):], start=len(results) + 1):
        promo_id = f"P{index:04d}"
        candidate["promo_id"] = promo_id
        segment = candidate["segments"][0]
        print(
            f"[{index}/{total_candidates}] Requesting evaluation for {promo_id} "
            f"(episode {episode.get('episode_id')}, {segment['start_time']}-{segment['end_time']})...",
            flush=True,
        )
        try:
            evaluated = evaluator.evaluate(episode, candidate, analyses)
        except Exception as exc:
            if results:
                output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"Completed candidates were preserved in {output_path}", flush=True)
            print(f"ERROR: Module 3 stopped while evaluating {promo_id}: {exc}", flush=True)
            raise SystemExit(1) from exc

        evaluated["promo_id"] = promo_id
        results.append(evaluated)
        print(
            f"[{index}/{total_candidates}] Done: {promo_id}: "
            f"{evaluated.get('overall_summary', '')}",
            flush=True,
        )
        if index % 25 == 0 or index == total_candidates:
            output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved checkpoint: {len(results)} candidates -> {output_path}", flush=True)

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(results)} promo candidates and wrote {args.output}")


if __name__ == "__main__":
    main()
