"""Module 2 CLI: narrative understanding and dramatic moment detection."""

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

from src.analysis.scene_analyzer import SceneAnalyzer
from src.config import get_settings
from src.llm.factory import create_provider


def main() -> None:
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Module 2: Narrative understanding and dramatic moment detection.")
    parser.add_argument("--input", default="outputs/episodes/parsed_episodes.json", help="Path to the parsed episode JSON file.")
    parser.add_argument("--output", default="outputs/scenes/scene_analysis.json", help="Path to the output JSON file.")
    parser.add_argument("--provider", default=None, help="Override provider selection: mock, gemini, openai, or openai_compatible.")
    args = parser.parse_args()

    settings = get_settings()
    provider = create_provider(args.provider or settings.llm.provider)
    analyzer = SceneAnalyzer(provider=provider)

    input_path = Path(args.input)
    output_path = Path(args.output)
    episodes = json.loads(input_path.read_text(encoding="utf-8"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        results = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(results, list):
            raise ValueError(f"Existing analysis output must be a JSON list: {output_path}")
    else:
        results = []

    analyses_by_scene = {
        (int(item.get("episode_id", 0)), int(item.get("scene_id", 0))): item
        for item in results
        if isinstance(item, dict)
    }
    total_scenes = sum(len(episode.get("scenes", [])) for episode in episodes)
    completed = len(analyses_by_scene)

    for episode in episodes:
        previous_summary = None
        for scene in episode.get("scenes", []):
            scene_copy = dict(scene)
            scene_copy["episode_id"] = episode.get("episode_id")
            scene_key = (int(scene_copy.get("episode_id", 0)), int(scene_copy.get("scene_id", 0)))
            if scene_key in analyses_by_scene:
                previous_summary = analyses_by_scene[scene_key].get("dramatic_summary")
                print(
                    f"[{completed}/{total_scenes}] Already analyzed episode "
                    f"{scene_key[0]} scene {scene_key[1]}; skipping.",
                    flush=True,
                )
                continue

            context_characters = scene_copy.get("characters", [])
            progress_index = completed + 1
            print(
                f"[{progress_index}/{total_scenes}] Requesting analysis for episode "
                f"{episode.get('episode_id')} scene {scene_copy.get('scene_id')}...",
                flush=True,
            )
            try:
                analysis = analyzer.analyze_scene(
                    scene_copy,
                    previous_summary=previous_summary,
                    context_characters=context_characters,
                )
            except Exception as exc:
                output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"ERROR: Module 2 stopped before completing the scene: {exc}", flush=True)
                print(f"Completed analyses were preserved in {output_path}", flush=True)
                raise SystemExit(1) from exc

            results.append(analysis)
            analyses_by_scene[scene_key] = analysis
            completed += 1
            previous_summary = analysis.get("dramatic_summary")

            output_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[{completed}/{total_scenes}] Done: "
                f"{analysis.get('event_type', 'unknown')}: "
                f"{analysis.get('dramatic_summary', '')}",
                flush=True,
            )

    print(f"Analyzed {len(results)} scenes and wrote {output_path}")


if __name__ == "__main__":
    main()
