"""Module 1 CLI: data ingestion and screenplay parsing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import ingest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Module 1: Data ingestion and screenplay parsing."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Input file or directory (default: ./data; searched recursively).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "episodes" / "parsed_episodes.json",
        help="Parsed episode JSON output path.",
    )
    args = parser.parse_args(argv)

    try:
        episodes = ingest(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(episodes, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    scene_count = sum(len(episode["scenes"]) for episode in episodes)
    print(f"Parsed {len(episodes)} episode(s), {scene_count} scene(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
