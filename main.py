"""Root orchestration CLI for Modules 1–8."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from tests.cmd import (
    ingest_episodes,
    run_module4_rank,
    run_module5_quality,
    run_module6_multi_segment,
    run_module8_evaluation,
    run_promo_generation,
    run_scene_analysis,
)

ROOT_DIR = Path(__file__).resolve().parent


def main() -> None:
    os.chdir(ROOT_DIR)
    parser = argparse.ArgumentParser(description="Run the Kuku Modules 1–8 workflow.")
    parser.add_argument(
        "command",
        choices=("ingest", "analyze-scenes", "generate-candidates", "rank", "quality", "multi-segment", "evaluate", "ui", "run-all"),
        help="Module 1 ingestion, 2 narrative analysis, 3 candidate generation, 4 ranking, 5 quality, 6 multi-segment, 7 UI, or 8 evaluation.",
    )
    args, remainder = parser.parse_known_args()

    if args.command == "ingest":
        raise SystemExit(ingest_episodes.main(remainder))
    elif args.command == "analyze-scenes":
        sys.argv = ["run_scene_analysis.py", *remainder]
        run_scene_analysis.main()
    elif args.command == "generate-candidates":
        sys.argv = ["run_promo_generation.py", *remainder]
        run_promo_generation.main()
    elif args.command == "rank":
        sys.argv = ["run_module4_rank.py", *remainder]
        run_module4_rank.main()
    elif args.command == "quality":
        sys.argv = ["run_module5_quality.py", *remainder]
        run_module5_quality.main()
    elif args.command == "multi-segment":
        sys.argv = ["run_module6_multi_segment.py", *remainder]
        run_module6_multi_segment.main()
    elif args.command == "evaluate":
        sys.argv = ["run_module8_evaluation.py", *remainder]
        run_module8_evaluation.main()
    elif args.command == "ui":
        app_path = ROOT_DIR / "tests" / "cmd" / "app.py"
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)], cwd=ROOT_DIR))
    else:
        sys.argv = ["run_module4_rank.py", *remainder]
        run_module4_rank.main()
        sys.argv = ["run_module5_quality.py", *remainder]
        run_module5_quality.main()
        sys.argv = ["run_module6_multi_segment.py", *remainder]
        run_module6_multi_segment.main()


if __name__ == "__main__":
    main()
