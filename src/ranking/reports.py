"""Module 4 JSON and Excel-compatible CSV output helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

RANKING_CSV_FIELDS = [
    "Rank",
    "Promo ID",
    "Episode ID",
    "Start timestamp",
    "End timestamp",
    "Duration",
    "Final score",
    "Hook score",
    "Cliffhanger score",
    "Standalone score",
    "Emotional intensity",
    "Narrative coherence",
    "Quality category",
    "Selection status",
    "Short explanation",
]


def write_ranking_outputs(rows: list[dict[str, Any]], json_path: str | Path, csv_path: str | Path) -> None:
    json_file = Path(json_path)
    csv_file = Path(csv_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_file.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKING_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            segment = (row.get("segments") or [{}])[0]
            scores = row.get("criterion_scores", {})
            writer.writerow({
                "Rank": row.get("rank"),
                "Promo ID": row.get("promo_id"),
                "Episode ID": segment.get("episode_id"),
                "Start timestamp": segment.get("start_time"),
                "End timestamp": segment.get("end_time"),
                "Duration": row.get("total_duration"),
                "Final score": row.get("raw_score"),
                "Hook score": scores.get("hook_strength"),
                "Cliffhanger score": scores.get("cliffhanger_strength"),
                "Standalone score": scores.get("standalone_understandability"),
                "Emotional intensity": scores.get("emotional_intensity"),
                "Narrative coherence": scores.get("narrative_coherence"),
                "Quality category": row.get("quality_category"),
                "Selection status": row.get("selection_status"),
                "Short explanation": row.get("ranking_explanation"),
            })


__all__ = ["RANKING_CSV_FIELDS", "write_ranking_outputs"]
