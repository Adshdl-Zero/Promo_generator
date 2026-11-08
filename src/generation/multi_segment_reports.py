"""Module 6 JSON and Excel-compatible CSV output helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

MULTI_SEGMENT_CSV_FIELDS = [
    "Promo ID", "Promo type", "Generation strategy", "Segment count", "Episode IDs",
    "Segment timestamps", "Total duration", "Hook score", "Cliffhanger score",
    "Standalone score", "Narrative coherence", "Overall promotional score",
    "Evaluation status", "Combined summary", "Weaknesses",
]


def write_multi_segment_outputs(
    candidates: list[dict[str, Any]],
    json_path: str | Path,
    csv_path: str | Path,
    metadata_path: str | Path,
    metadata: dict[str, Any],
) -> None:
    json_file = Path(json_path)
    csv_file = Path(csv_path)
    meta_file = Path(metadata_path)
    for path in (json_file, csv_file, meta_file):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_file.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MULTI_SEGMENT_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for candidate in candidates:
            evaluation = candidate.get("evaluation", {})
            timestamps = " | ".join(
                f"{item.get('segment_order')}. E{item.get('episode_id')}/S{','.join(map(str, item.get('scene_ids', [])))} "
                f"{item.get('start_time')}-{item.get('end_time')} ({item.get('segment_duration')}s)"
                for item in candidate.get("segments", [])
            )
            writer.writerow({
                "Promo ID": candidate.get("promo_id"),
                "Promo type": candidate.get("promo_type"),
                "Generation strategy": candidate.get("generation_strategy"),
                "Segment count": candidate.get("segment_count"),
                "Episode IDs": ", ".join(map(str, candidate.get("episode_ids", []))),
                "Segment timestamps": timestamps,
                "Total duration": candidate.get("total_duration"),
                "Hook score": evaluation.get("hook_score"),
                "Cliffhanger score": evaluation.get("cliffhanger_score"),
                "Standalone score": evaluation.get("standalone_context_score"),
                "Narrative coherence": evaluation.get("narrative_coherence_score"),
                "Overall promotional score": evaluation.get("promo_engagement_potential"),
                "Evaluation status": candidate.get("evaluation_status"),
                "Combined summary": candidate.get("combined_summary"),
                "Weaknesses": "; ".join(candidate.get("weaknesses", [])),
            })


__all__ = ["MULTI_SEGMENT_CSV_FIELDS", "write_multi_segment_outputs"]
