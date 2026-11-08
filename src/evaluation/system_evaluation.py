"""Module 8: human evaluation and quality assurance."""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from src.schemas.multi_segment import HumanReview
from src.utils.timestamp import duration_seconds, parse_timestamp

REVIEW_RATINGS = (
    "hook_rating", "cliffhanger_rating", "standalone_rating", "coherence_rating",
    "emotional_impact_rating", "promotional_potential_rating",
)
HUMAN_REVIEW_FIELDS = [
    "promo_id", "reviewer_id", "reviewed_at", "overall_label", *REVIEW_RATINGS,
    "comments", "failure_categories", "incorrect_timestamps", "missing_context",
]
REVIEW_DATASET_FIELDS = [
    "promo_id", "promo_type", "episode_ids", "segment_timestamps", "segments_json",
    "total_duration", "promo_summary", "hook_evidence", "cliffhanger_evidence",
    "automated_rank", "automated_score", "automated_quality_category", "automated_quality_score",
    "automated_quality_decision", "automated_hook_score", "automated_cliffhanger_score",
    "automated_standalone_score", "automated_coherence_score", "source_text",
    "human_reviewer_id", "human_overall_label", *REVIEW_RATINGS, "human_feedback",
    "human_failure_categories", "human_incorrect_timestamps", "human_missing_context",
]
EVALUATION_CSV_FIELDS = [
    "Promo ID", "Promo type", "AI rank", "AI score", "AI quality category", "Human label",
    "Human overall rating", "Human hook rating", "Human cliffhanger rating",
    "Human standalone rating", "Human coherence rating", "Human emotional impact rating",
    "Agreement status", "Technical validation status", "Failure labels", "Reviewer feedback",
]


def _json_read(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _evidence(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " | ".join(str(item) for item in value if isinstance(item, str))
    return ""


def _source_for_candidate(candidate: dict[str, Any], episodes_by_id: dict[int, dict[str, Any]], limit: int) -> str:
    lines: list[str] = []
    for index, segment in enumerate(candidate.get("segments", []), start=1):
        episode_id = int(segment.get("episode_id", 0))
        episode = episodes_by_id.get(episode_id, {})
        scene_ids = {int(value) for value in segment.get("scene_ids", [])}
        start = parse_timestamp(segment.get("start_time"))
        end = parse_timestamp(segment.get("end_time"))
        lines.append(f"SEGMENT {index}: episode {episode_id}, scenes {sorted(scene_ids)}, {segment.get('start_time')}-{segment.get('end_time')}")
        for scene in episode.get("scenes", []):
            if int(scene.get("scene_id", -1)) not in scene_ids:
                continue
            for kind, field in (("DIALOGUE", "dialogue"), ("ACTION", "actions")):
                for event in scene.get(field, []):
                    event_start = parse_timestamp(event.get("start_time"))
                    event_end = parse_timestamp(event.get("end_time"))
                    if start is not None and end is not None and event_start is not None:
                        if event_start > end or (event_end if event_end is not None else event_start) < start:
                            continue
                    if event.get("text"):
                        lines.append(f"{event.get('start_time')} {kind} {event.get('character', '')}: {event['text']}")
    text = "\n".join(lines)
    if len(text) <= limit:
        return text
    marker = "\n...[middle omitted]...\n"
    usable = limit - len(marker)
    head = usable * 2 // 3
    return text[:head] + marker + text[-(usable - head):]


def _candidate_key(candidate: dict[str, Any], index: int) -> str:
    value = candidate.get("promo_id")
    return str(value) if value else f"PROMO-{index:04d}"


def _sample_candidates(candidates: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    if len(candidates) <= sample_size:
        return list(candidates)
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []

    def take(item: dict[str, Any]) -> None:
        if item in remaining and len(selected) < sample_size:
            selected.append(item)
            remaining.remove(item)

    # Include QA-flagged and multi-segment examples before score-stratified sampling.
    for item in candidates:
        flags = item.get("duplicate_or_overlap_flags", [])
        failures = item.get("failure_labels", [])
        if item.get("promo_type") == "multi_segment" or flags or failures or item.get("technical_status") in {"FAIL", "PASS_WITH_WARNINGS"}:
            take(item)

    remaining.sort(key=lambda item: (-float(item.get("raw_score", 0) or 0), str(item.get("promo_id", ""))))
    buckets = [remaining[: max(1, len(remaining) // 3)], remaining[max(1, len(remaining) // 3): max(2, 2 * len(remaining) // 3)], remaining[max(2, 2 * len(remaining) // 3):]]
    positions = [0, 0, 0]
    while len(selected) < sample_size:
        made_progress = False
        for bucket_index, bucket in enumerate(buckets):
            while positions[bucket_index] < len(bucket) and bucket[positions[bucket_index]] not in remaining:
                positions[bucket_index] += 1
            if positions[bucket_index] < len(bucket):
                take(bucket[positions[bucket_index]])
                positions[bucket_index] += 1
                made_progress = True
                if len(selected) >= sample_size:
                    break
        if not made_progress:
            break
    return selected


def build_review_dataset(
    continuous_candidates: list[dict[str, Any]],
    multi_candidates: list[dict[str, Any]],
    ranked_promos: list[dict[str, Any]],
    quality_report: dict[str, Any],
    episodes: list[dict[str, Any]],
    *,
    sample_size: int,
    source_text_limit: int = 1800,
    existing_dataset: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a representative candidate review sheet without fabricating review values."""
    episode_map = {int(item.get("episode_id", 0)): item for item in episodes}
    ranked_by_id = {str(item.get("promo_id", "")): item for item in ranked_promos}
    quality_by_id = {
        str(item.get("promo_id", "")): item
        for item in quality_report.get("candidate_results", [])
    }
    previous_by_id = {str(item.get("promo_id", "")): item for item in (existing_dataset or [])}
    original_candidates: dict[str, dict[str, Any]] = {}
    for candidate in [*continuous_candidates, *multi_candidates]:
        original_candidates.setdefault(str(candidate.get("promo_id", "")), candidate)

    prepared: list[dict[str, Any]] = []
    for index, (promo_id, original) in enumerate(original_candidates.items(), start=1):
        candidate = ranked_by_id.get(promo_id, original)
        quality = quality_by_id.get(promo_id, {})
        segments = candidate.get("segments", [])
        episode_ids = list(dict.fromkeys(int(item.get("episode_id", 0)) for item in segments))
        evaluation = candidate.get("evaluation", {})
        criterion_scores = candidate.get("criterion_scores", {})
        narrative = quality.get("narrative_evaluation") or {}
        hook = narrative.get("opening_hook") or {}
        cliff = narrative.get("ending_cliffhanger") or {}
        standalone = narrative.get("standalone_understandability") or {}
        coherence = narrative.get("narrative_coherence") or {}
        summary = candidate.get("combined_summary") or candidate.get("overall_summary") or ""
        prepared.append({
            "promo_id": promo_id or _candidate_key(candidate, index),
            "promo_type": "multi_segment" if candidate.get("promo_type") == "multi_segment" or len(segments) > 1 else "continuous",
            "episode_ids": ",".join(map(str, episode_ids)),
            "segment_timestamps": " | ".join(
                f"{item.get('segment_order', n)}: E{item.get('episode_id')}/S{','.join(map(str, item.get('scene_ids', [])))} "
                f"{item.get('start_time')}-{item.get('end_time')}"
                for n, item in enumerate(segments, start=1)
            ),
            "segments_json": json.dumps(segments, ensure_ascii=False),
            "total_duration": candidate.get("total_duration"),
            "promo_summary": summary,
            "hook_evidence": _evidence((candidate.get("hook") or {}).get("evidence")),
            "cliffhanger_evidence": _evidence((candidate.get("cliffhanger") or {}).get("evidence")),
            "automated_rank": candidate.get("rank"),
            "automated_score": candidate.get("raw_score"),
            "automated_quality_category": candidate.get("quality_category"),
            "automated_quality_score": quality.get("narrative_average_score"),
            "automated_quality_decision": quality.get("quality_decision"),
            "automated_hook_score": hook.get("score", evaluation.get("hook_score", criterion_scores.get("hook_strength"))),
            "automated_cliffhanger_score": cliff.get("score", evaluation.get("cliffhanger_score", criterion_scores.get("cliffhanger_strength"))),
            "automated_standalone_score": standalone.get("score", evaluation.get("standalone_context_score", criterion_scores.get("standalone_understandability"))),
            "automated_coherence_score": coherence.get("score", evaluation.get("narrative_coherence_score", criterion_scores.get("narrative_coherence"))),
            "source_text": _source_for_candidate(candidate, episode_map, source_text_limit),
            "failure_labels": "; ".join(sorted({str(item.get("category")) for item in quality.get("failure_labels", [])})),
            "technical_validation_status": quality.get("technical_status", "unavailable"),
            "human_reviewer_id": "",
            "human_overall_label": "",
            "human_hook_rating": "",
            "human_cliffhanger_rating": "",
            "human_standalone_rating": "",
            "human_coherence_rating": "",
            "human_emotional_impact_rating": "",
            "human_promotional_potential_rating": "",
            "human_feedback": "",
            "human_failure_categories": "",
            "human_incorrect_timestamps": "",
            "human_missing_context": "",
        })

    selected = _sample_candidates(prepared, sample_size)
    for row in selected:
        previous = previous_by_id.get(str(row["promo_id"]), {})
        # Preserve any existing human fields from an earlier exported/imported review sheet.
        for field in REVIEW_DATASET_FIELDS:
            if field.startswith("human_") and previous.get(field) not in (None, ""):
                row[field] = previous[field]
    selected.sort(key=lambda row: (row.get("automated_rank") is None, row.get("automated_rank") or 10**9, row["promo_id"]))
    return selected


def write_review_dataset(rows: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_DATASET_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_human_reviews(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(value, dict):
        value = value.get("reviews", [])
    if not isinstance(value, list):
        return []
    valid: list[dict[str, Any]] = []
    for item in value:
        try:
            valid.append(HumanReview.model_validate(item).model_dump(mode="json"))
        except Exception:
            continue
    return valid


def save_human_review(review: dict[str, Any], json_path: str | Path, csv_path: str | Path) -> list[dict[str, Any]]:
    """Upsert a review by (promo_id, reviewer_id), retaining other reviewers."""
    normalized = HumanReview.model_validate(review).model_dump(mode="json")
    records = load_human_reviews(json_path)
    index = next((i for i, item in enumerate(records) if item["promo_id"] == normalized["promo_id"] and item["reviewer_id"] == normalized["reviewer_id"]), None)
    if index is None:
        records.append(normalized)
    else:
        records[index] = normalized
    records.sort(key=lambda item: (item["promo_id"], item["reviewer_id"]))
    json_file = Path(json_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text(json.dumps({"reviews": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with csv_file.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in records:
            row = dict(item)
            row["failure_categories"] = "; ".join(item.get("failure_categories", []))
            writer.writerow(row)
    return records


def import_human_reviews(uploaded: str | Path | Iterable[dict[str, Any]], json_path: str | Path, csv_path: str | Path) -> int:
    if isinstance(uploaded, (str, Path)):
        value = json.loads(Path(uploaded).read_text(encoding="utf-8")) if str(uploaded).lower().endswith(".json") else list(csv.DictReader(Path(uploaded).open(encoding="utf-8-sig", newline="")))
        records = value.get("reviews", []) if isinstance(value, dict) else value
    else:
        records = list(uploaded)
    accepted = 0
    for item in records:
        try:
            save_human_review(item, json_path, csv_path)
            accepted += 1
        except Exception:
            continue
    return accepted


def _average(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    return ranks


def _pearson_pairs(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs, ys = [item[0] for item in pairs], [item[1] for item in pairs]
    mx, my = mean(xs), mean(ys)
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return round(sum((x - mx) * (y - my) for x, y in pairs) / (dx * dy), 4)


def _spearman_pairs(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    rx = _average_ranks([item[0] for item in pairs])
    ry = _average_ranks([item[1] for item in pairs])
    return _pearson_pairs(list(zip(rx, ry)))


def _kendall_tau_b(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for left, right in itertools.combinations(pairs, 2):
        dx, dy = left[0] - right[0], left[1] - right[1]
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            ties_x += 1
        elif dy == 0:
            ties_y += 1
        elif dx * dy > 0:
            concordant += 1
        else:
            discordant += 1
    denominator = ((concordant + discordant + ties_x) * (concordant + discordant + ties_y)) ** 0.5
    return round((concordant - discordant) / denominator, 4) if denominator else None


def _cohen_kappa(left: list[str], right: list[str], labels: list[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    total = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / total
    expected = sum((left.count(label) / total) * (right.count(label) / total) for label in labels)
    return round((observed - expected) / (1 - expected), 4) if expected != 1 else None


def _fleiss_kappa(rows: list[list[str]], labels: list[str]) -> float | None:
    if len(rows) < 2 or not rows or len({len(row) for row in rows}) != 1 or len(rows[0]) < 2:
        return None
    n = len(rows[0])
    if not n:
        return None
    proportions: list[float] = []
    for row in rows:
        counts = Counter(row)
        proportions.append((sum(counts[label] ** 2 for label in labels) - n) / (n * (n - 1)))
    p = {label: sum(row.count(label) for row in rows) / (len(rows) * n) for label in labels}
    expected = sum(value * value for value in p.values())
    observed = mean(proportions)
    return round((observed - expected) / (1 - expected), 4) if expected != 1 else None


def _human_rollup(reviews: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_promo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in reviews:
        by_promo[str(review["promo_id"])].append(review)
    result: dict[str, dict[str, Any]] = {}
    label_score = {"Poor": 1.0, "Average": 2.0, "Strong": 3.0}
    for promo_id, entries in by_promo.items():
        labels = [str(item["overall_label"]) for item in entries]
        label_counts = Counter(labels)
        majority = sorted(label_counts, key=lambda label: (-label_counts[label], label))[0]
        aggregate: dict[str, Any] = {
            "review_count": len(entries),
            "human_label": majority,
            "human_label_score": mean(label_score[label] for label in labels),
            "comments": [item.get("comments", "") for item in entries if item.get("comments")],
            "reviews": entries,
        }
        for rating in REVIEW_RATINGS:
            values = [float(item[rating]) for item in entries if item.get(rating) is not None]
            aggregate[rating] = _average(values)
        aggregate["overall_rating"] = aggregate["promotional_potential_rating"] or _average([float(value) for value in (aggregate["hook_rating"], aggregate["cliffhanger_rating"], aggregate["standalone_rating"], aggregate["coherence_rating"], aggregate["emotional_impact_rating"]) if value is not None])
        result[promo_id] = aggregate
    return result


def _rating_comparison(auto_score: Any, human_score: Any, *, ai_min: float = 1, ai_max: float = 10, human_min: float = 1, human_max: float = 5) -> float | None:
    if auto_score is None or human_score is None:
        return None
    try:
        ai = float(auto_score)
        human = float(human_score)
    except (TypeError, ValueError):
        return None
    scaled = human_min + (ai - ai_min) / (ai_max - ai_min) * (human_max - human_min)
    return round(scaled, 4)


def _classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ["negative", "positive"]
    tp = fp = tn = fn = 0
    excluded = 0
    for row in rows:
        human_label = row.get("human_label")
        if human_label not in {"Strong", "Average", "Poor"}:
            excluded += 1
            continue
        if human_label == "Average":
            excluded += 1
            continue
        actual = human_label == "Strong"
        predicted = row.get("automated_quality_category") in {"Excellent", "Strong"}
        if actual and predicted: tp += 1
        elif predicted and not actual: fp += 1
        elif not predicted and not actual: tn += 1
        else: fn += 1
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    return {
        "positive_definition": "Human Strong; AI Excellent or Strong",
        "negative_definition": "Human Poor or Average; AI Moderate or Weak",
        "average_human_label_excluded_from_binary_metrics": True,
        "confusion_matrix": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
        "accuracy": round((tp + tn) / total, 4) if total else None,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "included_candidates": total,
        "excluded_unreviewed_or_average": excluded,
        "warning": None if total else "Unavailable until reviewed promos have Strong or Poor human labels.",
    }


def _technical_metrics(candidates: list[dict[str, Any]], quality_report: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    episode_map = {int(item.get("episode_id", 0)): item for item in episodes}
    total_segments = 0
    valid_timestamps = valid_episode_refs = valid_scene_refs = 0
    duration_compliant = 0
    duplicate_candidates = 0
    for candidate in candidates:
        segments = candidate.get("segments", [])
        total_seconds = 0.0
        candidate_timestamp_ok = bool(segments)
        candidate_episode_ok = bool(segments)
        candidate_scene_ok = bool(segments)
        for segment in segments:
            total_segments += 1
            start = parse_timestamp(segment.get("start_time"))
            end = parse_timestamp(segment.get("end_time"))
            if start is None or end is None or start >= end:
                candidate_timestamp_ok = False
            else:
                total_seconds += end - start
            episode_id = int(segment.get("episode_id", -1))
            episode = episode_map.get(episode_id)
            if episode is None:
                candidate_episode_ok = False
                candidate_scene_ok = False
                continue
            scenes = {int(item.get("scene_id", -1)) for item in episode.get("scenes", [])}
            if any(int(scene_id) not in scenes for scene_id in segment.get("scene_ids", [])):
                candidate_scene_ok = False
        valid_timestamps += int(candidate_timestamp_ok)
        valid_episode_refs += int(candidate_episode_ok)
        valid_scene_refs += int(candidate_scene_ok)
        duration_compliant += int(20 <= total_seconds <= 60)
        flags = candidate.get("duplicate_or_overlap_flags", [])
        duplicate_candidates += int(any(flag.get("type") in {"exact_duplicate", "high_overlap"} for flag in flags))
    total_candidates = len(candidates)
    qresults = quality_report.get("technical_validation_results", [])
    status_map = {str(item.get("promo_id")): item.get("status") for item in qresults}
    checked = [status_map.get(str(item.get("promo_id"))) for item in candidates]
    present = [value for value in checked if value]
    passing = sum(value in {"PASS", "PASS_WITH_WARNINGS"} for value in present)
    return {
        "timestamp_validity_rate": round(100 * valid_timestamps / total_candidates, 2) if total_candidates else None,
        "timestamp_validity_denominator": total_candidates,
        "duration_compliance_rate": round(100 * duration_compliant / total_candidates, 2) if total_candidates else None,
        "duration_compliance_denominator": total_candidates,
        "episode_reference_validity_rate": round(100 * valid_episode_refs / total_candidates, 2) if total_candidates else None,
        "scene_reference_validity_rate": round(100 * valid_scene_refs / total_candidates, 2) if total_candidates else None,
        "duplicate_candidate_rate": round(100 * duplicate_candidates / total_candidates, 2) if total_candidates else None,
        "module5_technical_pass_rate": round(100 * passing / len(present), 2) if present else None,
        "module5_denominator": len(present),
        "segments_checked": total_segments,
        "warning": None if total_candidates else "No candidates available for technical rate calculation.",
    }


def compute_system_evaluation(
    candidates: list[dict[str, Any]],
    quality_report: dict[str, Any],
    episodes: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    *,
    cutoffs: list[int] | None = None,
    module6_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Calculate metrics only from recorded candidates/reviews; missing data stays null."""
    cutoffs = cutoffs or [5, 10, 20]
    human_by_id = _human_rollup(reviews)
    quality_by_id = {str(item.get("promo_id")): item for item in quality_report.get("candidate_results", [])}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        promo_id = str(candidate.get("promo_id", ""))
        quality = quality_by_id.get(promo_id, {})
        human = human_by_id.get(promo_id, {})
        auto_rank = candidate.get("rank")
        if auto_rank is None:
            auto_rank = candidate.get("diversity_adjusted_rank")
        human_label = human.get("human_label")
        ai_category = candidate.get("quality_category")
        agreement = "unreviewed" if not human else (
            "agree" if ((human_label == "Strong") == (ai_category in {"Excellent", "Strong"}) and human_label != "Average") else "disagree"
        )
        narrative = quality.get("narrative_evaluation") or {}
        eval_fields = candidate.get("evaluation", {})
        rows.append({
            "promo_id": promo_id,
            "promo_type": "multi_segment" if candidate.get("promo_type") == "multi_segment" or len(candidate.get("segments", [])) > 1 else "continuous",
            "ai_rank": auto_rank,
            "ai_score": candidate.get("raw_score"),
            "ai_quality_category": ai_category,
            "ai_quality_decision": quality.get("quality_decision"),
            "ai_narrative_score": quality.get("narrative_average_score"),
            "ai_hook_score": (narrative.get("opening_hook") or {}).get("score", eval_fields.get("hook_score")),
            "ai_cliffhanger_score": (narrative.get("ending_cliffhanger") or {}).get("score", eval_fields.get("cliffhanger_score")),
            "ai_standalone_score": (narrative.get("standalone_understandability") or {}).get("score", eval_fields.get("standalone_context_score")),
            "ai_coherence_score": (narrative.get("narrative_coherence") or {}).get("score", eval_fields.get("narrative_coherence_score")),
            "ai_emotional_score": (narrative.get("emotional_engagement") or {}).get("score", eval_fields.get("emotional_intensity_score")),
            "human_label": human_label,
            "human_overall_rating": human.get("overall_rating"),
            "human_hook_rating": human.get("hook_rating"),
            "human_cliffhanger_rating": human.get("cliffhanger_rating"),
            "human_standalone_rating": human.get("standalone_rating"),
            "human_coherence_rating": human.get("coherence_rating"),
            "human_emotional_impact_rating": human.get("emotional_impact_rating"),
            "review_count": human.get("review_count", 0),
            "agreement_status": agreement,
            "technical_validation_status": quality.get("technical_status", "unavailable"),
            "failure_labels": "; ".join(sorted({str(item.get("category")) for item in quality.get("failure_labels", [])})),
            "reviewer_feedback": " | ".join(human.get("comments", [])),
            "selected": candidate.get("selection_status") == "selected",
        })

    reviewed = [row for row in rows if row["human_label"] in {"Strong", "Average", "Poor"}]
    review_count = len(reviews)
    reviewer_ids = sorted({str(item.get("reviewer_id", "anonymous")) for item in reviews})
    label_ord = {"Poor": 1.0, "Average": 2.0, "Strong": 3.0}
    correlation_pairs = [(float(row["ai_rank"]), label_ord[row["human_label"]]) for row in reviewed if row.get("ai_rank") is not None]
    metrics_by_rating: dict[str, Any] = {}
    rating_map = {
        "hook_rating": "ai_hook_score",
        "cliffhanger_rating": "ai_cliffhanger_score",
        "standalone_rating": "ai_standalone_score",
        "coherence_rating": "ai_coherence_score",
        "emotional_impact_rating": "ai_emotional_score",
        "promotional_potential_rating": "ai_narrative_score",
    }
    for human_field, ai_field in rating_map.items():
        pairs = []
        for row in rows:
            human_value, ai_value = row.get(human_field), row.get(ai_field)
            scaled = _rating_comparison(ai_value, human_value)
            if human_value is not None and scaled is not None:
                pairs.append((scaled, float(human_value)))
        diffs = [abs(left - right) for left, right in pairs]
        squared = [(left - right) ** 2 for left, right in pairs]
        metrics_by_rating[human_field] = {
            "sample_count": len(pairs),
            "mae": round(mean(diffs), 4) if diffs else None,
            "rmse": round(math.sqrt(mean(squared)), 4) if squared else None,
            "pearson_correlation": _pearson_pairs(pairs),
            "note": "AI 1-10 score linearly rescaled to human 1-5 scale; unavailable if either score is missing.",
        }

    rank_sorted = sorted(reviewed, key=lambda row: (row.get("ai_rank") is None, row.get("ai_rank") or 10**9, row["promo_id"]))
    precision_at_k: dict[str, Any] = {}
    for k in cutoffs:
        subset = rank_sorted[:max(0, k)]
        denominator = len(subset)
        strong_count = sum(row["human_label"] == "Strong" for row in subset)
        precision_at_k[str(k)] = {
            "value": round(strong_count / denominator, 4) if denominator else None,
            "strong_reviewed_in_top_k": strong_count,
            "reviewed_denominator_in_top_k": denominator,
            "requested_k": k,
            "note": "Unreviewed promos are excluded; denominator is reviewed promos within the AI top-K, not unreviewed slots.",
        }

    classification = _classification_metrics(rows)
    # Ranking quality: use only human-reviewed candidates; unreviewed records are never negative labels.
    ndcg: dict[str, Any] = {}
    for k in cutoffs:
        subset = rank_sorted[:max(0, k)]
        gains = [max(0.0, label_ord[row["human_label"]] - 1) for row in subset]
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal = sorted(gains, reverse=True)
        idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal))
        ndcg[str(k)] = {"value": round(dcg / idcg, 4) if idcg else None, "reviewed_count": len(subset), "note": "Graded gain mapping: Poor=0, Average=1, Strong=2; calculated on reviewed candidates only."}

    per_promo_reviews: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in reviews:
        per_promo_reviews[str(review["promo_id"])].append(review)
    label_names = ["Strong", "Average", "Poor"]
    pairwise_agreement: dict[str, Any] = {"method": None, "value": None, "reviewed_promos_with_multiple_reviewers": 0, "warning": None}
    repeated = {promo_id: entries for promo_id, entries in per_promo_reviews.items() if len(entries) >= 2}
    if repeated:
        pairwise_agreement["reviewed_promos_with_multiple_reviewers"] = len(repeated)
        if len(reviewer_ids) == 2:
            left_id, right_id = reviewer_ids
            left_labels: list[str] = []
            right_labels: list[str] = []
            for entries in repeated.values():
                by_reviewer = {item["reviewer_id"]: item["overall_label"] for item in entries}
                if left_id in by_reviewer and right_id in by_reviewer:
                    left_labels.append(by_reviewer[left_id])
                    right_labels.append(by_reviewer[right_id])
            pairwise_agreement.update(method="Cohen's kappa", value=_cohen_kappa(left_labels, right_labels, label_names), compared_promos=len(left_labels))
        else:
            rows_by_rater: list[list[str]] = []
            for entries in repeated.values():
                rows_by_rater.append([str(item["overall_label"]) for item in entries])
            if len({len(row) for row in rows_by_rater}) == 1:
                pairwise_agreement.update(method="Fleiss' kappa", value=_fleiss_kappa(rows_by_rater, label_names), compared_promos=len(rows_by_rater))
            else:
                pairwise_agreement["warning"] = "Unequal reviewer counts per promo; Fleiss kappa unavailable without a balanced rating matrix."
    else:
        pairwise_agreement["warning"] = "Need at least two reviewers on the same promos."

    metric_pairs = [(float(row["ai_rank"]), label_ord[row["human_label"]]) for row in reviewed if row.get("ai_rank") is not None]
    human_ratings = [float(row["human_overall_rating"]) for row in reviewed if row.get("human_overall_rating") is not None]
    decision_counts = Counter(str(item.get("overall_label")) for item in reviews)
    technical = _technical_metrics(candidates, quality_report, episodes)
    timestamps = [item.get("elapsed_seconds") for item in [module6_metadata or {}] if isinstance(item.get("elapsed_seconds"), (int, float))]
    processing = {
        "module6_elapsed_seconds": (module6_metadata or {}).get("elapsed_seconds"),
        "module2_elapsed_seconds": None,
        "module3_elapsed_seconds": None,
        "module4_elapsed_seconds": None,
        "module5_elapsed_seconds": None,
        "total_pipeline_elapsed_seconds": None,
        "average_seconds_per_candidate": round(sum(timestamps) / max(1, len(candidates)), 4) if timestamps and candidates else None,
        "llm_api_calls": (module6_metadata or {}).get("llm_requests"),
        "token_usage": None,
        "estimated_api_cost": None,
        "cache_hits": (module6_metadata or {}).get("cache_hits"),
        "failure_and_retry_counts": (module6_metadata or {}).get("errors", []),
        "note": "Only Module 6 timing/request/cache data is currently instrumented. Other module timings, token counts, and pricing are unavailable, not estimated.",
    }
    counts_by_type: dict[str, Any] = {}
    for promo_type in ("continuous", "multi_segment"):
        group = [row for row in rows if row["promo_type"] == promo_type]
        ratings = [float(row["human_overall_rating"]) for row in group if row["human_overall_rating"] is not None]
        counts_by_type[promo_type] = {"candidate_count": len(group), "reviewed_count": len([row for row in group if row["human_label"]]), "average_human_overall_rating": _average(ratings)}

    disagreement_cases: list[dict[str, Any]] = []
    if reviewed:
        ai_order = sorted(reviewed, key=lambda row: (row.get("ai_rank") is None, row.get("ai_rank") or 10**9))
        ai_rank_by_id = {row["promo_id"]: index + 1 for index, row in enumerate(ai_order)}
        max_rank = max(ai_rank_by_id.values(), default=1)
        for row in reviewed:
            label = row["human_label"]
            high_rank_poor = label == "Poor" and ai_rank_by_id.get(row["promo_id"], max_rank) <= max(5, len(reviewed) // 4)
            low_rank_strong = label == "Strong" and ai_rank_by_id.get(row["promo_id"], 1) > max(5, 3 * len(reviewed) // 4)
            if high_rank_poor or low_rank_strong:
                disagreement_cases.append({
                    "promo_id": row["promo_id"],
                    "promo_type": row["promo_type"],
                    "ai_rank": row.get("ai_rank"),
                    "ai_score": row.get("ai_score"),
                    "human_label": label,
                    "human_overall_rating": row.get("human_overall_rating"),
                    "human_ratings": {field: row.get(field) for field in ("human_hook_rating", "human_cliffhanger_rating", "human_standalone_rating", "human_coherence_rating", "human_emotional_impact_rating")},
                    "reviewer_feedback": row.get("reviewer_feedback"),
                    "likely_responsible_module": "Module 6" if row["promo_type"] == "multi_segment" else "Module 4 ranking calibration (hypothesis, not causal proof)",
                    "reason": "High-ranked human Poor" if high_rank_poor else "Low-ranked human Strong",
                })

    metrics = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "generated_candidate_count": len(candidates),
        "human_review_dataset_size": len(reviews),
        "reviewed_promo_count": len(reviewed),
        "review_submission_count": review_count,
        "reviewer_count": len(reviewer_ids),
        "approval_counts": {"strong": decision_counts.get("Strong", 0), "average": decision_counts.get("Average", 0), "poor": decision_counts.get("Poor", 0)},
        "strong_promo_percentage": round(100 * decision_counts.get("Strong", 0) / review_count, 2) if review_count else None,
        "poor_promo_percentage": round(100 * decision_counts.get("Poor", 0) / review_count, 2) if review_count else None,
        "average_human_ratings": {field: _average([float(item[field]) for item in reviews if item.get(field) is not None]) for field in REVIEW_RATINGS},
        "human_ai_agreement": {
            "ordinal_mapping": {"Poor": 1, "Average": 2, "Strong": 3},
            "spearman_rank_correlation": _spearman_pairs(metric_pairs),
            "kendall_tau_b": _kendall_tau_b(metric_pairs),
            "reviewed_candidate_count": len(metric_pairs),
            "score_comparison": metrics_by_rating,
            "quality_classification": classification,
            "limitations": "Correlation is null unless at least two reviewed candidates have AI ranks; classification excludes unreviewed and Average labels from the binary confusion matrix.",
        },
        "precision_at_k": precision_at_k,
        "ndcg_at_k": ndcg,
        "inter_reviewer_agreement": pairwise_agreement,
        "technical_metrics": technical,
        "performance_metrics": processing,
        "continuous_vs_multi_segment": counts_by_type,
        "limitations": [
            "No human labels or scores are inferred for unreviewed promos.",
            "Candidate acceptance and reviewer metrics are unavailable until humans submit annotations.",
            "Token usage and API cost are unavailable because provider usage/pricing telemetry is not recorded by the current pipeline.",
        ] if not reviews else ["Unreviewed promos are excluded from human quality and agreement calculations."],
    }

    quality_map = quality_by_id
    by_category: Counter[str] = Counter()
    by_module: Counter[str] = Counter()
    failure_items: list[dict[str, Any]] = []
    for candidate in candidates:
        promo_id = str(candidate.get("promo_id", ""))
        quality = quality_map.get(promo_id, {})
        human = human_by_id.get(promo_id, {})
        failures = quality.get("failure_labels", [])
        if quality.get("technical_errors"):
            failures = [*failures, *quality["technical_errors"]]
        for failure in failures:
            category = str(failure.get("category", "unknown"))
            by_category[category] += 1
            if failure.get("category") in {"invalid_timestamp", "invalid_timestamp_order", "segment_duration", "missing_episode", "missing_scene"}:
                module = "Module 3" if candidate.get("promo_type") != "multi_segment" else "Module 6"
            elif candidate.get("promo_type") == "multi_segment" and failure.get("category") in {"confusing_transitions", "weak_cliffhanger", "missing_context"}:
                module = "Module 6"
            else:
                module = "Module 5" if failure.get("category") in {"weak_opening_hook", "weak_cliffhanger", "missing_context", "confusing_transitions", "low_emotional_intensity"} else "undetermined"
            by_module[module] += 1
            failure_items.append({
                "promo_id": promo_id,
                "failure_category": category,
                "severity": failure.get("severity", "unknown"),
                "explanation": failure.get("explanation", failure.get("message", "")),
                "available_human_feedback": human.get("comments", []),
                "likely_responsible_module": module,
                "responsibility_note": "Module attribution is a hypothesis from the failure type and promo type, not proof of causation." if module != "undetermined" else "Insufficient evidence to assign responsibility to a module.",
            })

    disagreements = disagreement_cases
    human_weakness_counts: Counter[str] = Counter()
    for review in reviews:
        human_weakness_counts.update(review.get("failure_categories", []))
    high_ranked_rejected = [row for row in rows if row.get("human_label") == "Poor" and row.get("ai_rank") is not None and row["ai_rank"] <= 5]
    low_ranked_approved = [row for row in rows if row.get("human_label") == "Strong" and row.get("ai_rank") is not None and row["ai_rank"] > 5]
    recommendations: list[str] = []
    if not reviews:
        recommendations.append("Collect human reviews before drawing conclusions about AI-human agreement or pipeline quality.")
    if human_weakness_counts:
        recommendations.extend(f"Investigate reviewer-reported '{label}' ({count} report(s)) against the responsible module output." for label, count in human_weakness_counts.most_common())
    if high_ranked_rejected:
        recommendations.append("Review Module 4 score calibration against high-ranked candidates labeled Poor by humans.")
    if any(row["promo_type"] == "multi_segment" for row in reviewed if row.get("human_label") == "Poor"):
        recommendations.append("Inspect Module 6 transitions and context for multi-segment promos labeled Poor.")
    if reviews and not recommendations:
        recommendations.append("No recurring human-reported weaknesses are established by current annotations.")
    error_analysis = {
        "reviewed_candidate_count": len(reviewed),
        "human_ai_disagreements": disagreements,
        "high_ranked_rejected_candidates": [row["promo_id"] for row in high_ranked_rejected],
        "low_ranked_strong_candidates": [row["promo_id"] for row in low_ranked_approved],
        "common_human_reported_weaknesses": [{"category": key, "count": value} for key, value in human_weakness_counts.most_common()],
        "recorded_automated_failures": failure_items,
        "failure_frequency_by_category": dict(by_category),
        "failure_frequency_by_module": dict(by_module),
        "recommendations": recommendations,
        "warning": None if reviews else "No human evaluations exist; disagreement and reviewer-weakness findings are empty, not zero-quality evidence.",
    }
    return metrics, error_analysis, rows


def write_evaluation_outputs(
    metrics: dict[str, Any],
    error_analysis: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    metrics_path: str | Path = "outputs/evaluation/evaluation_metrics.json",
    csv_path: str | Path = "outputs/evaluation/evaluation_report.csv",
    error_path: str | Path = "outputs/evaluation/error_analysis.json",
    summary_path: str | Path = "outputs/evaluation/evaluation_summary.md",
) -> None:
    for path_value, payload in ((metrics_path, metrics), (error_path, error_analysis)):
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path_obj = Path(csv_path)
    csv_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with csv_path_obj.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "Promo ID": row.get("promo_id"),
                "Promo type": row.get("promo_type"),
                "AI rank": row.get("ai_rank"),
                "AI score": row.get("ai_score"),
                "AI quality category": row.get("ai_quality_category"),
                "Human label": row.get("human_label"),
                "Human overall rating": row.get("human_overall_rating"),
                "Human hook rating": row.get("human_hook_rating"),
                "Human cliffhanger rating": row.get("human_cliffhanger_rating"),
                "Human standalone rating": row.get("human_standalone_rating"),
                "Human coherence rating": row.get("human_coherence_rating"),
                "Human emotional impact rating": row.get("human_emotional_impact_rating"),
                "Agreement status": row.get("agreement_status"),
                "Technical validation status": row.get("technical_validation_status"),
                "Failure labels": row.get("failure_labels"),
                "Reviewer feedback": row.get("reviewer_feedback"),
            })
    reviewed_count = metrics.get("reviewed_promo_count", 0)
    human_mean = metrics.get("average_human_ratings", {}).get("promotional_potential_rating")
    summary = [
        "# Human Evaluation and Pipeline QA Summary",
        "",
        f"- Evaluation generated: {metrics.get('evaluation_timestamp')}",
        f"- Candidates in scope: {metrics.get('generated_candidate_count', 0)}",
        f"- Human-reviewed candidates: {reviewed_count}",
        f"- Individual review submissions: {metrics.get('review_submission_count', 0)}",
        f"- Strong / Average / Poor submissions: {metrics.get('approval_counts')}",
        f"- Average human promotional-potential rating: {human_mean if human_mean is not None else 'Unavailable (no human ratings yet)'}",
        f"- Timestamp validity: {metrics.get('technical_metrics', {}).get('timestamp_validity_rate') if metrics.get('technical_metrics', {}).get('timestamp_validity_rate') is not None else 'Unavailable'}",
        f"- Duration compliance: {metrics.get('technical_metrics', {}).get('duration_compliance_rate') if metrics.get('technical_metrics', {}).get('duration_compliance_rate') is not None else 'Unavailable'}",
        f"- Human/AI Spearman: {metrics.get('human_ai_agreement', {}).get('spearman_rank_correlation') if metrics.get('human_ai_agreement', {}).get('spearman_rank_correlation') is not None else 'Unavailable (insufficient human reviews)'}",
        "",
        "## Limitations",
        *[f"- {item}" for item in metrics.get("limitations", [])],
        "",
        "## Current evidence-based recommendations",
        *[f"- {item}" for item in error_analysis.get("recommendations", [])],
        "",
        "No human ratings or judgments are generated automatically; reviewers must submit them through the review app.",
    ]
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text("\n".join(summary) + "\n", encoding="utf-8")


__all__ = [
    "REVIEW_RATINGS", "HUMAN_REVIEW_FIELDS", "REVIEW_DATASET_FIELDS", "EVALUATION_CSV_FIELDS",
    "build_review_dataset", "write_review_dataset", "load_human_reviews", "save_human_review",
    "import_human_reviews", "compute_system_evaluation", "write_evaluation_outputs",
]
