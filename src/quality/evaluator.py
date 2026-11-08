"""Independent narrative evaluation and final quality reporting (Module 5)."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from threading import Lock
from typing import Any

from pydantic import ValidationError

from src.config import get_settings
from src.llm.factory import create_provider
from src.quality.prompts import NARRATIVE_QUALITY_TEMPLATE
from src.quality.validator import validate_candidate, validate_unique_ids
from src.schemas.quality_evaluation import FailureFinding, NarrativeQualityEvaluation
from src.utils.logging_config import get_logger
from src.utils.retry import RequestPacer, is_rate_limit_error, retry_delay_seconds
from src.utils.timestamp import parse_timestamp

logger = get_logger(__name__)
PROMPT_VERSION = "independent-narrative-quality-v1"
QUALITY_CSV_FIELDS = [
    "Promo ID", "Episode ID", "Original rank", "Final score", "Technical validation status",
    "Narrative quality score", "Hook score", "Cliffhanger score", "Standalone score",
    "Failure labels", "Quality decision", "Short explanation",
]
_CACHE_LOCK = Lock()


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read quality cache at %s", path)
        return {}


def _save_cache(path: Path, cache: dict[str, Any]) -> None:
    with _CACHE_LOCK:
        merged = _load_cache(path)
        merged.update(cache)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def _source_text(candidate: dict[str, Any], episodes_by_id: dict[int, dict[str, Any]], limit: int = 4500) -> str:
    evidence: list[str] = []
    for segment in candidate.get("segments", []):
        episode_id = int(segment.get("episode_id", 0))
        episode = episodes_by_id.get(episode_id, {})
        scene_ids = {int(value) for value in segment.get("scene_ids", [])}
        start = parse_timestamp(segment.get("start_time"))
        end = parse_timestamp(segment.get("end_time"))
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
                    timestamp = event.get("start_time", "?")
                    character = event.get("character", "")
                    text = event.get("text", "")
                    if text:
                        evidence.append(f"{timestamp} {kind} {character}: {text}")
    joined = "\n".join(evidence)
    if len(joined) <= limit:
        return joined or "No timestamped dialogue/actions were found in the candidate's referenced source scenes."
    marker = "\n...[middle omitted]...\n"
    available = limit - len(marker)
    head = available * 2 // 3
    return joined[:head] + marker + joined[-(available - head):]


def _cache_key(candidate: dict[str, Any], source: str, provider: Any, threshold: float) -> str:
    payload = {
        "candidate": candidate,
        "source": source,
        "prompt_version": PROMPT_VERSION,
        "provider": provider.provider_name,
        "model": getattr(provider, "model", "unknown"),
        "temperature": getattr(provider, "temperature", None),
        "threshold": threshold,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _is_rate_limit_error(error: str | None) -> bool:
    return is_rate_limit_error(error)


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _flatten_strings(item)]
    return []


def _normalize_narrative_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize nested evidence arrays without creating or changing evidence text."""
    normalized = dict(payload)
    criterion_keys = (
        "opening_hook", "ending_cliffhanger", "standalone_understandability",
        "narrative_coherence", "emotional_engagement",
    )
    for key in criterion_keys:
        criterion = normalized.get(key)
        if isinstance(criterion, dict):
            criterion = dict(criterion)
            criterion["evidence"] = _flatten_strings(criterion.get("evidence", []))
            criterion["weaknesses"] = _flatten_strings(criterion.get("weaknesses", []))
            normalized[key] = criterion
    findings = normalized.get("failure_findings")
    if isinstance(findings, list):
        fixed_findings = []
        for finding in findings:
            if isinstance(finding, dict):
                finding = dict(finding)
                finding["evidence"] = _flatten_strings(finding.get("evidence", []))
            fixed_findings.append(finding)
        normalized["failure_findings"] = fixed_findings
    if not str(normalized.get("overall_explanation", "")).strip():
        criterion_explanations = [
            str(normalized[key].get("explanation", "")).strip()
            for key in criterion_keys
            if isinstance(normalized.get(key), dict)
            and str(normalized[key].get("explanation", "")).strip()
        ]
        if criterion_explanations:
            normalized["overall_explanation"] = ("Criterion summary: " + " ".join(criterion_explanations))[:1000]
            normalized["overall_explanation_derived"] = True
    return normalized


def _narrative_evaluate(
    candidate: dict[str, Any],
    episodes_by_id: dict[int, dict[str, Any]],
    provider: Any,
    cache_path: Path,
    retries: int,
    backoff_seconds: float,
    threshold: float,
    use_cache: bool = True,
    pacer: RequestPacer | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    source_text = _source_text(candidate, episodes_by_id)
    cache = _load_cache(cache_path) if use_cache else {}
    key = _cache_key(candidate, source_text, provider, threshold)
    cached = cache.get(key) if use_cache else None
    if cached:
        try:
            return NarrativeQualityEvaluation.model_validate(cached).model_dump(), None
        except ValidationError:
            pass

    segments = candidate.get("segments") or [{}]
    segments_context = "\n".join(
        f"Segment {index}: episode {segment.get('episode_id')}, scenes {segment.get('scene_ids', [])}, "
        f"{segment.get('start_time', 'unknown')} to {segment.get('end_time', 'unknown')}"
        for index, segment in enumerate(segments, start=1)
    )
    prompt = NARRATIVE_QUALITY_TEMPLATE.format(
        duration_seconds=candidate.get("total_duration", "unknown"),
        promo_summary=str(candidate.get("overall_summary", ""))[:300],
        source_text=(
            f"Ordered source segments:\n{segments_context}\n\n" if segments_context else ""
        ) + source_text,
    )
    last_error: str | None = None
    for attempt in range(max(1, retries)):
        try:
            if pacer is not None:
                pacer.wait()
            response = provider.generate_json(prompt, response_schema=NarrativeQualityEvaluation.model_json_schema())
        except Exception as exc:
            last_error = f"Narrative provider call failed: {exc}"
            if _is_rate_limit_error(str(exc)):
                # Quota/rate limits will not improve by retrying this candidate.
                break
            if attempt + 1 < max(1, retries):
                delay = retry_delay_seconds(exc, attempt, backoff_seconds)
                logger.warning("Narrative evaluation request failed; retrying in %.1fs", delay)
                time.sleep(delay)
                continue
            break
        try:
            payload = provider.parse_json_response(response)
            result = NarrativeQualityEvaluation.model_validate(_normalize_narrative_payload(payload))
            if use_cache:
                cache[key] = result.model_dump()
                _save_cache(cache_path, cache)
            return result.model_dump(), None
        except (ValidationError, ValueError, TypeError) as exc:
            last_error = f"Narrative response did not match the required schema: {exc}"
            if attempt + 1 < max(1, retries):
                prompt += "\nCorrect the JSON to match the required schema, use integer scores 1-10, and return JSON only."
                delay = retry_delay_seconds(exc, attempt, backoff_seconds)
                time.sleep(delay)
    return None, last_error or "Narrative evaluation did not return valid structured data."


def _deterministic_findings(candidate: dict[str, Any], episode_map: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(candidate.get("segments", [])):
        episode = episode_map.get(int(segment.get("episode_id", -1)), {})
        start = parse_timestamp(segment.get("start_time"))
        if start is None:
            continue
        cutoff = start + 3
        early_events: list[str] = []
        for scene in episode.get("scenes", []):
            if int(scene.get("scene_id", -1)) not in {int(value) for value in segment.get("scene_ids", [])}:
                continue
            for field in ("dialogue", "actions"):
                for event in scene.get(field, []):
                    event_start = parse_timestamp(event.get("start_time"))
                    if event_start is not None and start <= event_start <= cutoff:
                        early_events.append(str(event.get("text", "")))
        if not early_events:
            findings.append(FailureFinding(
                category="slow_or_empty_opening",
                severity="medium",
                explanation="No timestamped dialogue or action begins within the first three seconds of this segment.",
                evidence=[],
                suggested_improvement="Consider an earlier source-supported dramatic moment; do not alter the original timestamps without review.",
            ).model_dump())
    return findings


def _narrative_score_findings(narrative: dict[str, Any], warning_threshold: float) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    failure_categories = {
        "opening_hook": "weak_opening_hook",
        "ending_cliffhanger": "weak_cliffhanger",
        "standalone_understandability": "missing_context",
        "narrative_coherence": "confusing_transitions",
        "emotional_engagement": "low_emotional_intensity",
    }
    for field, category in failure_categories.items():
        criterion = narrative.get(field) or {}
        score = criterion.get("score")
        if score is None or float(score) >= warning_threshold:
            continue
        findings.append(FailureFinding(
            category=category,
            severity="high" if float(score) <= 3 else "medium",
            explanation=str(criterion.get("explanation", f"{category.replace('_', ' ')} scored below the configured threshold.")),
            evidence=list(criterion.get("evidence", [])),
            suggested_improvement="Review the source-supported clip window and consider a different generated candidate.",
        ).model_dump())
    return findings


def _normalize_failure_finding(finding: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(finding)
    category = re.sub(r"[^a-z0-9]+", "_", str(normalized.get("category", "unknown")).lower()).strip("_")
    aliases = {
        "missing_context": "missing_context",
        "weak_hook": "weak_opening_hook",
        "limited_hook": "weak_opening_hook",
        "slow_opening": "slow_or_empty_opening",
        "weak_ending": "weak_cliffhanger",
        "abrupt_ending": "weak_cliffhanger",
        "repetitive_dialogue": "repetitive_dialogue",
        "weak_emotional_engagement": "low_emotional_intensity",
    }
    normalized["category"] = aliases.get(category, category)
    return normalized


def _pearson(values: list[tuple[float, float]]) -> float | None:
    if len(values) < 2:
        return None
    xs = [item[0] for item in values]
    ys = [item[1] for item in values]
    mean_x, mean_y = mean(xs), mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in values)
    denominator_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    denominator_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if not denominator_x or not denominator_y:
        return None
    return round(numerator / (denominator_x * denominator_y), 4)


def evaluate_ranked_promos(
    ranked_promos: list[dict[str, Any]],
    original_candidates: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    *,
    settings: Any | None = None,
    provider: Any | None = None,
    cache_path: str | Path | None = None,
    use_cache: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Validate all ranked promos and independently evaluate selected promos."""
    active = settings or get_settings()
    llm_provider = provider or create_provider()
    episode_map = {int(item.get("episode_id", 0)): item for item in episodes}
    original_by_id = {str(item.get("promo_id", "")): item for item in original_candidates}
    cache_file = Path(cache_path or active.quality.cache_path)
    duplicate_issues = validate_unique_ids(ranked_promos)
    duplicate_ids = {issue["message"].split()[2] for issue in duplicate_issues}

    records: list[dict[str, Any]] = []
    narrative_pairs: list[tuple[float, float]] = []
    selected_for_llm = [item for item in ranked_promos if item.get("selection_status") == "selected"]
    selected_for_llm = selected_for_llm[:active.quality.maximum_narrative_evaluations]
    evaluated_keys = {str(item.get("promo_id", "")) for item in selected_for_llm}
    pacer = RequestPacer(active.quality.request_interval_seconds)

    def evaluate_selected(candidate: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        promo_id = str(candidate.get("promo_id", "<missing>"))
        logger.info("Independently evaluating narrative quality for %s", promo_id)
        narrative, error = _narrative_evaluate(
            candidate,
            episode_map,
            llm_provider,
            cache_file,
            active.quality.max_narrative_attempts,
            active.llm.retry_backoff_seconds,
            active.quality.narrative_warning_threshold,
            use_cache,
            pacer,
        )
        return promo_id, narrative, error

    narrative_results = []
    rate_limit_reached = False
    for candidate in selected_for_llm:
        if rate_limit_reached:
            narrative_results.append((
                str(candidate.get("promo_id", "<missing>")),
                None,
                "Narrative evaluation skipped after the provider reported a rate limit; retry when quota resets.",
            ))
            continue
        result = evaluate_selected(candidate)
        narrative_results.append(result)
        if _is_rate_limit_error(result[2]):
            rate_limit_reached = True
            logger.warning("Provider rate limit reached; skipping remaining narrative calls in this run.")
    narratives_by_id = {promo_id: (narrative, error) for promo_id, narrative, error in narrative_results}

    for candidate in ranked_promos:
        promo_id = str(candidate.get("promo_id", "<missing>"))
        logger.info("Quality validating promo %s", promo_id)
        source_original = original_by_id.get(promo_id)
        technical = validate_candidate(
            candidate,
            episode_map,
            source_original,
            minimum_duration=active.candidate_generation.min_duration,
            maximum_duration=active.candidate_generation.max_duration,
            minimum_segment_duration=active.multi_segment.minimum_segment_duration,
        )
        if promo_id in duplicate_ids:
            technical["technical_errors"].extend(duplicate_issues)
            technical["technical_status"] = "FAIL"

        narrative: dict[str, Any] | None = None
        narrative_error: str | None = None
        narrative_status = "not_selected"
        if promo_id in evaluated_keys:
            narrative, narrative_error = narratives_by_id.get(promo_id, (None, "Narrative evaluation result missing."))
            narrative_status = "evaluated" if narrative is not None else "unavailable"
        elif candidate.get("selection_status") == "selected":
            narrative_status = "skipped"

        findings = _deterministic_findings(candidate, episode_map)
        if narrative:
            findings.extend(
                _normalize_failure_finding(item.model_dump())
                for item in NarrativeQualityEvaluation.model_validate(narrative).failure_findings
            )
            findings.extend(_narrative_score_findings(narrative, active.quality.narrative_warning_threshold))
        narrative_average: float | None = None
        explanation_parts: list[str] = []
        if narrative:
            criteria = (
                narrative["opening_hook"], narrative["ending_cliffhanger"],
                narrative["standalone_understandability"], narrative["narrative_coherence"],
                narrative["emotional_engagement"],
            )
            narrative_average = round(mean(item["score"] for item in criteria), 2)
            explanation_parts.append(narrative.get("overall_explanation", ""))
            narrative_pairs.append((float(candidate.get("raw_score", 0)), narrative_average))
        elif narrative_error:
            explanation_parts.append("Independent narrative evaluation unavailable.")
        elif candidate.get("selection_status") == "selected":
            skip_message = "Not independently evaluated because the configured narrative evaluation limit was reached."
            explanation_parts.append(skip_message)

        for warning in technical["technical_warnings"]:
            if warning["severity"] != "low":
                findings.append(_normalize_failure_finding(FailureFinding(
                    category=warning["category"],
                    severity=warning["severity"],
                    explanation=warning["message"],
                    evidence=[],
                    suggested_improvement="Review source timestamps and candidate metadata.",
                ).model_dump()))

        if narrative_average is not None and narrative_average < active.quality.narrative_warning_threshold:
            explanation_parts.append(f"Narrative average {narrative_average:.2f} is below warning threshold.")
        quality_decision = "FAIL" if technical["technical_status"] == "FAIL" else (
            "PASS_WITH_WARNINGS"
            if technical["technical_warnings"] or narrative_status in {"unavailable", "skipped"} or (narrative_average is not None and narrative_average < active.quality.narrative_warning_threshold) or findings
            else "PASS"
        )
        high_rank_low_quality = (
            int(candidate.get("rank", 999999)) <= active.quality.high_rank_threshold
            and narrative_average is not None
            and narrative_average <= active.quality.low_quality_threshold
        )
        if high_rank_low_quality:
            findings.append(FailureFinding(
                category="high_rank_low_quality",
                severity="high",
                explanation="This promo ranked highly but received a low independent narrative-quality score.",
                evidence=[],
                suggested_improvement="Review ranking weights and candidate-generation signals against the independent evaluation.",
            ).model_dump())
            quality_decision = "PASS_WITH_WARNINGS" if quality_decision != "FAIL" else quality_decision

        explanation = " ".join(part for part in explanation_parts if part) or (
            "Technical validation passed; no independent narrative call was required for this unselected promo."
            if candidate.get("selection_status") != "selected"
            else "Technical validation completed."
        )
        records.append({
            **candidate,
            "technical_status": technical["technical_status"],
            "technical_errors": technical["technical_errors"],
            "technical_warnings": technical["technical_warnings"],
            "narrative_evaluation": narrative,
            "narrative_evaluation_status": narrative_status,
            "narrative_evaluation_message": narrative_error,
            "narrative_average_score": narrative_average,
            "failure_labels": findings,
            "quality_decision": quality_decision,
            "high_rank_low_quality": high_rank_low_quality,
            "explanation": explanation,
        })

    failure_counts: Counter[str] = Counter()
    for record in records:
        for finding in record["failure_labels"]:
            failure_counts[str(finding.get("category", "unknown"))] += 1
    high_low = [
        {"promo_id": item.get("promo_id"), "rank": item.get("rank"), "raw_score": item.get("raw_score"), "narrative_average_score": item.get("narrative_average_score"), "explanation": item.get("explanation")}
        for item in records if item.get("high_rank_low_quality")
    ]
    weak_scores = [item for item in records if item.get("narrative_average_score") is not None and item["narrative_average_score"] < active.quality.narrative_warning_threshold]
    report = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "candidates_evaluated": len(records),
        "narrative_candidates_evaluated": sum(item.get("narrative_evaluation") is not None for item in records),
        "passing_validation": sum(item["technical_status"] == "PASS" for item in records),
        "passing_with_warnings": sum(item["technical_status"] == "PASS_WITH_WARNINGS" for item in records),
        "failing": sum(item["technical_status"] == "FAIL" for item in records),
        "quality_decision_counts": {
            decision: sum(item["quality_decision"] == decision for item in records)
            for decision in ("PASS", "PASS_WITH_WARNINGS", "FAIL")
        },
        "technical_validation_results": [
            {"promo_id": item["promo_id"], "status": item["technical_status"], "errors": item["technical_errors"], "warnings": item["technical_warnings"]}
            for item in records
        ],
        "candidate_results": records,
        "narrative_evaluation_warnings": [
            {
                "promo_id": item["promo_id"],
                "status": item["narrative_evaluation_status"],
                "message": item.get("narrative_evaluation_message") or item.get("explanation"),
            }
            for item in records
            if item["narrative_evaluation_status"] in {"unavailable", "skipped"}
        ],
        "high_rank_low_quality": high_low,
        "pipeline_observations": {
            "failure_category_counts": dict(failure_counts),
            "narrative_evaluation_unavailable_count": sum(item["narrative_evaluation_status"] == "unavailable" for item in records),
            "narrative_evaluation_skipped_count": sum(item["narrative_evaluation_status"] == "skipped" for item in records),
            "low_narrative_quality_count": len(weak_scores),
            "ranking_narrative_score_correlation": _pearson(narrative_pairs),
            "correlation_candidate_count": len(narrative_pairs),
            "note": "Correlation is null when fewer than two narrative evaluations are available; no quality statistics are inferred for absent candidates.",
        },
    }
    examples = [item["promo_id"] for item in high_low]
    if not records:
        suggested = ["No candidates were present; run Module 3 before interpreting quality results."]
    elif failure_counts:
        suggested = [f"Review failures in category '{category}' ({count} occurrence(s))." for category, count in failure_counts.most_common()]
    else:
        suggested = ["No observed failure categories; continue reviewing generated promos against source content."]
    failure_analysis = {
        "candidates_processed": len(records),
        "narrative_evaluation_availability": {
            "evaluated": sum(item["narrative_evaluation_status"] == "evaluated" for item in records),
            "unavailable": sum(item["narrative_evaluation_status"] == "unavailable" for item in records),
            "skipped": sum(item["narrative_evaluation_status"] == "skipped" for item in records),
        },
        "most_frequent_failure_categories": [
            {"category": category, "count": count}
            for category, count in failure_counts.most_common()
        ],
        "common_weaknesses": [
            {"category": category, "count": count}
            for category, count in failure_counts.most_common()
        ],
        "ranking_quality_alignment": {
            "pearson_correlation": _pearson(narrative_pairs),
            "evaluated_candidate_count": len(narrative_pairs),
        },
        "high_ranked_poor_quality_examples": examples,
        "suggested_pipeline_improvements": suggested,
    }
    return report, failure_analysis, records


def write_quality_outputs(
    report: dict[str, Any],
    failure_analysis: dict[str, Any],
    records: list[dict[str, Any]],
    json_path: str | Path = "outputs/quality_evaluation_and_validation/quality_report.json",
    csv_path: str | Path = "outputs/quality_evaluation_and_validation/quality_report.csv",
    failure_path: str | Path = "outputs/quality_evaluation_and_validation/failure_analysis.json",
) -> None:
    for filename, payload in ((json_path, report), (failure_path, failure_analysis)):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with csv_file.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUALITY_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in records:
            segment = (item.get("segments") or [{}])[0]
            narrative = item.get("narrative_evaluation") or {}
            writer.writerow({
                "Promo ID": item.get("promo_id"),
                "Episode ID": segment.get("episode_id"),
                "Original rank": item.get("rank"),
                "Final score": item.get("raw_score"),
                "Technical validation status": item.get("technical_status"),
                "Narrative quality score": item.get("narrative_average_score"),
                "Hook score": (narrative.get("opening_hook") or {}).get("score"),
                "Cliffhanger score": (narrative.get("ending_cliffhanger") or {}).get("score"),
                "Standalone score": (narrative.get("standalone_understandability") or {}).get("score"),
                "Failure labels": "; ".join(sorted({str(label.get("category")) for label in item.get("failure_labels", [])})),
                "Quality decision": item.get("quality_decision"),
                "Short explanation": item.get("explanation"),
            })


__all__ = ["evaluate_ranked_promos", "write_quality_outputs", "QUALITY_CSV_FIELDS"]
