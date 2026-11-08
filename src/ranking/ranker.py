"""Module 4: weighted promo ranking and diversity-aware selection."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.config import RankingSettings, get_settings
from src.llm.factory import create_provider
from src.ranking.prompts import MISSING_SCORE_TEMPLATE
from src.schemas.ranking import MissingScoreAssessment
from src.utils.logging_config import get_logger
from src.utils.retry import RequestPacer, is_rate_limit_error, retry_delay_seconds
from src.utils.timestamp import parse_timestamp

logger = get_logger(__name__)
PROMPT_VERSION = "ranking-missing-scores-v1"
SCORE_FIELDS = {
    "hook_strength": "hook_score",
    "cliffhanger_strength": "cliffhanger_score",
    "standalone_understandability": "standalone_context_score",
    "emotional_intensity": "emotional_intensity_score",
    "narrative_coherence": "narrative_coherence_score",
    "promo_engagement_potential": "promo_engagement_potential",
}


def _read_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read ranking cache at %s", path)
        return {}


def _write_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_key(candidate: dict[str, Any], provider: Any, config: dict[str, Any]) -> str:
    payload = {
        "candidate": candidate,
        "prompt_version": PROMPT_VERSION,
        "provider": provider.provider_name,
        "model": getattr(provider, "model", "unknown"),
        "temperature": getattr(provider, "temperature", None),
        "config": config,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _source_text(episode: dict[str, Any], candidate: dict[str, Any], limit: int = 3000) -> str:
    lines: list[str] = []
    segments = candidate.get("segments") or []
    for segment in segments:
        scene_ids = {int(scene_id) for scene_id in segment.get("scene_ids", [])}
        start = parse_timestamp(segment.get("start_time"))
        end = parse_timestamp(segment.get("end_time"))
        for scene in episode.get("scenes", []):
            if int(scene.get("scene_id", 0)) not in scene_ids:
                continue
            for kind, events in (("DIALOGUE", scene.get("dialogue", [])), ("ACTION", scene.get("actions", []))):
                for event in events:
                    event_start = parse_timestamp(event.get("start_time"))
                    event_end = parse_timestamp(event.get("end_time"))
                    if start is not None and end is not None and event_start is not None:
                        if event_start > end or (event_end if event_end is not None else event_start) < start:
                            continue
                    timestamp = event.get("start_time", "?")
                    character = event.get("character", "")
                    text = event.get("text", "")
                    if text:
                        lines.append(f"{timestamp} {kind} {character}: {text}")
    content = "\n".join(lines)
    if len(content) <= limit:
        return content or "No source dialogue/actions were found in this candidate window."
    marker = "\n...[middle omitted]...\n"
    usable = limit - len(marker)
    head = usable * 2 // 3
    return content[:head] + marker + content[-(usable - head):]


def _candidate_key(candidate: dict[str, Any], position: int) -> str:
    value = candidate.get("promo_id")
    return str(value) if value not in (None, "") else f"PROMO-{position:04d}"


def _criterion_evidence(candidate: dict[str, Any], criterion: str) -> str:
    hook = candidate.get("hook") or {}
    cliffhanger = candidate.get("cliffhanger") or {}
    evaluation = candidate.get("evaluation") or {}
    if criterion == "hook_strength":
        return str(hook.get("evidence") or candidate.get("overall_summary") or "Existing Module 3 hook score.")
    if criterion == "cliffhanger_strength":
        return str(cliffhanger.get("evidence") or candidate.get("overall_summary") or "Existing Module 3 cliffhanger score.")
    return str(candidate.get("overall_summary") or "; ".join(candidate.get("strengths", [])) or f"Existing Module 3 {evaluation.get(SCORE_FIELDS[criterion], 'evaluation')} score.")


def _get_scores(
    candidate: dict[str, Any],
    episode: dict[str, Any],
    provider: Any,
    ranking: RankingSettings,
    cache_path: Path,
    retries: int,
    backoff_seconds: float,
    max_api_calls: int,
    pacer: RequestPacer,
) -> tuple[dict[str, int | None], dict[str, str], list[str], int, bool]:
    evaluation = candidate.get("evaluation") or {}
    scores: dict[str, int | None] = {}
    evidence: dict[str, str] = {}
    missing: list[str] = []
    for criterion, field in SCORE_FIELDS.items():
        value = evaluation.get(field)
        try:
            numeric = int(value) if value is not None else None
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None and 1 <= numeric <= 10:
            scores[criterion] = numeric
            evidence[criterion] = _criterion_evidence(candidate, criterion)
        else:
            scores[criterion] = None
            missing.append(criterion)

    warnings: list[str] = []
    if not missing:
        return scores, evidence, warnings, 0, False

    segment = (candidate.get("segments") or [{}])[0]
    prompt_context = {
        "ranking_weights": ranking.weights,
        "missing_criteria": missing,
    }
    key = _cache_key({"candidate": candidate, "source": _source_text(episode, candidate), "missing": missing}, provider, prompt_context)
    cache = _read_cache(cache_path)
    cached = cache.get(key)
    assessment = None
    if cached:
        try:
            assessment = MissingScoreAssessment.model_validate(cached)
        except Exception:
            assessment = None

    api_calls = 0
    rate_limited = False
    if assessment is None:
        if max_api_calls <= 0:
            warnings.append("Missing score assessment skipped to honor the configured Module 4 API-call limit.")
            for criterion in missing:
                evidence[criterion] = "Score unavailable; no value was fabricated."
            return scores, evidence, warnings, api_calls, rate_limited
        prompt = MISSING_SCORE_TEMPLATE.format(
            missing_criteria=", ".join(missing),
            episode_id=segment.get("episode_id", episode.get("episode_id")),
            scene_ids=segment.get("scene_ids", []),
            start_time=segment.get("start_time", "unknown"),
            end_time=segment.get("end_time", "unknown"),
            duration_seconds=candidate.get("total_duration", "unknown"),
            summary=str(candidate.get("overall_summary", ""))[:300],
            source_text=_source_text(episode, candidate),
        )
        attempts = min(max(1, retries), max_api_calls)
        for attempt in range(attempts):
            try:
                pacer.wait()
                api_calls += 1
                response = provider.generate_json(prompt, response_schema=MissingScoreAssessment.model_json_schema())
                assessment = MissingScoreAssessment.model_validate(provider.parse_json_response(response))
                cache[key] = assessment.model_dump()
                _write_cache(cache_path, cache)
                break
            except Exception as exc:
                if is_rate_limit_error(exc):
                    rate_limited = True
                    warning = f"Could not obtain missing criterion scores because the provider rate limit was reached: {exc}"
                    warnings.append(warning)
                    logger.warning("%s; skipping further Module 4 score assessments this run.", warning)
                    break
                if attempt + 1 >= attempts:
                    warning = f"Could not obtain missing criterion scores: {exc}"
                    warnings.append(warning)
                    logger.warning("%s", warning)
                else:
                    delay = retry_delay_seconds(exc, attempt, backoff_seconds)
                    logger.warning("Ranking score request failed; retrying in %.1fs", delay)
                    time.sleep(delay)

    if assessment is not None:
        values = assessment.model_dump()
        for criterion in missing:
            item = values[criterion]
            scores[criterion] = item["score"]
            evidence[criterion] = item["evidence"]
    else:
        for criterion in missing:
            evidence[criterion] = "Score unavailable; no value was fabricated."
    return scores, evidence, warnings, api_calls, rate_limited


def weighted_score(scores: dict[str, int | None], weights: dict[str, float]) -> float:
    """Return a 0-100 weighted score, renormalizing only across available scores."""
    weighted_sum = 0.0
    available_weight = 0.0
    for criterion, weight in weights.items():
        score = scores.get(criterion)
        if score is None or weight <= 0:
            continue
        weighted_sum += float(score) * weight
        available_weight += weight
    if available_weight <= 0:
        return 0.0
    return round((weighted_sum / available_weight) * 10, 2)


def _segments(candidate: dict[str, Any]) -> list[tuple[int, float, float, set[int]]]:
    parsed: list[tuple[int, float, float, set[int]]] = []
    for segment in candidate.get("segments", []):
        start = parse_timestamp(segment.get("start_time"))
        end = parse_timestamp(segment.get("end_time"))
        if start is None or end is None:
            continue
        parsed.append((int(segment.get("episode_id", 0)), start, end, {int(value) for value in segment.get("scene_ids", [])}))
    return parsed


def _overlap(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, float]:
    scene_overlap = 0.0
    time_overlap = 0.0
    for left_episode, left_start, left_end, left_scenes in _segments(left):
        for right_episode, right_start, right_end, right_scenes in _segments(right):
            if left_episode != right_episode:
                continue
            union = left_scenes | right_scenes
            if union:
                scene_overlap = max(scene_overlap, len(left_scenes & right_scenes) / len(union))
            shorter_duration = min(left_end - left_start, right_end - right_start)
            if shorter_duration > 0:
                intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
                time_overlap = max(time_overlap, intersection / shorter_duration)
    return scene_overlap, time_overlap


def _text_similarity(left: str, right: str) -> float:
    tokens_left = set(re.findall(r"[a-z0-9]+", left.lower()))
    tokens_right = set(re.findall(r"[a-z0-9]+", right.lower()))
    union = tokens_left | tokens_right
    return len(tokens_left & tokens_right) / len(union) if union else 0.0


def _event_types(candidate: dict[str, Any], scene_analysis: list[dict[str, Any]]) -> set[str]:
    scene_keys = {
        (int(segment.get("episode_id", 0)), int(scene_id))
        for segment in candidate.get("segments", [])
        for scene_id in segment.get("scene_ids", [])
    }
    return {
        str(item.get("event_type", "")).lower()
        for item in scene_analysis
        if (int(item.get("episode_id", 0)), int(item.get("scene_id", 0))) in scene_keys and item.get("event_type")
    }


def _category(score: float, thresholds: dict[str, float]) -> str:
    for category in ("excellent", "strong", "moderate"):
        if score >= float(thresholds.get(category, 0)):
            return category.title()
    return "Weak"


def rank_candidates(
    candidates: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    scene_analysis: list[dict[str, Any]],
    *,
    settings: Any | None = None,
    provider: Any | None = None,
    cache_path: str | Path = "outputs/promo_ranking_and_selection/ranking_score_cache.json",
) -> list[dict[str, Any]]:
    """Add quality ranking and selection metadata to all candidate dictionaries."""
    active = settings or get_settings()
    ranking: RankingSettings = active.ranking
    llm_provider = provider or create_provider()
    episode_by_id = {int(item.get("episode_id", 0)): item for item in episodes}
    work: list[dict[str, Any]] = []
    cache_file = Path(cache_path)
    remaining_score_calls = ranking.max_missing_score_llm_calls
    rate_limited = False
    pacer = RequestPacer(ranking.request_interval_seconds)

    for position, source_candidate in enumerate(candidates, start=1):
        logger.info("Ranking candidate %d/%d", position, len(candidates))
        candidate = dict(source_candidate)
        candidate.setdefault("promo_id", _candidate_key(candidate, position))
        first_segment = (candidate.get("segments") or [{}])[0]
        episode_id = int(first_segment.get("episode_id", candidate.get("episode_id", 0)))
        episode_ids = {
            int(segment.get("episode_id", episode_id))
            for segment in candidate.get("segments", [])
        } or {episode_id}
        episode = episode_by_id.get(episode_id, {"episode_id": episode_id, "scenes": []})
        scores, evidence, warnings, calls_made, candidate_rate_limited = _get_scores(
            candidate,
            episode,
            llm_provider,
            ranking,
            cache_file,
            active.llm.max_retries,
            active.llm.retry_backoff_seconds,
            0 if rate_limited else remaining_score_calls,
            pacer,
        )
        remaining_score_calls = max(0, remaining_score_calls - calls_made)
        rate_limited = rate_limited or candidate_rate_limited
        raw_score = weighted_score(scores, ranking.weights)
        missing_criteria = [name for name, value in scores.items() if value is None]
        item = {
            **candidate,
            "raw_score": raw_score,
            "diversity_adjusted_score": raw_score,
            "criterion_scores": scores,
            "criterion_evidence": evidence,
            "quality_category": _category(raw_score, ranking.category_thresholds),
            "ranking_explanation": (
                f"Weighted score {raw_score:.2f}/100 using available criteria."
                + (f" Missing: {', '.join(missing_criteria)}." if missing_criteria else "")
            ),
            "selection_status": "not_selected",
            "selection_reason": "Not processed for diversity selection yet.",
            "duplicate_or_overlap_flags": [],
            "ranking_warnings": warnings,
            "_episode_id": episode_id,
            "_episode_ids": episode_ids,
            "_scene_ids": {int(scene_id) for segment in candidate.get("segments", []) for scene_id in segment.get("scene_ids", [])},
            "_event_types": _event_types(candidate, scene_analysis),
        }
        work.append(item)

    work.sort(key=lambda item: (-item["raw_score"], str(item["promo_id"])))
    for rank, item in enumerate(work, start=1):
        item["rank"] = rank
        item["diversity_adjusted_rank"] = None

    for index, item in enumerate(work):
        for other in work[index + 1:]:
            scene_ratio, time_ratio = _overlap(item, other)
            summary_similarity = _text_similarity(str(item.get("overall_summary", "")), str(other.get("overall_summary", "")))
            if item.get("_episode_id") == other.get("_episode_id") and (scene_ratio >= ranking.maximum_scene_overlap or time_ratio >= ranking.maximum_scene_overlap):
                kind = "exact_duplicate" if scene_ratio == 1 and time_ratio == 1 else "high_overlap"
                item["duplicate_or_overlap_flags"].append({"type": kind, "related_promo_id": other["promo_id"], "scene_overlap": round(scene_ratio, 3), "time_overlap": round(time_ratio, 3)})
                other["duplicate_or_overlap_flags"].append({"type": kind, "related_promo_id": item["promo_id"], "scene_overlap": round(scene_ratio, 3), "time_overlap": round(time_ratio, 3)})
            if summary_similarity >= ranking.similarity_threshold:
                item["duplicate_or_overlap_flags"].append({"type": "similar_summary", "related_promo_id": other["promo_id"], "similarity": round(summary_similarity, 3)})
                other["duplicate_or_overlap_flags"].append({"type": "similar_summary", "related_promo_id": item["promo_id"], "similarity": round(summary_similarity, 3)})

    chosen: list[dict[str, Any]] = []
    episode_counts: Counter[int] = Counter()
    remaining = list(work)
    while remaining and len(chosen) < ranking.maximum_selected_promos:
        best: dict[str, Any] | None = None
        best_adjusted = float("-inf")
        best_reason = ""
        for item in remaining:
            if item["raw_score"] < ranking.minimum_selection_score:
                continue
            episode_id = item["_episode_id"]
            if any(episode_counts[value] >= ranking.maximum_promos_per_episode for value in item["_episode_ids"]):
                continue
            penalties: list[float] = []
            reasons: list[str] = []
            for selected in chosen:
                if item["_episode_ids"] & selected["_episode_ids"]:
                    penalties.append(ranking.diversity_penalty)
                    reasons.append("same-episode diversity penalty")
                scene_ratio, time_ratio = _overlap(item, selected)
                if scene_ratio >= ranking.maximum_scene_overlap or time_ratio >= ranking.maximum_scene_overlap:
                    penalties.append(ranking.diversity_penalty * max(scene_ratio, time_ratio))
                    reasons.append("overlap diversity penalty")
                similarity = _text_similarity(str(item.get("overall_summary", "")), str(selected.get("overall_summary", "")))
                if similarity >= ranking.similarity_threshold:
                    penalties.append(ranking.diversity_penalty * similarity)
                    reasons.append("summary similarity penalty")
                if item["_event_types"] and item["_event_types"] & selected["_event_types"]:
                    penalties.append(ranking.diversity_penalty * 0.25)
                    reasons.append("repeated event type penalty")
            adjusted = max(0.0, item["raw_score"] - sum(penalties))
            if adjusted > best_adjusted:
                best, best_adjusted = item, adjusted
                best_reason = ", ".join(dict.fromkeys(reasons)) or "Highest quality score while preserving episode and story diversity."

        if best is None:
            break
        remaining.remove(best)
        best["diversity_adjusted_score"] = round(best_adjusted, 2)
        best["selection_reason"] = best_reason
        if best_adjusted < ranking.minimum_selection_score:
            best["selection_reason"] += f" Adjusted score below minimum {ranking.minimum_selection_score} after diversity penalties."
            continue
        best["selection_status"] = "selected"
        best["selection_reason"] = best_reason
        best["diversity_adjusted_rank"] = len(chosen) + 1
        chosen.append(best)
        for episode_id in best["_episode_ids"]:
            episode_counts[episode_id] += 1

    for item in remaining:
        if item["raw_score"] < ranking.minimum_selection_score:
            item["selection_reason"] = f"Raw score below minimum selection score {ranking.minimum_selection_score}."
        elif any(episode_counts[value] >= ranking.maximum_promos_per_episode for value in item["_episode_ids"]):
            item["selection_reason"] = "Episode reached maximum selected promos."
        else:
            item["selection_reason"] = "Not selected within maximum selected promo count or diversity constraints."

    for item in work:
        item.pop("_episode_id", None)
        item.pop("_episode_ids", None)
        item.pop("_scene_ids", None)
        item.pop("_event_types", None)
    return work


__all__ = ["weighted_score", "rank_candidates"]
