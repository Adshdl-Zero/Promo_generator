"""Module 6: source-grounded multi-segment promo discovery and evaluation."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.config import get_settings
from src.llm.factory import create_provider
from src.generation.multi_segment_prompts import MULTI_SEGMENT_EVALUATION_TEMPLATE
from src.schemas.multi_segment import MultiSegmentCandidate, MultiSegmentNarrativeEvaluation
from src.utils.logging_config import get_logger
from src.utils.retry import RequestPacer, is_rate_limit_error, retry_delay_seconds
from src.utils.timestamp import parse_timestamp, seconds_to_timestamp

logger = get_logger(__name__)
PROMPT_VERSION = "multi-segment-evaluation-v1"


def _score(value: Any, default: float = 5.0) -> float:
    try:
        return max(1.0, min(10.0, float(value)))
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(_clean_text(item) for item in value if _clean_text(item))
    return ""


def _events_for_scene(scene: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for kind, field in (("DIALOGUE", "dialogue"), ("ACTION", "actions")):
        for item in scene.get(field, []):
            start = parse_timestamp(item.get("start_time"))
            end = parse_timestamp(item.get("end_time"))
            text = str(item.get("text", "")).strip()
            if start is None or end is None or end <= start or not text:
                continue
            events.append({
                "start": float(start),
                "end": float(end),
                "kind": kind,
                "character": str(item.get("character", "") or "").strip(),
                "text": text,
            })
    return sorted(events, key=lambda event: (event["start"], event["end"], event["kind"]))


def _tokens(text: str) -> set[str]:
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "is", "was", "are", "with", "for", "her", "his", "she", "he", "they", "their", "that", "this", "from", "after", "before"}
    import re
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2 and token not in stop}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _scene_analysis_map(scene_analysis: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(item.get("episode_id", 0)), int(item.get("scene_id", 0))): item
        for item in scene_analysis
        if isinstance(item, dict)
    }


def _candidate_scores_by_scene(candidates: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    result: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        evaluation = candidate.get("evaluation") or {}
        for segment in candidate.get("segments", []):
            episode_id = int(segment.get("episode_id", 0))
            for scene_id in segment.get("scene_ids", []):
                result[(episode_id, int(scene_id))].append(evaluation)
    return result


def _moment_score(analysis: dict[str, Any], candidate_evaluations: list[dict[str, Any]]) -> dict[str, float]:
    if candidate_evaluations:
        def avg(field: str, fallback: float) -> float:
            vals = [_score(item.get(field), fallback) for item in candidate_evaluations]
            return sum(vals) / len(vals)
        return {
            "hook": avg("hook_score", _score(analysis.get("hook_potential"))),
            "cliffhanger": avg("cliffhanger_score", _score(analysis.get("cliffhanger_potential"))),
            "standalone": avg("standalone_context_score", _score(analysis.get("standalone_potential"))),
            "emotional": avg("emotional_intensity_score", _score(analysis.get("emotional_intensity"))),
            "coherence": avg("narrative_coherence_score", 6.0),
        }
    return {
        "hook": _score(analysis.get("hook_potential")),
        "cliffhanger": _score(analysis.get("cliffhanger_potential")),
        "standalone": _score(analysis.get("standalone_potential")),
        "emotional": _score(analysis.get("emotional_intensity")),
        "coherence": _score(analysis.get("suspense_level")),
    }


def _make_moment(
    episode: dict[str, Any],
    scene: dict[str, Any],
    start: float,
    end: float,
    events: list[dict[str, Any]],
    analysis: dict[str, Any],
    candidate_evaluations: list[dict[str, Any]],
    source_promo_id: str | None = None,
) -> dict[str, Any]:
    scene_id = int(scene.get("scene_id", 0))
    episode_id = int(episode.get("episode_id", 0))
    event_text = "\n".join(
        f"{seconds_to_timestamp(item['start'])} {item['kind']} {item['character']}: {item['text']}"
        for item in events
    )
    return {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "scene_ids": [scene_id],
        "start_seconds": round(float(start), 3),
        "end_seconds": round(float(end), 3),
        "start_time": seconds_to_timestamp(start),
        "end_time": seconds_to_timestamp(end),
        "segment_duration": round(float(end - start), 3),
        "event_type": str(analysis.get("event_type", "unknown")),
        "summary": str(analysis.get("dramatic_summary", "")),
        "characters": {
            str(item.get("character", "")).strip().lower()
            for item in events
            if item.get("kind") == "DIALOGUE" and str(item.get("character", "")).strip()
        },
        "text": event_text,
        "scores": _moment_score(analysis, candidate_evaluations),
        "source_promo_id": source_promo_id,
        "source_key": f"{episode_id}:{scene_id}:{start:.3f}:{end:.3f}",
    }


def discover_moments(
    episodes: list[dict[str, Any]],
    scene_analysis: list[dict[str, Any]],
    continuous_candidates: list[dict[str, Any]],
    settings: Any | None = None,
) -> list[dict[str, Any]]:
    """Use Module 2 scores, Module 3 windows, and exact source event boundaries."""
    active = settings or get_settings()
    cfg = active.multi_segment
    analyses = _scene_analysis_map(scene_analysis)
    candidate_scores = _candidate_scores_by_scene(continuous_candidates)
    found: dict[str, dict[str, Any]] = {}

    for episode in episodes:
        episode_id = int(episode.get("episode_id", 0))
        for scene in episode.get("scenes", []):
            scene_id = int(scene.get("scene_id", 0))
            analysis = analyses.get((episode_id, scene_id))
            if not analysis:
                continue
            events = _events_for_scene(scene)
            if len(events) < 2:
                continue
            evaluations = candidate_scores.get((episode_id, scene_id), [])

            # Derive short moments from complete source event boundaries only.
            for start_index, first in enumerate(events[:-1]):
                for end_index in range(start_index + 1, len(events)):
                    last = events[end_index]
                    duration = last["end"] - first["start"]
                    if duration > cfg.maximum_segment_duration:
                        break
                    if duration < cfg.minimum_segment_duration:
                        continue
                    selected_events = events[start_index:end_index + 1]
                    moment = _make_moment(
                        episode, scene, first["start"], last["end"], selected_events,
                        analysis, evaluations,
                    )
                    found.setdefault(moment["source_key"], moment)

            # Preserve compact existing Module 3 windows as candidate moments.
            for candidate in continuous_candidates:
                for segment in candidate.get("segments", []):
                    if int(segment.get("episode_id", -1)) != episode_id or scene_id not in {int(x) for x in segment.get("scene_ids", [])}:
                        continue
                    start = parse_timestamp(segment.get("start_time"))
                    end = parse_timestamp(segment.get("end_time"))
                    if start is None or end is None or not cfg.minimum_segment_duration <= end - start <= cfg.maximum_segment_duration:
                        continue
                    inside = [event for event in events if event["start"] >= start and event["end"] <= end]
                    if len(inside) < 2:
                        continue
                    moment = _make_moment(episode, scene, start, end, inside, analysis, evaluations, str(candidate.get("promo_id", "")))
                    found.setdefault(moment["source_key"], moment)

    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for moment in found.values():
        by_episode[moment["episode_id"]].append(moment)

    # Keep high-potential moments while diversifying the source time windows.
    pruned: list[dict[str, Any]] = []
    for episode_id, moments in by_episode.items():
        moments.sort(key=lambda item: (-_moment_priority(item), item["start_seconds"], item["end_seconds"], item["source_key"]))
        kept: list[dict[str, Any]] = []
        for moment in moments:
            if any(_time_overlap_ratio(moment, previous) >= 0.7 for previous in kept):
                continue
            kept.append(moment)
            if len(kept) >= cfg.max_moments_per_episode:
                break
        pruned.extend(kept)
    return sorted(pruned, key=lambda item: (item["episode_id"], item["start_seconds"], item["source_key"]))


def _moment_priority(moment: dict[str, Any]) -> float:
    scores = moment["scores"]
    return 0.30 * scores["hook"] + 0.25 * scores["cliffhanger"] + 0.20 * scores["emotional"] + 0.15 * scores["standalone"] + 0.10 * scores["coherence"]


def _time_overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left["episode_id"] != right["episode_id"]:
        return 0.0
    intersection = max(0.0, min(left["end_seconds"], right["end_seconds"]) - max(left["start_seconds"], right["start_seconds"]))
    shorter = min(left["segment_duration"], right["segment_duration"])
    return intersection / shorter if shorter > 0 else 0.0


def _relationship(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, list[str], float]:
    shared_characters = sorted(left["characters"] & right["characters"])
    event_match = _tokens(left["event_type"]) & _tokens(right["event_type"])
    summary_similarity = _jaccard(_tokens(left["summary"]), _tokens(right["summary"]))
    text_similarity = _jaccard(_tokens(left["text"]), _tokens(right["text"]))
    links: list[str] = []
    if shared_characters:
        links.append("shared_characters: " + ", ".join(shared_characters))
    if event_match:
        links.append("related_event_type: " + ", ".join(sorted(event_match)))
    if summary_similarity >= 0.08:
        links.append(f"related_scene_summaries: {summary_similarity:.2f} token overlap")
    if text_similarity >= 0.12:
        links.append(f"related_source_text: {text_similarity:.2f} token overlap")
    return bool(links), links, max(summary_similarity, text_similarity, 1.0 if shared_characters else 0.0, 0.5 if event_match else 0.0)


def _segments_non_overlapping(moments: list[dict[str, Any]]) -> bool:
    for left, right in itertools.combinations(moments, 2):
        if left["episode_id"] == right["episode_id"] and _time_overlap_ratio(left, right) > 0:
            return False
    return True


def _sequence(moments: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    chosen = list(moments)
    if len(chosen) <= 1:
        return chosen
    hook = max(chosen, key=lambda item: (item["scores"]["hook"], -item["start_seconds"], item["source_key"]))
    chosen.remove(hook)
    ending = max(chosen, key=lambda item: (item["scores"]["cliffhanger"], item["end_seconds"], item["source_key"]))
    chosen.remove(ending)
    middle = sorted(chosen, key=lambda item: (-item["scores"]["emotional"], -item["scores"]["coherence"], item["source_key"]))
    return [hook, *middle, ending]


def _combo_signature(sequence: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted(item["source_key"] for item in sequence))


def _connected(sequence: list[dict[str, Any]]) -> tuple[bool, list[str], float]:
    relationships: list[str] = []
    values: list[float] = []
    adjacency = {i: set() for i in range(len(sequence))}
    for i, j in itertools.combinations(range(len(sequence)), 2):
        related, links, strength = _relationship(sequence[i], sequence[j])
        if related:
            adjacency[i].add(j)
            adjacency[j].add(i)
            relationships.extend(links)
            values.append(strength)
    reached = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    return len(reached) == len(sequence), sorted(set(relationships)), (sum(values) / len(values) if values else 0.0)


def _cache_key(candidate: dict[str, Any], provider: Any) -> str:
    payload = {
        "candidate": candidate,
        "prompt_version": PROMPT_VERSION,
        "provider": provider.provider_name,
        "model": getattr(provider, "model", "unknown"),
        "temperature": getattr(provider, "temperature", None),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _read_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not load Module 6 cache at %s", path)
        return {}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _source_for_moment(moment: dict[str, Any]) -> str:
    return moment["text"] or "No dialogue/action text exists inside this source-timestamped segment."


def _evaluate_candidate(
    candidate: dict[str, Any],
    moment_map: dict[str, dict[str, Any]],
    provider: Any,
    cache_path: Path,
    use_cache: bool,
    retries: int,
    backoff_seconds: float,
    max_api_calls: int,
    pacer: RequestPacer,
) -> tuple[dict[str, Any] | None, str | None, bool, int]:
    moments = [moment_map[key] for key in candidate["_moment_keys"]]
    segments_text = "\n\n".join(
        f"SEGMENT {index + 1} (fixed order): episode {moment['episode_id']}, scene {moment['scene_id']}, "
        f"{moment['start_time']}–{moment['end_time']} ({moment['segment_duration']:.3f}s)\n"
        f"Scene context: {moment['summary']}\nSource screenplay:\n{_source_for_moment(moment)}"
        for index, moment in enumerate(moments)
    )
    prompt = MULTI_SEGMENT_EVALUATION_TEMPLATE.format(segments_text=segments_text)
    key = _cache_key({"candidate": {k: v for k, v in candidate.items() if not k.startswith("_")}, "prompt": prompt}, provider)
    cache = _read_cache(cache_path) if use_cache else {}
    cached = cache.get(key)
    if cached:
        try:
            return MultiSegmentNarrativeEvaluation.model_validate(cached).model_dump(), None, True, 0
        except ValidationError:
            pass

    api_calls = 0
    attempts = min(max(1, retries), max_api_calls)
    for attempt in range(attempts):
        try:
            pacer.wait()
            api_calls += 1
            response = provider.generate_json(prompt, response_schema=MultiSegmentNarrativeEvaluation.model_json_schema())
            payload = provider.parse_json_response(response)
            evaluation = MultiSegmentNarrativeEvaluation.model_validate(payload).model_dump()
            if use_cache:
                cache[key] = evaluation
                _write_json(cache_path, cache)
            return evaluation, None, False, api_calls
        except Exception as exc:
            message = str(exc)
            if is_rate_limit_error(exc) or "invalid api key" in message.lower():
                return None, message, False, api_calls
            if attempt + 1 >= attempts:
                return None, message, False, api_calls
            delay = retry_delay_seconds(exc, attempt, backoff_seconds)
            logger.warning("Multi-segment evaluation retry in %.1fs", delay)
            time.sleep(delay)
    return None, "No valid structured evaluation received.", False, api_calls


def _make_combination(
    sequence: list[dict[str, Any]],
    relationships: list[str],
    relationship_score: float,
    index: int,
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    for order, moment in enumerate(sequence, start=1):
        segments.append({
            "episode_id": moment["episode_id"],
            "scene_ids": moment["scene_ids"],
            "start_time": moment["start_time"],
            "end_time": moment["end_time"],
            "start_seconds": moment["start_seconds"],
            "end_seconds": moment["end_seconds"],
            "segment_duration": moment["segment_duration"],
            "segment_order": order,
        })
    strategy = "same_episode" if len({item["episode_id"] for item in sequence}) == 1 else (
        "cross_episode" if len({item["episode_id"] for item in sequence}) == len(sequence) else "mixed"
    )
    sequence_score = (
        0.25 * sequence[0]["scores"]["hook"]
        + 0.20 * sequence[-1]["scores"]["cliffhanger"]
        + 0.20 * sum(item["scores"]["emotional"] for item in sequence) / len(sequence)
        + 0.15 * sum(item["scores"]["standalone"] for item in sequence) / len(sequence)
        + 0.10 * sum(item["scores"]["coherence"] for item in sequence) / len(sequence)
        + 0.10 * relationship_score * 10
    )
    first_event = next((line for line in sequence[0]["text"].splitlines() if line), "")
    last_event = next((line for line in reversed(sequence[-1]["text"].splitlines()) if line), "")
    summary = " → ".join(item["summary"] for item in sequence if item["summary"])
    ids = list(dict.fromkeys(item["episode_id"] for item in sequence))
    return {
        "promo_id": f"MS{index:04d}",
        "promo_type": "multi_segment",
        "generation_method": "multi_segment",
        "generation_strategy": strategy,
        "segment_count": len(segments),
        "episode_ids": ids,
        "segments": segments,
        "total_duration": round(sum(item["segment_duration"] for item in sequence), 3),
        "hook": {"description": "Opening source moment: " + first_event[:180], "evidence": first_event, "timestamp": segments[0]["start_time"]},
        "cliffhanger": {"description": "Final source moment: " + last_event[:180], "evidence": last_event, "timestamp": segments[-1]["end_time"]},
        "combined_summary": summary,
        "overall_summary": summary,
        "character_connections": sorted(set.intersection(*(item["characters"] for item in sequence))) if sequence else [],
        "event_relationships": relationships,
        "generation_explanation": f"Python selected {len(segments)} source-grounded moments in hook-to-cliffhanger order; relationship evidence: {'; '.join(relationships)}.",
        "generation_reason_score": round(sequence_score, 3),
        "evaluation_status": "pending",
        "evaluation": {},
        "strengths": [],
        "weaknesses": [],
        "_moment_keys": [item["source_key"] for item in sequence],
    }


def generate_multi_segment_promos(
    episodes: list[dict[str, Any]],
    scene_analysis: list[dict[str, Any]],
    continuous_candidates: list[dict[str, Any]],
    ranked_promos: list[dict[str, Any]] | None = None,
    *,
    settings: Any | None = None,
    provider: Any | None = None,
    cache_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    use_cache: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discover, prune, construct, evaluate, validate, and rank joined promo sequences."""
    active = settings or get_settings()
    cfg = active.multi_segment
    if not cfg.enabled:
        return [], {"generated": 0, "reason": "multi-segment generation disabled in configuration"}
    llm_provider = provider or create_provider()
    moments = discover_moments(episodes, scene_analysis, continuous_candidates, active)
    moment_map = {item["source_key"]: item for item in moments}
    stats: Counter[str] = Counter(moments=len(moments))
    if len(moments) < cfg.min_segments:
        return [], {**dict(stats), "generated": 0, "reason": "not enough source-grounded moments"}

    proposals: list[tuple[float, dict[str, Any]]] = []
    seen_signatures: set[tuple[str, ...]] = set()
    explored = 0
    per_episode_selected: Counter[int] = Counter()
    for size in range(cfg.min_segments, min(cfg.max_segments, len(moments)) + 1):
        for group in itertools.combinations(moments, size):
            explored += 1
            if explored > cfg.max_combinations * 20:
                break
            episode_ids = {item["episode_id"] for item in group}
            same = len(episode_ids) == 1
            if same and not cfg.same_episode_enabled:
                continue
            if len(episode_ids) > 1 and not cfg.cross_episode_enabled:
                continue
            if not _segments_non_overlapping(list(group)):
                continue
            total = sum(item["segment_duration"] for item in group)
            if not cfg.minimum_duration <= total <= cfg.maximum_duration:
                continue
            sequence = _sequence(group)
            connected, relations, relationship_score = _connected(sequence)
            if not connected:
                continue
            signature = _combo_signature(sequence)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            proposal = _make_combination(sequence, relations, relationship_score, len(proposals) + 1)
            proposals.append((proposal["generation_reason_score"], proposal))
        if explored > cfg.max_combinations * 20:
            break

    # Prefer strong anchor scores and relationship evidence; cap work before API evaluation.
    proposals.sort(key=lambda item: (-item[0], item[1]["generation_strategy"], tuple(item[1]["_moment_keys"])))
    selected: list[dict[str, Any]] = []
    per_episode_selected.clear()
    for _, proposal in proposals:
        if len(selected) >= cfg.max_combinations:
            break
        episode_ids = proposal["episode_ids"]
        if any(per_episode_selected[episode_id] >= cfg.max_candidates_per_episode for episode_id in episode_ids):
            continue
        signature = set(proposal["_moment_keys"])
        if any(len(signature & set(item["_moment_keys"])) / len(signature | set(item["_moment_keys"])) >= 0.8 for item in selected):
            stats["near_duplicate_pruned"] += 1
            continue
        selected.append(proposal)
        for episode_id in episode_ids:
            per_episode_selected[episode_id] += 1

    evaluated: list[dict[str, Any]] = []
    cache_file = Path(cache_path or cfg.cache_path)
    checkpoint_file = Path(checkpoint_path or cfg.checkpoint_path)
    api_calls = 0
    cache_hits = 0
    errors: list[dict[str, str]] = []
    rate_limited = False
    pacer = RequestPacer(cfg.request_interval_seconds)
    for index, candidate in enumerate(selected, start=1):
        if rate_limited:
            candidate["evaluation_status"] = "not_evaluated_rate_limit"
            stats["rate_limit_skips"] += 1
            continue
        if api_calls >= cfg.max_llm_calls:
            candidate["evaluation_status"] = "not_evaluated_limit"
            stats["evaluation_limit_skips"] += 1
            continue
        logger.info("Evaluating multi-segment combination %d/%d", index, len(selected))
        result, error, cached, calls_made = _evaluate_candidate(
            candidate,
            moment_map,
            llm_provider,
            cache_file,
            use_cache,
            active.llm.max_retries,
            active.llm.retry_backoff_seconds,
            cfg.max_llm_calls - api_calls,
            pacer,
        )
        if cached:
            cache_hits += 1
        api_calls += calls_made
        if error or result is None:
            candidate["evaluation_status"] = "unavailable"
            candidate["evaluation_error"] = error or "No structured evaluation returned."
            errors.append({"promo_id": candidate["promo_id"], "error": candidate["evaluation_error"]})
            if is_rate_limit_error(candidate["evaluation_error"]):
                rate_limited = True
            continue

        candidate["multi_segment_evaluation"] = result
        candidate["evaluation"] = {
            "hook_score": result["hook_strength"],
            "cliffhanger_score": result["cliffhanger_strength"],
            "standalone_context_score": result["standalone_understandability"],
            "emotional_intensity_score": result["emotional_impact"],
            "narrative_coherence_score": result["narrative_coherence"],
            "promo_engagement_potential": result["overall_promotional_potential"],
        }
        candidate["narrative_coherence_score"] = result["narrative_coherence"]
        candidate["standalone_understandability_score"] = result["standalone_understandability"]
        candidate["overall_promotional_score"] = result["overall_promotional_potential"]
        candidate["strengths"] = result["strengths"]
        candidate["weaknesses"] = result["weaknesses"]
        candidate["combined_summary"] = result["explanation"]
        candidate["overall_summary"] = result["explanation"]
        candidate["evaluation_status"] = "evaluated"
        try:
            validated = MultiSegmentCandidate.model_validate({
                **candidate,
                "validation_status": "valid",
                "segment_count": len(candidate["segments"]),
            }).model_dump(mode="json")
        except ValidationError as exc:
            candidate["evaluation_status"] = "invalid"
            candidate["validation_status"] = "invalid"
            candidate["validation_errors"] = [str(error) for error in exc.errors(include_url=False)]
            errors.append({"promo_id": candidate["promo_id"], "error": str(exc)})
            continue
        validated["validation_status"] = "valid"
        validated["evaluation_status"] = "evaluated"
        validated["generation_reason_score"] = candidate["generation_reason_score"]
        if result["overall_promotional_potential"] < cfg.minimum_quality_score:
            stats["below_quality_threshold"] += 1
            continue
        evaluated.append(validated)
        # Save a recoverable checkpoint after each accepted, validated candidate.
        _write_json(checkpoint_file, evaluated)

    for item in evaluated:
        item.pop("_moment_keys", None)
    metadata = dict(stats)
    metadata.update({
        "moments_discovered": len(moments),
        "combinations_explored": explored,
        "combinations_prepared": len(selected),
        "generated": len(evaluated),
        "llm_requests": api_calls,
        "cache_hits": cache_hits,
        "provider": llm_provider.provider_name,
        "model": getattr(llm_provider, "model", "unknown"),
    })
    if errors:
        metadata["errors"] = errors
    return evaluated, metadata


__all__ = ["discover_moments", "generate_multi_segment_promos"]
