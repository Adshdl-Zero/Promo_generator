"""Deterministic technical validation for promo candidates and ranked outputs."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.schemas.promo_candidate import PromoCandidate
from src.schemas.quality_evaluation import ValidationIssue
from src.utils.timestamp import duration_seconds, parse_timestamp


def _issue(category: str, message: str, severity: str = "high", segment_index: int | None = None) -> dict[str, Any]:
    return ValidationIssue(category=category, message=message, severity=severity, segment_index=segment_index).model_dump()


def validate_candidate(
    candidate: dict[str, Any],
    episodes_by_id: dict[int, dict[str, Any]],
    original_candidate: dict[str, Any] | None = None,
    minimum_duration: float = 20.0,
    maximum_duration: float = 60.0,
    minimum_segment_duration: float = 1.0,
) -> dict[str, Any]:
    """Return deterministic errors/warnings; never mutate or repair input data."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    promo_id = str(candidate.get("promo_id", "<missing>"))

    validation_candidate = dict(candidate)
    legacy_evidence_fields: list[str] = []
    for evidence_key in ("hook", "cliffhanger"):
        evidence_object = candidate.get(evidence_key)
        if not isinstance(evidence_object, dict):
            continue
        evidence_value = evidence_object.get("evidence")
        if isinstance(evidence_value, list):
            if evidence_value and all(isinstance(item, str) for item in evidence_value):
                normalized_object = dict(evidence_object)
                normalized_object["evidence"] = " | ".join(evidence_value)
                validation_candidate[evidence_key] = normalized_object
                legacy_evidence_fields.append(evidence_key)
    try:
        PromoCandidate.model_validate(validation_candidate)
    except ValidationError as exc:
        for entry in exc.errors(include_url=False):
            errors.append(_issue("schema", f"{'.'.join(str(part) for part in entry['loc'])}: {entry['msg']}"))
    for evidence_key in legacy_evidence_fields:
        warnings.append(_issue(
            "legacy_evidence_list",
            f"{evidence_key}.evidence is an array of strings; accepted for backward compatibility without changing source data.",
            "low",
        ))

    if not promo_id or promo_id == "<missing>":
        errors.append(_issue("missing_promo_id", "Promo ID is missing."))

    segments = candidate.get("segments")
    if not isinstance(segments, list) or not segments:
        errors.append(_issue("missing_segments", "Promo must contain at least one segment."))
        segments = []

    is_multi_segment = candidate.get("promo_type") == "multi_segment" or candidate.get("generation_method") == "multi_segment" or len(segments) > 1

    calculated_total = 0.0
    parsed_segments: list[tuple[int, float, float, int]] = []
    previous: tuple[int, float, float] | None = None
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            errors.append(_issue("invalid_segment", "Segment must be a JSON object.", segment_index=index))
            continue
        start = parse_timestamp(segment.get("start_time"))
        end = parse_timestamp(segment.get("end_time"))
        episode_id_value = segment.get("episode_id")
        try:
            episode_id = int(episode_id_value)
        except (TypeError, ValueError):
            episode_id = -1
            errors.append(_issue("invalid_episode_reference", "Segment episode_id must be an integer.", segment_index=index))

        if start is None or end is None:
            errors.append(_issue("invalid_timestamp", "Start and end timestamps must be valid.", segment_index=index))
            continue
        if start >= end:
            errors.append(_issue("invalid_timestamp_order", "Segment start must be before its end.", segment_index=index))
            continue

        length = end - start
        calculated_total += length
        parsed_segments.append((episode_id, start, end, index))
        if is_multi_segment:
            if length < minimum_segment_duration or length > maximum_duration:
                errors.append(_issue("segment_duration", f"Segment duration {length:.2f}s is outside the allowed {minimum_segment_duration:g}-{maximum_duration:g} second range.", segment_index=index))
            declared_segment_duration = segment.get("segment_duration")
            if declared_segment_duration is not None:
                try:
                    if abs(float(declared_segment_duration) - length) > 0.05:
                        errors.append(_issue("segment_duration_mismatch", "segment_duration does not match the fixed timestamp boundaries.", segment_index=index))
                except (TypeError, ValueError):
                    errors.append(_issue("invalid_segment_duration", "segment_duration must be numeric.", segment_index=index))
            declared_order = segment.get("segment_order")
            if declared_order is not None and int(declared_order) != index + 1:
                errors.append(_issue("segment_order_mismatch", "segment_order must match the segment's array position.", segment_index=index))
        elif not minimum_duration <= length <= maximum_duration:
            errors.append(_issue("segment_duration", f"Segment duration {length:.2f}s is outside the {minimum_duration:g}-{maximum_duration:g} second range.", segment_index=index))

        episode = episodes_by_id.get(episode_id)
        if episode is None:
            errors.append(_issue("missing_episode", f"Referenced episode {episode_id} does not exist.", segment_index=index))
            continue

        source_scenes = episode.get("scenes", [])
        scene_map = {int(scene.get("scene_id", -1)): scene for scene in source_scenes}
        try:
            scene_ids = [int(value) for value in segment.get("scene_ids", [])]
        except (TypeError, ValueError):
            scene_ids = []
            errors.append(_issue("invalid_scene_reference", "scene_ids must contain integers.", segment_index=index))
        missing_scene_ids = [value for value in scene_ids if value not in scene_map]
        if missing_scene_ids:
            errors.append(_issue("missing_scene", f"Referenced scene IDs do not exist in episode {episode_id}: {missing_scene_ids}.", segment_index=index))
        if not scene_ids:
            warnings.append(_issue("scene_reference_missing", "No scene IDs were supplied for this segment.", "medium", index))

        valid_scene_ranges: list[tuple[float, float]] = []
        for scene_id in scene_ids:
            scene = scene_map.get(scene_id)
            if scene is None:
                continue
            scene_start = parse_timestamp(scene.get("start_time"))
            scene_end = parse_timestamp(scene.get("end_time"))
            if scene_start is not None and scene_end is not None:
                valid_scene_ranges.append((scene_start, scene_end))
        if scene_ids and valid_scene_ranges and not any(start < scene_end and end > scene_start for scene_start, scene_end in valid_scene_ranges):
            errors.append(_issue("timestamp_scene_mismatch", "Segment timestamps do not overlap any referenced source scene.", segment_index=index))

        episode_starts = [parse_timestamp(scene.get("start_time")) for scene in source_scenes]
        episode_ends = [parse_timestamp(scene.get("end_time")) for scene in source_scenes]
        known_starts = [value for value in episode_starts if value is not None]
        known_ends = [value for value in episode_ends if value is not None]
        if known_starts and start < min(known_starts):
            errors.append(_issue("before_episode", "Segment starts before the source episode duration.", segment_index=index))
        if known_ends and end > max(known_ends):
            errors.append(_issue("after_episode", "Segment ends after the source episode duration.", segment_index=index))

        event_times: list[float] = []
        for scene_id in scene_ids:
            scene = scene_map.get(scene_id)
            if scene is None:
                continue
            for field in ("dialogue", "actions"):
                for event in scene.get(field, []):
                    event_start = parse_timestamp(event.get("start_time"))
                    event_end = parse_timestamp(event.get("end_time"))
                    if event_start is not None and event_start <= end and (event_end if event_end is not None else event_start) >= start:
                        event_times.append(event_start)
        if not event_times:
            warnings.append(_issue("no_source_events", "No timestamped dialogue or actions from referenced scenes overlap this promo segment.", "medium", index))

        for seconds_field, expected in (("start_seconds", start), ("end_seconds", end)):
            if segment.get(seconds_field) is not None:
                try:
                    stored = float(segment[seconds_field])
                    if abs(stored - expected) > 1.0:
                        errors.append(_issue("timestamp_seconds_mismatch", f"{seconds_field} does not match the timestamp string.", segment_index=index))
                except (TypeError, ValueError):
                    errors.append(_issue("timestamp_seconds_invalid", f"{seconds_field} must be numeric.", segment_index=index))

        if previous is not None and previous[0] == episode_id and start < previous[2]:
            errors.append(_issue("segment_order", "Segments in the same episode overlap or are not ordered chronologically.", segment_index=index))
        previous = (episode_id, start, end)

    if parsed_segments and not minimum_duration <= calculated_total <= maximum_duration:
        errors.append(_issue("promo_duration", f"Calculated total duration {calculated_total:.2f}s is outside the {minimum_duration:g}-{maximum_duration:g} second range."))
    recorded_total = candidate.get("total_duration")
    try:
        if recorded_total is None or abs(float(recorded_total) - calculated_total) > 1.0:
            errors.append(_issue("total_duration_mismatch", f"Recorded total_duration {recorded_total!r} does not match calculated duration {calculated_total:.2f}s."))
    except (TypeError, ValueError):
        errors.append(_issue("invalid_total_duration", "total_duration must be numeric."))

    for key in ("hook", "cliffhanger"):
        value = candidate.get(key)
        if not isinstance(value, dict):
            errors.append(_issue(f"missing_{key}", f"{key} evidence object is required."))
        elif not (
            str(value.get("evidence", "")).strip()
            if isinstance(value.get("evidence"), str)
            else any(str(part).strip() for part in value.get("evidence", []) if isinstance(part, str))
            if isinstance(value.get("evidence"), list)
            else ""
        ):
            warnings.append(_issue(f"missing_{key}_evidence", f"{key} evidence text is empty.", "medium"))
        elif parse_timestamp(value.get("timestamp")) is None:
            errors.append(_issue(f"invalid_{key}_timestamp", f"{key} timestamp is invalid."))
        elif segments and isinstance(segments[0], dict):
            boundary_segment = segments[0] if key == "hook" else segments[-1]
            expected = boundary_segment.get("start_time" if key == "hook" else "end_time")
            if expected and abs((parse_timestamp(value.get("timestamp")) or 0) - (parse_timestamp(expected) or 0)) > 1.0:
                warnings.append(_issue(f"unsupported_{key}_timestamp", f"{key} timestamp does not match the fixed promo boundary.", "medium"))

    evaluation = candidate.get("evaluation")
    score_fields = (
        "hook_score", "cliffhanger_score", "standalone_context_score",
        "emotional_intensity_score", "narrative_coherence_score",
    )
    if isinstance(evaluation, dict):
        for key in score_fields:
            value = evaluation.get(key)
            if not isinstance(value, (int, float)) or not 1 <= value <= 10:
                errors.append(_issue("invalid_score", f"{key} must be numeric and within 1-10."))
    for key in ("overall_summary",):
        if not str(candidate.get(key, "")).strip():
            warnings.append(_issue("missing_summary", f"{key} is empty.", "low"))

    if original_candidate is not None:
        if candidate.get("segments") != original_candidate.get("segments"):
            errors.append(_issue("source_segment_changed", "Ranked segment data differs from the original candidate."))

    technical_status = "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS"
    return {
        "promo_id": promo_id,
        "technical_status": technical_status,
        "technical_errors": errors,
        "technical_warnings": warnings,
    }


def validate_unique_ids(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return duplicate promo ID validation errors for each affected record."""
    counts: dict[str, int] = {}
    for candidate in candidates:
        promo_id = str(candidate.get("promo_id", "<missing>"))
        counts[promo_id] = counts.get(promo_id, 0) + 1
    return [
        _issue("duplicate_promo_id", f"Promo ID {candidate.get('promo_id')} is not unique.")
        for candidate in candidates
        if counts.get(str(candidate.get("promo_id", "<missing>")), 0) > 1
    ]


__all__ = ["validate_candidate", "validate_unique_ids"]
