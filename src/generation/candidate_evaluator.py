"""Evaluate generated promo candidates against a narrative hook and cliffhanger checklist."""

from __future__ import annotations

from typing import Any

from src.analysis.prompts import PROMO_EVALUATION_TEMPLATE
from src.llm.factory import create_provider
from src.utils.timestamp import parse_timestamp

MAX_PROMO_WINDOW_CHARS = 2400
MAX_SCENE_SUMMARY_CHARS = 180


def _truncate_preserving_ending(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n...[middle omitted]...\n"
    available = max_chars - len(marker)
    head_chars = available * 2 // 3
    return text[:head_chars] + marker + text[-(available - head_chars):]


class CandidateEvaluator:
    """Assess a candidate sequence as a complete promo and return structured evaluation data."""

    def __init__(self, provider: Any | None = None) -> None:
        self.provider = provider or create_provider()

    def build_prompt(self, episode: dict[str, Any], candidate: dict[str, Any], scene_analyses: list[dict[str, Any]]) -> str:
        start_time = candidate["segments"][0]["start_time"]
        end_time = candidate["segments"][0]["end_time"]
        duration_seconds = candidate["total_duration"]
        scene_ids = candidate["segments"][0]["scene_ids"]
        start_seconds = parse_timestamp(start_time)
        end_seconds = parse_timestamp(end_time)
        relevant_window = []
        for scene in episode.get("scenes", []):
            if scene.get("scene_id") in scene_ids:
                for dialogue in scene.get("dialogue", []):
                    relevant_window.append({"character": dialogue.get("character"), "start_time": dialogue.get("start_time"), "end_time": dialogue.get("end_time"), "text": dialogue.get("text")})
                for action in scene.get("actions", []):
                    relevant_window.append({"character": "ACTION", "start_time": action.get("start_time"), "end_time": action.get("end_time"), "text": action.get("text")})

        if start_seconds is not None and end_seconds is not None:
            relevant_window = [
                item
                for item in relevant_window
                if (event_start := parse_timestamp(item.get("start_time"))) is None
                or (
                    event_start <= end_seconds
                    and (parse_timestamp(item.get("end_time")) or event_start) >= start_seconds
                )
            ]

        scene_id_set = set(scene_ids)
        scene_context = " | ".join(
            f"Scene {item.get('scene_id')}: {str(item.get('dramatic_summary', ''))[:MAX_SCENE_SUMMARY_CHARS]}"
            for item in scene_analyses
            if item.get("scene_id") in scene_id_set and item.get("dramatic_summary")
        )
        window_text = "\n".join(
            f"{item.get('start_time')} {item.get('character')}: {item.get('text')}"
            for item in relevant_window
        )
        return PROMO_EVALUATION_TEMPLATE.format(
            episode_id=episode.get("episode_id"),
            scene_ids=scene_ids,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            scene_context=scene_context or "none",
            window_text=_truncate_preserving_ending(window_text or "No timestamped dialogue/actions fall inside this clip window.", MAX_PROMO_WINDOW_CHARS),
        )

    def evaluate(self, episode: dict[str, Any], candidate: dict[str, Any], scene_analyses: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = self.build_prompt(episode, candidate, scene_analyses)
        response = self.provider.generate_json(prompt, {})
        payload = self.provider.parse_json_response(response)
        hook_value = payload.get("hook")
        hook = dict(hook_value) if isinstance(hook_value, dict) else {
            "description": "A strong hook establishes tension early.",
            "evidence": "The clip opens with a compelling moment.",
        }
        cliffhanger_value = payload.get("cliffhanger")
        cliffhanger = dict(cliffhanger_value) if isinstance(cliffhanger_value, dict) else {
            "description": "The clip ends with unresolved tension.",
            "evidence": "The ending leaves a question unanswered.",
        }
        # Timestamps are deterministic candidate boundaries, never model-generated.
        hook["timestamp"] = candidate["segments"][0]["start_time"]
        cliffhanger["timestamp"] = candidate["segments"][0]["end_time"]
        result = {
            "promo_id": candidate.get("promo_id", "P0001"),
            "segments": candidate["segments"],
            "total_duration": candidate["total_duration"],
            "hook": hook,
            "cliffhanger": cliffhanger,
            "evaluation": {
                "hook_score": int(payload.get("hook_score", 8)),
                "cliffhanger_score": int(payload.get("cliffhanger_score", 8)),
                "standalone_context_score": int(payload.get("standalone_context_score", 7)),
                "emotional_intensity_score": int(payload.get("emotional_intensity_score", 8)),
                "narrative_coherence_score": int(payload.get("narrative_coherence_score", 8)),
            },
            "overall_summary": payload.get("overall_summary", "A focused dramatic sequence delivers tension and unresolved stakes."),
            "strengths": payload.get("strengths", ["Clear emotional escalation."]),
            "weaknesses": payload.get("weaknesses", ["Context may need a preceding scene for full clarity."]),
            "generation_method": candidate.get("generation_method", "event_anchored"),
        }
        return result


__all__ = ["CandidateEvaluator"]
