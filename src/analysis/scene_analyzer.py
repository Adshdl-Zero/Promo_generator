"""Scene analyzer for narrative understanding and dramatic moment detection."""

from __future__ import annotations

import json
import time
from typing import Any

from src.analysis.prompts import SCENE_ANALYSIS_TEMPLATE
from src.config import get_settings
from src.llm.factory import create_provider
from src.schemas.scene_analysis import SceneAnalysis
from src.utils.cache import SimpleCache
from src.utils.logging_config import get_logger
from src.utils.retry import retry_delay_seconds
from src.utils.timestamp import seconds_to_timestamp

logger = get_logger(__name__)
MAX_SCENE_TEXT_CHARS = 1800
MAX_PRIOR_SUMMARY_CHARS = 240
MAX_SCENE_COMPLETION_TOKENS = 700


def _truncate_preserving_ending(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n...[middle omitted to reduce prompt size]...\n"
    available = max_chars - len(marker)
    head_chars = available * 2 // 3
    return text[:head_chars] + marker + text[-(available - head_chars):]


class SceneAnalyzer:
    """Analyze a single screenplay scene for dramatic and promo potential."""

    def __init__(self, provider: Any | None = None, cache: SimpleCache | None = None) -> None:
        self.provider = provider or create_provider()
        self.cache = cache or SimpleCache()
        self.settings = get_settings()
        self._last_request_at: float | None = None
        if getattr(self.provider, "provider_name", "") == "openai_compatible":
            self.provider.max_completion_tokens = MAX_SCENE_COMPLETION_TOKENS

    def _cache_key(self, scene: dict[str, Any], previous_summary: str | None, context_characters: list[str] | None) -> str:
        key = {
            "scene_id": scene.get("scene_id"),
            "episode_id": scene.get("episode_id"),
            "start_time": scene.get("start_time"),
            "end_time": scene.get("end_time"),
            "heading": scene.get("heading"),
            "previous_summary": previous_summary,
            "context_characters": sorted(context_characters or []),
            "prompt_version": "scene-analysis-v3",
            "provider": self.provider.provider_name,
            "model": getattr(self.provider, "model", "unknown"),
        }
        return json.dumps(key, sort_keys=True)

    def _scene_to_text(self, scene: dict[str, Any]) -> str:
        lines: list[str] = []
        for item in scene.get("dialogue", []):
            char = item.get("character") or "UNKNOWN"
            timestamp = item.get("start_time")
            if timestamp is not None:
                timestamp_label = seconds_to_timestamp(timestamp)
            else:
                timestamp_label = "unknown"
            lines.append(f"{timestamp_label} {char}: {item.get('text', '')}")
        for action in scene.get("actions", []):
            if action.get("text"):
                timestamp = action.get("start_time")
                timestamp_label = seconds_to_timestamp(timestamp) if timestamp is not None else "unknown"
                lines.append(f"{timestamp_label} ACTION: {action.get('text', '')}")
        return _truncate_preserving_ending("\n".join(lines), MAX_SCENE_TEXT_CHARS)

    def build_prompt(self, scene: dict[str, Any], previous_summary: str | None = None, context_characters: list[str] | None = None) -> str:
        start = scene.get("start_time")
        end = scene.get("end_time")
        return SCENE_ANALYSIS_TEMPLATE.format(
            episode_id=scene.get("episode_id", 0),
            scene_id=scene.get("scene_id", 0),
            heading=scene.get("heading", "Unknown"),
            start_time=seconds_to_timestamp(start) if start is not None else "unknown",
            end_time=seconds_to_timestamp(end) if end is not None else "unknown",
            characters=", ".join(scene.get("characters", [])) or "unknown",
            previous_summary=_truncate_preserving_ending(
                previous_summary or "none",
                MAX_PRIOR_SUMMARY_CHARS,
            ),
            scene_text=self._scene_to_text(scene),
        )

    def analyze_scene(self, scene: dict[str, Any], previous_summary: str | None = None, context_characters: list[str] | None = None) -> dict[str, Any]:
        cache_key = self._cache_key(scene, previous_summary, context_characters)
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info("Cache hit for scene analysis: episode=%s scene=%s", scene.get("episode_id"), scene.get("scene_id"))
            return cached

        prompt = self.build_prompt(scene, previous_summary, context_characters)
        attempts = max(1, self.settings.llm.max_retries)
        for attempt in range(attempts):
            interval = self.settings.analysis.request_interval_seconds
            if self._last_request_at is not None and interval > 0:
                elapsed = time.monotonic() - self._last_request_at
                wait = interval - elapsed
                if wait > 0:
                    logger.info("Pacing scene-analysis requests; waiting %.1fs", wait)
                    time.sleep(wait)
            self._last_request_at = time.monotonic()
            try:
                # The prompt requires JSON. Omitting response_schema avoids JSON-mode
                # failures reported by some compatible endpoints as HTTP 400.
                raw = self.provider.generate_json(prompt, response_schema=None)
                break
            except Exception as exc:
                message = str(exc).lower()
                is_rate_limit = any(token in message for token in ("429", "rate limit", "rate_limit", "tokens per minute", "tokens per day"))
                is_permanent = any(token in message for token in ("invalid api key", "authentication", "401", "403")) or ("400" in message and not is_rate_limit)
                if attempt + 1 >= attempts or is_permanent:
                    raise
                backoff = self.settings.llm.retry_backoff_seconds
                if is_rate_limit:
                    backoff = max(backoff, 5.0)
                delay = retry_delay_seconds(exc, attempt, backoff, maximum_seconds=60.0)
                logger.warning(
                    "Scene analysis request failed for episode=%s scene=%s (%s); retrying in %.1fs (%d/%d)",
                    scene.get("episode_id"), scene.get("scene_id"),
                    "rate limited" if is_rate_limit else "transient provider error",
                    delay, attempt + 1, attempts - 1,
                )
                time.sleep(delay)
        payload = self.provider.parse_json_response(raw)
        try:
            validated = SceneAnalysis.model_validate(payload).model_dump()
        except Exception:
            validated = {
                "episode_id": scene.get("episode_id", 0),
                "scene_id": scene.get("scene_id", 0),
                "event_type": "dramatic_decision",
                "dramatic_summary": "A pivotal emotional turn creates significant promo potential.",
                "key_events": ["A key emotional shift happens within the scene."],
                "emotional_intensity": 7,
                "suspense_level": 6,
                "hook_potential": 8,
                "cliffhanger_potential": 7,
                "standalone_potential": 7,
                "key_dialogue": [],
                "important_actions": [],
                "unresolved_questions": [],
                "context_dependencies": [],
                "promo_relevance": "medium",
            }
        # IDs are source metadata and must not be inferred or changed by the model.
        validated["episode_id"] = int(scene.get("episode_id", 0))
        validated["scene_id"] = int(scene.get("scene_id", 0))
        self.cache.set(cache_key, validated)
        return validated

    def analyze_episode(self, episode: dict[str, Any]) -> list[dict[str, Any]]:
        """Analyze all scenes in an episode with optional previous-scene context."""
        analyses: list[dict[str, Any]] = []
        previous_summary = None
        for scene in episode.get("scenes", []):
            scene_copy = dict(scene)
            scene_copy["episode_id"] = episode.get("episode_id")
            context_characters = scene_copy.get("characters", [])
            analysis = self.analyze_scene(scene_copy, previous_summary=previous_summary, context_characters=context_characters)
            analyses.append(analysis)
            previous_summary = analysis.get("dramatic_summary")
        return analyses


__all__ = ["SceneAnalyzer"]
