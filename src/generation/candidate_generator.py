"""Generate promotional candidate windows from a parsed episode."""

from __future__ import annotations

from typing import Any

from src.generation.candidate_validator import CandidateValidator
from src.generation.window_generator import WindowGenerator
from src.utils.timestamp import duration_seconds, parse_timestamp, seconds_to_timestamp


class CandidateGenerator:
    """Deterministic candidate builder that uses scene timestamps and narrative scores."""

    def __init__(self, min_duration: int = 20, max_duration: int = 60, preferred_methods: list[str] | None = None) -> None:
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.preferred_methods = preferred_methods or ["event_anchored", "sliding_window"]
        self.window_generator = WindowGenerator(min_duration=min_duration, max_duration=max_duration, step=5)
        self.validator = CandidateValidator(min_duration=min_duration, max_duration=max_duration)

    def _window_candidate(self, episode: dict[str, Any], start_seconds: float, end_seconds: float, scene_ids: list[int], method: str) -> dict[str, Any]:
        start_time = seconds_to_timestamp(start_seconds)
        end_time = seconds_to_timestamp(end_seconds)
        total_duration = int(duration_seconds(start_time, end_time))
        return {
            "segments": [{
                "episode_id": episode.get("episode_id"),
                "scene_ids": scene_ids,
                "start_time": start_time,
                "end_time": end_time,
                "start_seconds": int(start_seconds),
                "end_seconds": int(end_seconds),
            }],
            "total_duration": total_duration,
            "generation_method": method,
        }

    def generate_for_episode(self, episode: dict[str, Any], scene_analyses: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Generate promotional windows for a single episode."""
        episodes = episode.get("scenes", [])
        if not episodes:
            return []

        generated: list[dict[str, Any]] = []
        if "event_anchored" in self.preferred_methods:
            raw_windows = self.window_generator.generate_event_anchored(episode, scene_analyses)
            for item in raw_windows:
                start = parse_timestamp(item["start_time"])
                end = parse_timestamp(item["end_time"])
                if start is None or end is None:
                    continue
                if self.min_duration <= (end - start) <= self.max_duration:
                    generated.append(self._window_candidate(episode, start, end, item["scene_ids"], "event_anchored"))
        if "sliding_window" in self.preferred_methods:
            raw_windows = self.window_generator.generate_sliding_windows(episode)
            for item in raw_windows:
                start = parse_timestamp(item["start_time"])
                end = parse_timestamp(item["end_time"])
                if start is None or end is None:
                    continue
                if self.min_duration <= (end - start) <= self.max_duration:
                    generated.append(self._window_candidate(episode, start, end, item["scene_ids"], "sliding_window"))

        filtered = [candidate for candidate in generated if self.validator.validate(candidate)]
        return self.validator.deduplicate(filtered)


__all__ = ["CandidateGenerator"]
