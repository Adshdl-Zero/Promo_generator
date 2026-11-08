"""Sliding-window and event-anchored promo generation utilities."""

from __future__ import annotations

from typing import Any

from src.utils.timestamp import parse_timestamp, seconds_to_timestamp


class WindowGenerator:
    """Build candidate windows around scene boundaries and high-potential events."""

    def __init__(self, min_duration: int = 20, max_duration: int = 60, step: int = 5) -> None:
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.step = step

    def generate_event_anchored(self, episode: dict[str, Any], scene_analyses: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        scenes = episode.get("scenes", [])
        if not scenes:
            return []
        candidate_windows: list[dict[str, Any]] = []
        for index, scene in enumerate(scenes):
            start = scene.get("start_time")
            end = scene.get("end_time")
            if parse_timestamp(start) is None or parse_timestamp(end) is None:
                continue
            if index + 1 < len(scenes):
                next_scene = scenes[index + 1]
                next_start = next_scene.get("start_time")
                if next_start is not None:
                    end = max(float(end), float(next_start))
            for duration in range(self.min_duration, self.max_duration + 1, self.step):
                window_end = float(start) + duration
                if window_end <= float(end):
                    candidate_windows.append({
                        "episode_id": episode.get("episode_id"),
                        "scene_ids": [scene.get("scene_id")],
                        "start_time": seconds_to_timestamp(start),
                        "end_time": seconds_to_timestamp(window_end),
                        "start_seconds": int(parse_timestamp(start) or 0),
                        "end_seconds": int(window_end),
                        "generation_method": "event_anchored",
                    })
        return candidate_windows

    def generate_sliding_windows(self, episode: dict[str, Any]) -> list[dict[str, Any]]:
        scenes = episode.get("scenes", [])
        if not scenes:
            return []
        windows: list[dict[str, Any]] = []
        max_episode_end = max(parse_timestamp(scene.get("end_time")) or 0 for scene in scenes)
        for duration in range(self.min_duration, self.max_duration + 1, self.step):
            for start in range(0, int(max_episode_end) + 1, self.step):
                end = start + duration
                if end > max_episode_end:
                    continue
                scene_ids = [scene.get("scene_id") for scene in scenes if (parse_timestamp(scene.get("start_time")) or 0) <= start and (parse_timestamp(scene.get("end_time")) or 0) >= end]
                if not scene_ids:
                    continue
                windows.append({
                    "episode_id": episode.get("episode_id"),
                    "scene_ids": scene_ids,
                    "start_time": seconds_to_timestamp(start),
                    "end_time": seconds_to_timestamp(end),
                    "start_seconds": start,
                    "end_seconds": end,
                    "generation_method": "sliding_window",
                })
        return windows


__all__ = ["WindowGenerator"]
