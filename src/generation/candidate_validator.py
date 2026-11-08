"""Validation logic for generated promo candidates."""

from __future__ import annotations

from typing import Any

from src.utils.timestamp import duration_seconds, parse_timestamp


class CandidateValidator:
    """Validate and normalize generated promo windows."""

    def __init__(self, min_duration: int = 20, max_duration: int = 60) -> None:
        self.min_duration = min_duration
        self.max_duration = max_duration

    def validate(self, candidate: dict[str, Any]) -> bool:
        segments = candidate.get("segments", [])
        if not segments:
            return False
        total_duration = 0.0
        for segment in segments:
            start = segment.get("start_time")
            end = segment.get("end_time")
            if parse_timestamp(start) is None or parse_timestamp(end) is None:
                return False
            total_duration += duration_seconds(start, end)
            if segment.get("episode_id") is None:
                return False
        return self.min_duration <= total_duration <= self.max_duration

    def deduplicate(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str, int]] = set()
        unique: list[dict[str, Any]] = []
        for candidate in candidates:
            segments = candidate.get("segments") or []
            if not segments:
                continue
            first = segments[0]
            key = (str(first.get("episode_id", "")), str(first.get("start_time", "")), str(first.get("end_time", "")))
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique


__all__ = ["CandidateValidator"]
