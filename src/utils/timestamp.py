"""Utilities for working with screenplay timestamps."""

from __future__ import annotations

import math
import re
from typing import Any

_TIMESTAMP_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d+))?$")


def parse_timestamp(value: Any) -> float | None:
    """Convert a screenplay timestamp into seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().strip("()")
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    match = _TIMESTAMP_RE.fullmatch(text)
    if not match:
        return None
    hours_part, minutes_part, seconds_part, fraction = match.groups()
    hours = int(hours_part or 0)
    minutes = int(minutes_part)
    seconds = float(f"{seconds_part}.{fraction or '0'}")
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_timestamp(seconds: float | int | str | None) -> str:
    """Convert a number of seconds to an HH:MM:SS timestamp."""
    if seconds is None or seconds == "":
        return "00:00:00"
    total = float(seconds)
    if math.isnan(total) or math.isinf(total):
        return "00:00:00"
    total = max(0.0, total)
    hours = int(total // 3600)
    remaining = total - hours * 3600
    minutes = int(remaining // 60)
    secs = remaining - minutes * 60
    if secs.is_integer():
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".rstrip("0").rstrip(".")


def duration_seconds(start: Any, end: Any) -> float:
    """Return the duration between two timestamps in seconds."""
    start_value = parse_timestamp(start)
    end_value = parse_timestamp(end)
    if start_value is None or end_value is None:
        raise ValueError(f"Both timestamps must be valid; got {start!r} and {end!r}.")
    return max(0.0, end_value - start_value)


def validate_duration_window(start: Any, end: Any, minimum: float = 20.0, maximum: float = 60.0) -> bool:
    """Validate that a timestamp window is within the allowed promo duration range."""
    try:
        duration = duration_seconds(start, end)
    except ValueError:
        return False
    return minimum <= duration <= maximum


__all__ = [
    "parse_timestamp",
    "seconds_to_timestamp",
    "duration_seconds",
    "validate_duration_window",
]
