"""Small helpers for bounded transient-request retry backoff."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
import time
from threading import Lock
from typing import Any


def retry_delay_seconds(exc: Exception, attempt: int, base_seconds: float, maximum_seconds: float = 30.0) -> float:
    """Use Retry-After when available; otherwise exponential backoff, both capped."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    retry_after: Any = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is not None:
        try:
            return min(maximum_seconds, max(0.0, float(retry_after)))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(retry_after))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return min(maximum_seconds, max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds()))
            except (TypeError, ValueError, OverflowError):
                pass
    # Some compatible providers put their reset estimate in the error body
    # instead of a Retry-After response header (for example, "try again in 18s"
    # or "Please try again in 2m38.112s").
    message = str(exc)
    compound = re.search(
        r"(?:try again in|retry after)\s+(?:(\d+(?:\.\d+)?)m)?(\d+(?:\.\d+)?)s\b",
        message,
        re.IGNORECASE,
    )
    if compound:
        minutes = float(compound.group(1) or 0)
        seconds = float(compound.group(2) or 0)
        return min(maximum_seconds, minutes * 60 + seconds)
    minutes_only = re.search(r"(?:try again in|retry after)\s+(\d+)m\b", message, re.IGNORECASE)
    if minutes_only:
        return min(maximum_seconds, float(minutes_only.group(1)) * 60)
    match = re.search(r"(?:try again in|retry after)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)?\b", message, re.IGNORECASE)
    if match:
        return min(maximum_seconds, max(0.0, float(match.group(1))))
    return min(maximum_seconds, max(0.0, base_seconds) * (2**max(0, attempt)))


def is_rate_limit_error(error: Exception | str | None) -> bool:
    """Recognize common HTTP and provider-specific RPM/TPM quota errors."""
    message = str(error or "").lower()
    return any(marker in message for marker in (
        "429", "rate limit", "rate_limit", "tokens per minute", "tokens per day",
        "requests per minute", "too many requests", "quota exceeded",
    ))


class RequestPacer:
    """Serialize request starts so successive calls respect a minimum interval."""

    def __init__(self, minimum_interval_seconds: float = 0.0) -> None:
        self.minimum_interval_seconds = max(0.0, minimum_interval_seconds)
        self._last_request_at: float | None = None
        self._lock = Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_request_at is not None:
                delay = self.minimum_interval_seconds - (now - self._last_request_at)
                if delay > 0:
                    time.sleep(delay)
            self._last_request_at = time.monotonic()


__all__ = ["RequestPacer", "is_rate_limit_error", "retry_delay_seconds"]
