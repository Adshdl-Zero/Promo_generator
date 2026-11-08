"""Simple caching utilities for LLM and analysis responses."""

from __future__ import annotations

import hashlib
import json
from typing import Any, MutableMapping


class SimpleCache:
    """A lightweight in-memory cache keyed by a deterministic hash."""

    def __init__(self) -> None:
        self._store: MutableMapping[str, Any] = {}

    @staticmethod
    def _make_key(payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, key: Any) -> Any:
        return self._store.get(self._make_key(key) if not isinstance(key, str) else key)

    def set(self, key: Any, value: Any) -> None:
        cache_key = self._make_key(key) if not isinstance(key, str) else key
        self._store[cache_key] = value

    def clear(self) -> None:
        self._store.clear()


__all__ = ["SimpleCache"]
