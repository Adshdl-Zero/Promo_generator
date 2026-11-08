"""Base interface for all LLM providers."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """Provider-independent interface for the analysis and generation pipeline."""

    provider_name = "base"

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None, temperature: float = 0.2, timeout: int = 60, **_: Any) -> None:
        self.model = model or "default-model"
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout

    @abstractmethod
    def generate_json(self, prompt: str, response_schema: dict | None = None) -> dict:
        """Generate a structured JSON response from the provider."""

    def generate_text(self, prompt: str) -> str:
        """Generate textual output if a provider requires it."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support text generation.")

    def handle_error(self, exc: Exception, context: str) -> None:
        """Translate provider-specific exceptions into a project-aware error."""
        raise RuntimeError(f"LLM provider {self.provider_name} failed during {context}: {exc}") from exc

    @staticmethod
    def parse_json_response(raw_response: str | dict | list | None) -> dict:
        """Parse a provider response into a JSON object."""
        if raw_response is None:
            return {}
        if isinstance(raw_response, dict):
            return raw_response
        if isinstance(raw_response, list):
            return {"items": raw_response}
        text = str(raw_response).strip()
        if not text:
            return {}
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|```\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    return {"items": parsed}
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse JSON from provider response: {text[:200]}")


__all__ = ["BaseLLMProvider"]
