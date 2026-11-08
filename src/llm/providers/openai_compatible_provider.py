"""OpenAI-compatible provider implementation for Groq and similar services."""

from __future__ import annotations

from typing import Any

from src.llm.base import BaseLLMProvider


def _is_response_format_unsupported(exc: Exception) -> bool:
    """Retry without JSON mode only when the JSON response mode itself failed."""
    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()
    if status_code != 400:
        return False
    rejects_option = "response_format" in message and any(
        marker in message
        for marker in ("unsupported", "not supported", "unrecognized", "unknown parameter", "invalid parameter")
    )
    rejects_json_output = "validate json" in message or "failed to validate json" in message
    return rejects_option or rejects_json_output


class OpenAICompatibleProvider(BaseLLMProvider):
    """Provider wrapper for OpenAI-compatible APIs such as Groq."""

    provider_name = "openai_compatible"

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None, temperature: float = 0.2, timeout: int = 60, **_: Any) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url, temperature=temperature, timeout=timeout)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is required for OpenAI-compatible APIs. Install the dependency first.") from exc
        if not api_key:
            raise ValueError("LLM_API_KEY must be set when using an OpenAI-compatible provider.")
        self.client = OpenAI(api_key=api_key, base_url=base_url or "https://api.openai.com/v1", timeout=timeout)
        self.model = model or "meta-llama/llama-3.1-8b-instant"

    def generate_json(self, prompt: str, response_schema: dict | None = None) -> dict:
        try:
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt + "\nRespond with valid JSON only."}],
                "temperature": self.temperature,
            }
            max_completion_tokens = getattr(self, "max_completion_tokens", None)
            if max_completion_tokens is not None:
                kwargs["max_completion_tokens"] = max_completion_tokens
            if response_schema is not None:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                if "response_format" not in kwargs or not _is_response_format_unsupported(exc):
                    raise
                kwargs.pop("response_format", None)
                response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return self.parse_json_response(content)
        except Exception as exc:
            self.handle_error(exc, "OpenAI-compatible JSON generation")
            return {}

    def generate_text(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            self.handle_error(exc, "OpenAI-compatible text generation")
            return ""


__all__ = ["OpenAICompatibleProvider"]
