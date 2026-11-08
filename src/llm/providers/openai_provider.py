"""OpenAI provider implementation."""

from __future__ import annotations

from typing import Any

from src.llm.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider wrapper around the official SDK."""

    provider_name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None, temperature: float = 0.2, timeout: int = 60, **_: Any) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url, temperature=temperature, timeout=timeout)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is required for the OpenAI provider. Install the dependency first.") from exc
        if not api_key:
            raise ValueError("LLM_API_KEY must be set when using the OpenAI provider.")
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model or "gpt-4o-mini"

    def generate_json(self, prompt: str, response_schema: dict | None = None) -> dict:
        try:
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt + "\nRespond with valid JSON only."}],
                "temperature": self.temperature,
            }
            if response_schema is not None:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception:
                kwargs.pop("response_format", None)
                response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return self.parse_json_response(content)
        except Exception as exc:
            self.handle_error(exc, "OpenAI JSON generation")
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
            self.handle_error(exc, "OpenAI text generation")
            return ""


__all__ = ["OpenAIProvider"]
