"""Gemini LLM provider implementation."""

from __future__ import annotations

from typing import Any

from src.llm.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """Gemini provider wrapper around the official SDK."""

    provider_name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None, temperature: float = 0.2, timeout: int = 60, **_: Any) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url, temperature=temperature, timeout=timeout)
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError("google-generativeai is required for the Gemini provider. Install the dependency first.") from exc
        if not api_key:
            raise ValueError("LLM_API_KEY must be set when using the Gemini provider.")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name=model or "gemini-1.5-flash")

    def generate_json(self, prompt: str, response_schema: dict | None = None) -> dict:
        try:
            generation_config = {"temperature": self.temperature, "response_mime_type": "application/json"}
            if response_schema:
                generation_config["response_schema"] = response_schema
            response = self.client.generate_content(prompt, generation_config=generation_config)
            text = getattr(response, "text", None)
            if not text:
                text = response.candidates[0].content.parts[0].text
            return self.parse_json_response(text)
        except Exception as exc:
            self.handle_error(exc, "Gemini JSON generation")
            return {}

    def generate_text(self, prompt: str) -> str:
        try:
            response = self.client.generate_content(prompt, generation_config={"temperature": self.temperature})
            return getattr(response, "text", "")
        except Exception as exc:
            self.handle_error(exc, "Gemini text generation")
            return ""


__all__ = ["GeminiProvider"]
