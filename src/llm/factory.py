"""Factory for constructing LLM providers based on configuration."""

from __future__ import annotations

from src.config import get_settings
from src.llm.base import BaseLLMProvider
from src.llm.providers.gemini_provider import GeminiProvider
from src.llm.providers.mock_provider import MockLLMProvider
from src.llm.providers.openai_compatible_provider import OpenAICompatibleProvider
from src.llm.providers.openai_provider import OpenAIProvider


def create_provider(provider_name: str | None = None) -> BaseLLMProvider:
    """Instantiate the configured provider or the named provider."""
    settings = get_settings()
    provider = (provider_name or settings.llm.provider or "mock").strip().lower()
    if provider == "mock":
        return MockLLMProvider(model=settings.llm.model, api_key=settings.llm.api_key, temperature=settings.llm.temperature)
    if provider == "gemini":
        return GeminiProvider(model=settings.llm.model, api_key=settings.llm.api_key, temperature=settings.llm.temperature, timeout=settings.llm.timeout)
    if provider == "openai":
        return OpenAIProvider(model=settings.llm.model, api_key=settings.llm.api_key, temperature=settings.llm.temperature, timeout=settings.llm.timeout)
    if provider in {"openai_compatible", "groq", "compatible"}:
        return OpenAICompatibleProvider(model=settings.llm.model, api_key=settings.llm.api_key, base_url=settings.llm.base_url, temperature=settings.llm.temperature, timeout=settings.llm.timeout)
    raise ValueError(f"Unsupported provider: {provider}")


__all__ = ["create_provider"]
