"""Mock LLM provider used for deterministic local development and tests."""

from __future__ import annotations

import json
import re
from typing import Any

from src.llm.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """A deterministic mock provider that returns schema-shaped outputs without external APIs."""

    provider_name = "mock"

    def generate_json(self, prompt: str, response_schema: dict | None = None) -> dict:
        prompt_lower = prompt.lower()
        if "ordered promo as one short-form story" in prompt_lower:
            return {
                "hook_strength": 8,
                "narrative_coherence": 7,
                "context_completeness": 7,
                "transition_quality": 7,
                "emotional_impact": 8,
                "cliffhanger_strength": 8,
                "standalone_understandability": 7,
                "overall_promotional_potential": 8,
                "strengths": ["The ordered moments establish a hook and build toward an unresolved ending."],
                "weaknesses": ["Context between source moments may need editorial review."],
                "explanation": "Mock evaluation for offline integration testing; not a human or provider judgment.",
            }
        if "scene" in prompt_lower and "episode_id" in prompt_lower:
            event_type = self._detect_event_type(prompt_lower)
            content_block = prompt.split("Scene content:\n", 1)[1] if "Scene content:\n" in prompt else prompt
            dialogue_hits = re.findall(r"(?P<ts>\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\s+(?P<character>[^:\n]+?):\s*\"?(?P<text>[^\n\"]+)\"?", content_block)
            action_hits = re.findall(r"(?P<ts>\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\s+ACTION:\s*(?P<text>.+)", content_block)
            episode_match = re.search(r"episode[_\- ]?id\s*[:=]?\s*(\d+)", prompt, flags=re.IGNORECASE)
            scene_match = re.search(r"scene[_\- ]?id\s*[:=]?\s*(\d+)", prompt, flags=re.IGNORECASE)
            episode_id = int(episode_match.group(1)) if episode_match else 1
            scene_id = int(scene_match.group(1)) if scene_match else 1

            if dialogue_hits:
                timestamp, character, text = dialogue_hits[0]
                dialogue_text = text.strip()
            else:
                timestamp, character, dialogue_text = None, None, "An accusation turns the moment into a confrontation."

            if action_hits:
                action_timestamp, action_text = action_hits[0]
            else:
                action_timestamp, action_text = timestamp, "A pivotal confrontation unfolds."

            return {
                "episode_id": episode_id,
                "scene_id": scene_id,
                "event_type": event_type,
                "dramatic_summary": self._summary_for_event(event_type, dialogue_text),
                "key_events": [
                    "A high-stakes emotional conflict escalates in the scene.",
                    "An unexpected revelation or confrontation changes the emotional direction."
                ],
                "emotional_intensity": 8,
                "suspense_level": 7,
                "hook_potential": 9,
                "cliffhanger_potential": 8,
                "standalone_potential": 7,
                "key_dialogue": [{
                    "text": dialogue_text,
                    "timestamp": timestamp,
                    "character": character.strip() if character else None,
                }],
                "important_actions": [{
                    "description": action_text.strip(),
                    "timestamp": action_timestamp,
                }],
                "unresolved_questions": ["What will happen after this moment?"],
                "context_dependencies": ["Prior relationship context and scene stakes."],
                "promo_relevance": "high",
            }
        return {
            "answer": "ok",
            "score": 8,
            "summary": "Mock provider generated a valid JSON response.",
        }

    @staticmethod
    def _detect_event_type(prompt: str) -> str:
        if any(term in prompt for term in ["lie", "betray", "abandon", "sister", "marrying"]):
            return "betrayal"
        if any(term in prompt for term in ["secret", "hidden", "identity", "reveal", "truth"]):
            return "unexpected_revelation"
        if any(term in prompt for term in ["threat", "danger", "attack", "gun", "escape"]):
            return "threat"
        if any(term in prompt for term in ["love", "kiss", "touch", "embrace", "romance"]):
            return "romantic_tension"
        if any(term in prompt for term in ["argue", "accuse", "confront", "angry"]):
            return "confrontation"
        return "dramatic_decision"

    @staticmethod
    def _summary_for_event(event_type: str, quote: str) -> str:
        summaries = {
            "betrayal": f"A betrayal changes the emotional stakes as the scene turns from trust to accusation: {quote}",
            "unexpected_revelation": f"An unexpected revelation reshapes the stakes and leaves the characters scrambling to respond: {quote}",
            "threat": f"A looming threat raises the tension and turns a quiet moment into a life-or-death decision: {quote}",
            "romantic_tension": f"Romantic tension reaches a breaking point and pushes the scene toward an emotional turning point: {quote}",
            "confrontation": f"A confrontation erupts with volatile emotion and exposes the deeper conflict beneath the surface: {quote}",
        }
        return summaries.get(event_type, f"A dramatic turning point demands a reaction and creates a compelling promo hook: {quote}")

    def generate_text(self, prompt: str) -> str:
        payload = self.generate_json(prompt)
        return json.dumps(payload, ensure_ascii=False)


__all__ = ["MockLLMProvider"]
