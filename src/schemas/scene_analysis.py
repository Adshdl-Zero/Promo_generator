"""Pydantic models for scene-level narrative analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DialogueEvidence(BaseModel):
    text: str
    timestamp: str | None = None
    character: str | None = None


class ActionEvidence(BaseModel):
    description: str
    timestamp: str | None = None


class SceneAnalysis(BaseModel):
    """Narrative and promotional assessment of a single scene."""

    model_config = ConfigDict(extra="ignore")

    episode_id: int
    scene_id: int
    event_type: str
    dramatic_summary: str
    key_events: list[str] = Field(default_factory=list)
    emotional_intensity: int = Field(ge=1, le=10)
    suspense_level: int = Field(ge=1, le=10)
    hook_potential: int = Field(ge=1, le=10)
    cliffhanger_potential: int = Field(ge=1, le=10)
    standalone_potential: int = Field(ge=1, le=10)
    key_dialogue: list[DialogueEvidence] = Field(default_factory=list)
    important_actions: list[ActionEvidence] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    context_dependencies: list[str] = Field(default_factory=list)
    promo_relevance: Literal["low", "medium", "high"] = "low"


__all__ = ["SceneAnalysis", "DialogueEvidence", "ActionEvidence"]
