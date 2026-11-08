"""Pydantic models for promo candidate generation and evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PromoSegment(BaseModel):
    episode_id: int
    scene_ids: list[int] = Field(default_factory=list)
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    segment_duration: float | None = None
    segment_order: int | None = None


class HookInfo(BaseModel):
    description: str
    evidence: str
    timestamp: str


class CliffhangerInfo(BaseModel):
    description: str
    evidence: str
    timestamp: str


class CandidateEvaluation(BaseModel):
    hook_score: int = Field(ge=1, le=10)
    cliffhanger_score: int = Field(ge=1, le=10)
    standalone_context_score: int = Field(ge=1, le=10)
    emotional_intensity_score: int = Field(ge=1, le=10)
    narrative_coherence_score: int = Field(ge=1, le=10)
    promo_engagement_potential: int | None = Field(default=None, ge=1, le=10)


class PromoCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    promo_id: str
    segments: list[PromoSegment] = Field(default_factory=list)
    total_duration: float
    hook: HookInfo
    cliffhanger: CliffhangerInfo
    evaluation: CandidateEvaluation
    overall_summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    generation_method: Literal["event_anchored", "sliding_window", "multi_segment"]
    promo_type: Literal["continuous", "multi_segment"] = "continuous"


__all__ = ["PromoSegment", "HookInfo", "CliffhangerInfo", "CandidateEvaluation", "PromoCandidate"]
