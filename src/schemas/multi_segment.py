"""Schemas for Module 6 multi-segment narrative evaluation and candidates."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.promo_candidate import PromoSegment


class MultiSegmentNarrativeEvaluation(BaseModel):
    """LLM assessment of the complete ordered promo, not per-segment averages."""

    model_config = ConfigDict(extra="ignore")

    hook_strength: int = Field(ge=1, le=10)
    narrative_coherence: int = Field(ge=1, le=10)
    context_completeness: int = Field(ge=1, le=10)
    transition_quality: int = Field(ge=1, le=10)
    emotional_impact: int = Field(ge=1, le=10)
    cliffhanger_strength: int = Field(ge=1, le=10)
    standalone_understandability: int = Field(ge=1, le=10)
    overall_promotional_potential: int = Field(ge=1, le=10)
    strengths: list[str] = Field(default_factory=list, max_length=10)
    weaknesses: list[str] = Field(default_factory=list, max_length=10)
    explanation: str = Field(min_length=1, max_length=1200)


class MultiSegmentCandidate(BaseModel):
    """A promo assembled from source-grounded timestamp windows."""

    model_config = ConfigDict(extra="allow")

    promo_id: str
    promo_type: Literal["multi_segment"] = "multi_segment"
    generation_method: Literal["multi_segment"] = "multi_segment"
    generation_strategy: Literal["same_episode", "cross_episode", "mixed"]
    segment_count: int = Field(ge=2, le=4)
    segments: list[PromoSegment] = Field(min_length=2, max_length=4)
    episode_ids: list[int]
    total_duration: float = Field(ge=20, le=60)
    hook: dict[str, str]
    cliffhanger: dict[str, str]
    combined_summary: str
    narrative_coherence_score: int = Field(ge=1, le=10)
    standalone_understandability_score: int = Field(ge=1, le=10)
    overall_promotional_score: int = Field(ge=1, le=10)
    evaluation: dict[str, int]
    strengths: list[str]
    weaknesses: list[str]
    generation_explanation: str
    validation_status: Literal["valid", "invalid"]


class HumanReview(BaseModel):
    """One reviewer submission for a promo; scores are intentionally separate from AI fields."""

    model_config = ConfigDict(extra="ignore")

    promo_id: str
    reviewer_id: str = Field(default="anonymous", min_length=1, max_length=80)
    reviewed_at: str
    overall_label: Literal["Strong", "Average", "Poor"]
    hook_rating: int | None = Field(default=None, ge=1, le=5)
    cliffhanger_rating: int | None = Field(default=None, ge=1, le=5)
    standalone_rating: int | None = Field(default=None, ge=1, le=5)
    coherence_rating: int | None = Field(default=None, ge=1, le=5)
    emotional_impact_rating: int | None = Field(default=None, ge=1, le=5)
    promotional_potential_rating: int | None = Field(default=None, ge=1, le=5)
    comments: str = Field(default="", max_length=3000)
    failure_categories: list[str] = Field(default_factory=list, max_length=20)
    incorrect_timestamps: bool = False
    missing_context: bool = False


__all__ = ["MultiSegmentNarrativeEvaluation", "MultiSegmentCandidate", "HumanReview"]
