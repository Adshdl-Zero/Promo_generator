"""Pydantic response schemas for Module 4 ranking fallback scores."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CriterionEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    score: int = Field(ge=1, le=10)
    evidence: str = Field(min_length=1, max_length=500)


class MissingScoreAssessment(BaseModel):
    """Scores and concise evidence when an input candidate lacks criteria."""

    model_config = ConfigDict(extra="ignore")

    hook_strength: CriterionEvidence
    cliffhanger_strength: CriterionEvidence
    standalone_understandability: CriterionEvidence
    emotional_intensity: CriterionEvidence
    narrative_coherence: CriterionEvidence
    promo_engagement_potential: CriterionEvidence


__all__ = ["CriterionEvidence", "MissingScoreAssessment"]
