"""Pydantic schemas for independent Module 5 quality evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class NarrativeCriterion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    score: int = Field(ge=1, le=10)
    explanation: str = Field(min_length=1, max_length=600)
    evidence: list[str] = Field(default_factory=list, max_length=10)
    weaknesses: list[str] = Field(default_factory=list, max_length=10)


class FailureFinding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: str = Field(min_length=1, max_length=80)
    severity: Literal["low", "medium", "high"]
    explanation: str = Field(min_length=1, max_length=600)
    evidence: list[str] = Field(default_factory=list, max_length=10)
    suggested_improvement: str = Field(default="", max_length=500)


class NarrativeQualityEvaluation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    opening_hook: NarrativeCriterion
    ending_cliffhanger: NarrativeCriterion
    standalone_understandability: NarrativeCriterion
    narrative_coherence: NarrativeCriterion
    emotional_engagement: NarrativeCriterion
    failure_findings: list[FailureFinding] = Field(default_factory=list, max_length=20)
    overall_explanation: str = Field(min_length=1, max_length=1000)
    overall_explanation_derived: bool = False


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: str
    severity: Literal["low", "medium", "high"]
    message: str
    segment_index: int | None = None


class CandidateQualityResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    promo_id: str
    technical_status: Literal["PASS", "PASS_WITH_WARNINGS", "FAIL"]
    technical_errors: list[ValidationIssue] = Field(default_factory=list)
    technical_warnings: list[ValidationIssue] = Field(default_factory=list)
    narrative_evaluation: NarrativeQualityEvaluation | None = None
    narrative_average_score: float | None = None
    failure_labels: list[FailureFinding] = Field(default_factory=list)
    quality_decision: Literal["PASS", "PASS_WITH_WARNINGS", "FAIL"]
    explanation: str


__all__ = [
    "NarrativeCriterion",
    "FailureFinding",
    "NarrativeQualityEvaluation",
    "ValidationIssue",
    "CandidateQualityResult",
]
