"""Schema models for analysis and promo generation."""

from .promo_candidate import CandidateEvaluation, CliffhangerInfo, HookInfo, PromoCandidate, PromoSegment
from .scene_analysis import ActionEvidence, DialogueEvidence, SceneAnalysis

__all__ = [
    "ActionEvidence",
    "CandidateEvaluation",
    "CliffhangerInfo",
    "DialogueEvidence",
    "HookInfo",
    "PromoCandidate",
    "PromoSegment",
    "SceneAnalysis",
]
