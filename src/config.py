"""Configuration loader for provider selection and analysis defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"


@dataclass
class LLMSettings:
    provider: str = "mock"
    model: str = "mock-model"
    api_key: str | None = None
    temperature: float = 0.2
    max_retries: int = 3
    timeout: int = 60
    retry_backoff_seconds: float = 1.0
    base_url: str | None = None


@dataclass
class AnalysisSettings:
    batch_size: int = 5
    context_enabled: bool = True
    min_promo_relevance: int = 6
    context_window_scenes: int = 1
    request_interval_seconds: float = 12.0


@dataclass
class CandidateGenerationSettings:
    min_duration: int = 20
    max_duration: int = 60
    sliding_step: int = 5
    durations: list[int] = field(default_factory=lambda: [20, 30, 40, 50, 60])
    methods: list[str] = field(default_factory=lambda: ["event_anchored", "sliding_window"])


@dataclass
class EvaluationSettings:
    minimum_hook_score: int = 6
    minimum_cliffhanger_score: int = 6
    minimum_context_score: int = 5
    max_candidates: int = 25


@dataclass
class RankingSettings:
    weights: dict[str, float] = field(default_factory=lambda: {
        "hook_strength": 0.25,
        "cliffhanger_strength": 0.25,
        "standalone_understandability": 0.20,
        "emotional_intensity": 0.15,
        "narrative_coherence": 0.10,
        "promo_engagement_potential": 0.05,
    })
    category_thresholds: dict[str, float] = field(default_factory=lambda: {
        "excellent": 85,
        "strong": 70,
        "moderate": 55,
    })
    minimum_selection_score: float = 50
    maximum_selected_promos: int = 20
    maximum_promos_per_episode: int = 3
    maximum_scene_overlap: float = 0.6
    diversity_penalty: float = 10
    similarity_threshold: float = 0.5
    max_missing_score_llm_calls: int = 0
    request_interval_seconds: float = 12.0


@dataclass
class QualitySettings:
    high_rank_threshold: int = 5
    low_quality_threshold: float = 5.0
    narrative_warning_threshold: float = 6.0
    maximum_narrative_evaluations: int = 20
    max_narrative_attempts: int = 1
    concurrency: int = 1
    request_interval_seconds: float = 12.0
    cache_path: str = "outputs/quality_evaluation_and_validation/quality_evaluation_cache.json"


@dataclass
class MultiSegmentSettings:
    enabled: bool = True
    same_episode_enabled: bool = True
    cross_episode_enabled: bool = True
    min_segments: int = 2
    max_segments: int = 4
    minimum_duration: int = 20
    maximum_duration: int = 60
    minimum_segment_duration: int = 6
    maximum_segment_duration: int = 24
    max_moments_per_episode: int = 12
    max_candidates_per_episode: int = 10
    max_combinations: int = 100
    max_llm_calls: int = 1
    request_interval_seconds: float = 12.0
    minimum_quality_score: int = 6
    cache_path: str = "outputs/multi_segment_promo_generation/evaluation_cache.json"
    checkpoint_path: str = "outputs/multi_segment_promo_generation/checkpoint.json"


@dataclass
class SystemEvaluationSettings:
    human_review_sample_size: int = 30
    human_rating_scale: int = 5
    ranking_cutoffs: list[int] = field(default_factory=lambda: [5, 10, 20])
    review_dataset_path: str = "outputs/evaluation/human_review_dataset.csv"
    human_reviews_path: str = "outputs/evaluation/human_reviews.json"
    human_reviews_csv_path: str = "outputs/evaluation/human_reviews.csv"
    source_text_limit: int = 1800


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    candidate_generation: CandidateGenerationSettings = field(default_factory=CandidateGenerationSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    ranking: RankingSettings = field(default_factory=RankingSettings)
    quality: QualitySettings = field(default_factory=QualitySettings)
    multi_segment: MultiSegmentSettings = field(default_factory=MultiSegmentSettings)
    system_evaluation: SystemEvaluationSettings = field(default_factory=SystemEvaluationSettings)

    @classmethod
    def from_env(cls) -> "Settings":
        # Treat this workspace's .env as authoritative so long-lived VS Code
        # terminals cannot keep using a stale provider key after .env changes.
        load_dotenv(ROOT_DIR / ".env", override=True)
        config: Dict[str, Any] = {}
        config_path = DEFAULT_CONFIG_PATH
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}

        llm_cfg = config.get("llm", {})
        analysis_cfg = config.get("analysis", {})
        generation_cfg = config.get("candidate_generation", {})
        evaluation_cfg = config.get("evaluation", {})
        ranking_cfg = config.get("ranking", {})
        quality_cfg = config.get("quality", {})
        multi_cfg = config.get("multi_segment", {})
        system_eval_cfg = config.get("system_evaluation", {})

        return cls(
            llm=LLMSettings(
                provider=os.getenv("LLM_PROVIDER", str(llm_cfg.get("provider", "mock"))),
                model=os.getenv("LLM_MODEL", str(llm_cfg.get("model", "mock-model"))),
                api_key=os.getenv("LLM_API_KEY") or llm_cfg.get("api_key"),
                temperature=float(os.getenv("LLM_TEMPERATURE", str(llm_cfg.get("temperature", 0.2)))),
                max_retries=int(os.getenv("LLM_MAX_RETRIES", str(llm_cfg.get("max_retries", 3)))),
                timeout=int(os.getenv("LLM_TIMEOUT", str(llm_cfg.get("timeout", 60)))),
                retry_backoff_seconds=float(llm_cfg.get("retry_backoff_seconds", 1.0)),
                base_url=os.getenv("LLM_BASE_URL") or llm_cfg.get("base_url"),
            ),
            analysis=AnalysisSettings(
                batch_size=int(analysis_cfg.get("batch_size", 5)),
                context_enabled=bool(analysis_cfg.get("context_enabled", True)),
                min_promo_relevance=int(analysis_cfg.get("minimum_promo_relevance", analysis_cfg.get("min_promo_relevance", 6))),
                context_window_scenes=int(analysis_cfg.get("context_window_scenes", 1)),
                request_interval_seconds=max(0.0, float(analysis_cfg.get("request_interval_seconds", 12.0))),
            ),
            candidate_generation=CandidateGenerationSettings(
                min_duration=int(generation_cfg.get("min_duration", 20)),
                max_duration=int(generation_cfg.get("max_duration", 60)),
                sliding_step=int(generation_cfg.get("sliding_step", 5)),
                durations=list(generation_cfg.get("durations", [20, 30, 40, 50, 60])),
                methods=list(generation_cfg.get("methods", ["event_anchored", "sliding_window"])),
            ),
            evaluation=EvaluationSettings(
                minimum_hook_score=int(evaluation_cfg.get("minimum_hook_score", 6)),
                minimum_cliffhanger_score=int(evaluation_cfg.get("minimum_cliffhanger_score", 6)),
                minimum_context_score=int(evaluation_cfg.get("minimum_context_score", 5)),
                max_candidates=int(evaluation_cfg.get("max_candidates", 25)),
            ),
            ranking=RankingSettings(
                weights={**RankingSettings().weights, **ranking_cfg.get("weights", {})},
                category_thresholds={**RankingSettings().category_thresholds, **ranking_cfg.get("category_thresholds", {})},
                minimum_selection_score=float(ranking_cfg.get("minimum_selection_score", 50)),
                maximum_selected_promos=int(ranking_cfg.get("maximum_selected_promos", 20)),
                maximum_promos_per_episode=int(ranking_cfg.get("maximum_promos_per_episode", 3)),
                maximum_scene_overlap=float(ranking_cfg.get("maximum_scene_overlap", 0.6)),
                diversity_penalty=float(ranking_cfg.get("diversity_penalty", 10)),
                similarity_threshold=float(ranking_cfg.get("similarity_threshold", 0.5)),
                max_missing_score_llm_calls=max(0, int(ranking_cfg.get("max_missing_score_llm_calls", 0))),
                request_interval_seconds=max(0.0, float(ranking_cfg.get("request_interval_seconds", 12.0))),
            ),
            quality=QualitySettings(
                high_rank_threshold=int(quality_cfg.get("high_rank_threshold", 5)),
                low_quality_threshold=float(quality_cfg.get("low_quality_threshold", 5)),
                narrative_warning_threshold=float(quality_cfg.get("narrative_warning_threshold", 6)),
                maximum_narrative_evaluations=max(0, int(quality_cfg.get("maximum_narrative_evaluations", 1))),
                max_narrative_attempts=max(1, int(quality_cfg.get("max_narrative_attempts", 1))),
                concurrency=max(1, int(quality_cfg.get("concurrency", 1))),
                request_interval_seconds=max(0.0, float(quality_cfg.get("request_interval_seconds", 12.0))),
                cache_path=str(quality_cfg.get("cache_path", "outputs/quality_evaluation_and_validation/quality_evaluation_cache.json")),
            ),
            multi_segment=MultiSegmentSettings(
                enabled=bool(multi_cfg.get("enabled", True)),
                same_episode_enabled=bool(multi_cfg.get("same_episode_enabled", True)),
                cross_episode_enabled=bool(multi_cfg.get("cross_episode_enabled", True)),
                min_segments=max(2, int(multi_cfg.get("min_segments", 2))),
                max_segments=min(4, max(2, int(multi_cfg.get("max_segments", 4)))),
                minimum_duration=int(multi_cfg.get("minimum_duration", 20)),
                maximum_duration=int(multi_cfg.get("maximum_duration", 60)),
                minimum_segment_duration=int(multi_cfg.get("minimum_segment_duration", 6)),
                maximum_segment_duration=int(multi_cfg.get("maximum_segment_duration", 24)),
                max_moments_per_episode=max(1, int(multi_cfg.get("max_moments_per_episode", 12))),
                max_candidates_per_episode=max(1, int(multi_cfg.get("max_candidates_per_episode", 10))),
                max_combinations=max(1, int(multi_cfg.get("max_combinations", 100))),
                max_llm_calls=max(0, int(multi_cfg.get("max_llm_calls", 1))),
                request_interval_seconds=max(0.0, float(multi_cfg.get("request_interval_seconds", 12.0))),
                minimum_quality_score=max(1, min(10, int(multi_cfg.get("minimum_quality_score", 6)))),
                cache_path=str(multi_cfg.get("cache_path", "outputs/multi_segment_promo_generation/evaluation_cache.json")),
                checkpoint_path=str(multi_cfg.get("checkpoint_path", "outputs/multi_segment_promo_generation/checkpoint.json")),
            ),
            system_evaluation=SystemEvaluationSettings(
                human_review_sample_size=max(1, int(system_eval_cfg.get("human_review_sample_size", 30))),
                human_rating_scale=int(system_eval_cfg.get("human_rating_scale", 5)),
                ranking_cutoffs=[int(value) for value in system_eval_cfg.get("ranking_cutoffs", [5, 10, 20])],
                review_dataset_path=str(system_eval_cfg.get("review_dataset_path", "outputs/evaluation/human_review_dataset.csv")),
                human_reviews_path=str(system_eval_cfg.get("human_reviews_path", "outputs/evaluation/human_reviews.json")),
                human_reviews_csv_path=str(system_eval_cfg.get("human_reviews_csv_path", "outputs/evaluation/human_reviews.csv")),
                source_text_limit=max(200, int(system_eval_cfg.get("source_text_limit", 1800))),
            ),
        )


_SETTINGS = Settings.from_env()


def get_settings() -> Settings:
    return _SETTINGS


__all__ = [
    "LLMSettings",
    "AnalysisSettings",
    "CandidateGenerationSettings",
    "EvaluationSettings",
    "RankingSettings",
    "QualitySettings",
    "MultiSegmentSettings",
    "SystemEvaluationSettings",
    "Settings",
    "get_settings",
]
