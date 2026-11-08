"""Offline coverage for Module 6 generation and Module 8 human evaluation."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

from src.config import get_settings
from src.evaluation.system_evaluation import (
    build_review_dataset,
    compute_system_evaluation,
    load_human_reviews,
    save_human_review,
    write_review_dataset,
)
from src.generation.multi_segment_generator import generate_multi_segment_promos
from src.llm.providers.mock_provider import MockLLMProvider
from src.quality.validator import validate_candidate
from src.ranking.ranker import rank_candidates


def make_episode(episode_id: int = 1, start_offset: float = 0.0) -> dict:
    events = []
    for index, start in enumerate((0, 5, 30, 35, 60, 65), start=1):
        shifted = start + start_offset
        events.append({
            "character": "HERO",
            "start_time": shifted,
            "end_time": shifted + 5,
            "text": f"Dramatic source line {index} episode {episode_id}.",
        })
    return {
        "episode_id": episode_id,
        "scenes": [{
            "scene_id": 1,
            "heading": "INT. ROOM - DAY",
            "start_time": start_offset,
            "end_time": start_offset + 100,
            "characters": ["HERO"],
            "dialogue": events,
            "actions": [],
        }],
    }


def make_analysis(episode_id: int = 1) -> dict:
    return {
        "episode_id": episode_id,
        "scene_id": 1,
        "event_type": "betrayal",
        "dramatic_summary": "The hero discovers a betrayal and confronts an ally.",
        "key_events": ["A betrayal is revealed."],
        "emotional_intensity": 8,
        "suspense_level": 8,
        "hook_potential": 9,
        "cliffhanger_potential": 9,
        "standalone_potential": 7,
        "promo_relevance": "high",
        "context_dependencies": [],
    }


def test_module6_generates_valid_same_episode_multi_segment_promo(tmp_path):
    settings = deepcopy(get_settings())
    settings.multi_segment.minimum_duration = 20
    settings.multi_segment.maximum_duration = 60
    settings.multi_segment.minimum_segment_duration = 6
    settings.multi_segment.maximum_segment_duration = 12
    settings.multi_segment.min_segments = 2
    settings.multi_segment.max_segments = 3
    settings.multi_segment.max_moments_per_episode = 10
    settings.multi_segment.max_combinations = 20
    settings.multi_segment.max_llm_calls = 2
    settings.multi_segment.request_interval_seconds = 0
    episode = make_episode()

    candidates, metadata = generate_multi_segment_promos(
        [episode], [make_analysis()], [], settings=settings,
        provider=MockLLMProvider(model="mock"),
        cache_path=tmp_path / "cache.json", checkpoint_path=tmp_path / "checkpoint.json",
    )

    assert candidates
    candidate = candidates[0]
    assert candidate["generation_strategy"] == "same_episode"
    assert 20 <= candidate["total_duration"] <= 60
    assert candidate["segment_count"] == len(candidate["segments"])
    assert [segment["segment_order"] for segment in candidate["segments"]] == list(range(1, candidate["segment_count"] + 1))
    assert all(segment["start_seconds"] < segment["end_seconds"] for segment in candidate["segments"])
    assert metadata["llm_requests"] > 0


def test_module6_generates_related_cross_episode_segments(tmp_path):
    settings = deepcopy(get_settings())
    settings.multi_segment.minimum_duration = 20
    settings.multi_segment.maximum_duration = 60
    settings.multi_segment.minimum_segment_duration = 6
    settings.multi_segment.maximum_segment_duration = 12
    settings.multi_segment.max_segments = 2
    settings.multi_segment.max_moments_per_episode = 6
    settings.multi_segment.max_combinations = 20
    settings.multi_segment.max_llm_calls = 3
    settings.multi_segment.request_interval_seconds = 0
    episodes = [make_episode(1), make_episode(2, 10)]
    analysis = [make_analysis(1), make_analysis(2)]

    candidates, _ = generate_multi_segment_promos(
        episodes, analysis, [], settings=settings,
        provider=MockLLMProvider(model="mock"),
        cache_path=tmp_path / "cache.json", checkpoint_path=tmp_path / "checkpoint.json",
    )

    assert any(item["generation_strategy"] == "cross_episode" for item in candidates)
    for candidate in candidates:
        assert len(candidate["episode_ids"]) >= 2
        assert 20 <= candidate["total_duration"] <= 60


def test_module6_stops_after_rate_limit_and_honors_request_cap(tmp_path):
    class RateLimitedProvider:
        provider_name = "rate-limited-test"
        model = "test-model"
        calls = 0

        def generate_json(self, prompt, response_schema=None):
            self.calls += 1
            raise RuntimeError("429 tokens per minute rate_limit_exceeded")

        @staticmethod
        def parse_json_response(response):
            return response

    settings = deepcopy(get_settings())
    settings.multi_segment.minimum_duration = 20
    settings.multi_segment.maximum_duration = 60
    settings.multi_segment.minimum_segment_duration = 6
    settings.multi_segment.maximum_segment_duration = 12
    settings.multi_segment.min_segments = 2
    settings.multi_segment.max_segments = 2
    settings.multi_segment.max_moments_per_episode = 10
    settings.multi_segment.max_combinations = 10
    settings.multi_segment.max_llm_calls = 1
    settings.multi_segment.request_interval_seconds = 0
    settings.llm.max_retries = 3
    provider = RateLimitedProvider()

    _, metadata = generate_multi_segment_promos(
        [make_episode()], [make_analysis()], [], settings=settings,
        provider=provider,
        cache_path=tmp_path / "rate-limit-cache.json",
        checkpoint_path=tmp_path / "rate-limit-checkpoint.json",
    )

    assert provider.calls == 1
    assert metadata["llm_requests"] == 1
    assert metadata["rate_limit_skips"] > 0


def test_multi_segment_candidate_validates_through_module5_and_module4():
    episodes = [make_episode()]
    candidate = {
        "promo_id": "MS0001",
        "promo_type": "multi_segment",
        "segments": [
            {"episode_id": 1, "scene_ids": [1], "start_time": "00:00:00", "end_time": "00:00:10", "start_seconds": 0, "end_seconds": 10, "segment_duration": 10, "segment_order": 1},
            {"episode_id": 1, "scene_ids": [1], "start_time": "00:00:30", "end_time": "00:00:40", "start_seconds": 30, "end_seconds": 40, "segment_duration": 10, "segment_order": 2},
        ],
        "total_duration": 20,
        "hook": {"description": "Opening source moment", "evidence": "Dramatic source line 1", "timestamp": "00:00:00"},
        "cliffhanger": {"description": "Ending source moment", "evidence": "Dramatic source line 4", "timestamp": "00:00:40"},
        "evaluation": {"hook_score": 8, "cliffhanger_score": 8, "standalone_context_score": 7, "emotional_intensity_score": 8, "narrative_coherence_score": 8, "promo_engagement_potential": 8},
        "overall_summary": "A connected two-moment sequence.",
        "generation_method": "multi_segment",
        "generation_strategy": "same_episode",
        "segment_count": 2,
        "episode_ids": [1],
        "combined_summary": "A connected two-moment sequence.",
        "narrative_coherence_score": 8,
        "standalone_understandability_score": 7,
        "overall_promotional_score": 8,
        "strengths": [],
        "weaknesses": [],
        "generation_explanation": "Source-supported segments share a character.",
        "validation_status": "valid",
    }
    validation = validate_candidate(candidate, {1: episodes[0]}, candidate, minimum_segment_duration=6)
    assert validation["technical_status"] == "PASS"

    settings = deepcopy(get_settings())
    settings.ranking.minimum_selection_score = 0
    rows = rank_candidates([candidate], episodes, [make_analysis()], settings=settings, provider=MockLLMProvider(model="mock"), cache_path=Path("unused-cache.json"))
    assert rows[0]["raw_score"] == 78.0
    assert rows[0]["promo_type"] == "multi_segment"


def test_module8_review_sheet_and_multi_reviewer_upsert(tmp_path):
    candidate = {
        "promo_id": "P1", "segments": [{"episode_id": 1, "scene_ids": [1], "start_time": "00:00:00", "end_time": "00:00:30"}],
        "total_duration": 30, "overall_summary": "A promo.", "evaluation": {"hook_score": 7},
    }
    episodes = [make_episode()]
    rows = build_review_dataset([candidate], [], [candidate], {"candidate_results": []}, episodes, sample_size=10)
    assert len(rows) == 1
    assert rows[0]["human_overall_label"] == ""
    assert "Dramatic source line" in rows[0]["source_text"]
    dataset_path = tmp_path / "review.csv"
    write_review_dataset(rows, dataset_path)
    with dataset_path.open(encoding="utf-8-sig", newline="") as handle:
        assert next(csv.DictReader(handle))["human_overall_label"] == ""

    json_path, csv_path = tmp_path / "human_reviews.json", tmp_path / "human_reviews.csv"
    review_a = {"promo_id": "P1", "reviewer_id": "anonymous", "reviewed_at": "2026-01-01T00:00:00Z", "overall_label": "Strong", "hook_rating": 5}
    review_b = {"promo_id": "P1", "reviewer_id": "reviewer-b", "reviewed_at": "2026-01-01T00:00:00Z", "overall_label": "Average", "hook_rating": 3}
    save_human_review(review_a, json_path, csv_path)
    save_human_review(review_b, json_path, csv_path)
    save_human_review({**review_a, "hook_rating": 4}, json_path, csv_path)
    saved = load_human_reviews(json_path)
    assert len(saved) == 2
    assert next(item for item in saved if item["reviewer_id"] == "anonymous")["hook_rating"] == 4


def test_module8_metrics_are_unavailable_without_human_reviews():
    candidate = {"promo_id": "P1", "segments": [{"episode_id": 1, "scene_ids": [1], "start_time": "00:00:00", "end_time": "00:00:30"}], "total_duration": 30, "rank": 1, "raw_score": 80}
    metrics, errors, rows = compute_system_evaluation([candidate], {"candidate_results": []}, [make_episode()], [])
    assert metrics["reviewed_promo_count"] == 0
    assert metrics["human_ai_agreement"]["spearman_rank_correlation"] is None
    assert metrics["precision_at_k"]["5"]["value"] is None
    assert rows[0]["human_label"] is None
    assert errors["warning"] is not None
