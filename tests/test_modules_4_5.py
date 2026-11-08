"""Offline unit and integration coverage for Modules 4 and 5."""

from __future__ import annotations

import csv
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from src.config import get_settings
from src.quality.evaluator import _normalize_failure_finding, _normalize_narrative_payload, evaluate_ranked_promos, write_quality_outputs
from src.quality.prompts import NARRATIVE_QUALITY_TEMPLATE
from src.quality.validator import validate_candidate, validate_unique_ids
from src.ranking.ranker import rank_candidates, weighted_score
from src.ranking.reports import write_ranking_outputs
from src.utils.retry import RequestPacer


class FixedProvider:
    provider_name = "fixed-test"
    model = "fixed-model"
    temperature = 0

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate_json(self, prompt, response_schema=None):
        self.calls += 1
        return self.response

    @staticmethod
    def parse_json_response(response):
        return response


def make_candidate(promo_id="P0001", episode_id=1, scene_id=1, start=0, end=30, score=8):
    return {
        "promo_id": promo_id,
        "segments": [{
            "episode_id": episode_id,
            "scene_ids": [scene_id],
            "start_time": f"00:00:{start:02d}",
            "end_time": f"00:00:{end:02d}",
            "start_seconds": start,
            "end_seconds": end,
        }],
        "total_duration": end - start,
        "hook": {"description": "Opening hook", "evidence": "An accusation", "timestamp": f"00:00:{start:02d}"},
        "cliffhanger": {"description": "Unresolved ending", "evidence": "A question remains", "timestamp": f"00:00:{end:02d}"},
        "evaluation": {
            "hook_score": score,
            "cliffhanger_score": score,
            "standalone_context_score": score,
            "emotional_intensity_score": score,
            "narrative_coherence_score": score,
        },
        "overall_summary": "A character confronts an accusation before an unresolved reveal.",
        "strengths": ["Clear confrontation"],
        "weaknesses": [],
        "generation_method": "event_anchored",
    }


def make_episode(episode_id=1, scene_id=1):
    return {
        "episode_id": episode_id,
        "scenes": [{
            "scene_id": scene_id,
            "start_time": 0,
            "end_time": 120,
            "dialogue": [{"character": "A", "start_time": 1, "end_time": 3, "text": "An accusation."}],
            "actions": [{"start_time": 25, "end_time": 27, "text": "A door slams."}],
        }],
    }


def make_narrative_response(score=8):
    criterion = {
        "score": score,
        "explanation": "The scene has clear, supported dramatic evidence.",
        "evidence": ["An accusation."],
        "weaknesses": [],
    }
    return {
        "opening_hook": criterion,
        "ending_cliffhanger": criterion,
        "standalone_understandability": criterion,
        "narrative_coherence": criterion,
        "emotional_engagement": criterion,
        "failure_findings": [],
        "overall_explanation": "Independent review found a clear mini-story.",
    }


def test_weighted_score_calculation_and_normalization():
    scores = {"a": 10, "b": 5}
    assert weighted_score(scores, {"a": 0.25, "b": 0.75}) == 62.5
    assert weighted_score({"a": 10, "missing": None}, {"a": 0.5, "missing": 0.5}) == 100
    assert weighted_score({"a": None}, {"a": 1}) == 0


def test_ranker_reuses_existing_scores_and_uses_llm_only_for_missing_score(tmp_path):
    candidate = make_candidate()
    settings = copy.deepcopy(get_settings())
    settings.ranking.max_missing_score_llm_calls = 1
    provider = FixedProvider({
        "hook_strength": {"score": 7, "evidence": "Existing hook."},
        "cliffhanger_strength": {"score": 7, "evidence": "Existing cliffhanger."},
        "standalone_understandability": {"score": 7, "evidence": "Understandable."},
        "emotional_intensity": {"score": 7, "evidence": "Emotional."},
        "narrative_coherence": {"score": 7, "evidence": "Coherent."},
        "promo_engagement_potential": {"score": 9, "evidence": "Strong unresolved question."},
    })
    rows = rank_candidates([candidate], [make_episode()], [], settings=settings, provider=provider, cache_path=tmp_path / "rank_cache.json")
    assert provider.calls == 1
    assert rows[0]["criterion_scores"]["promo_engagement_potential"] == 9
    assert rows[0]["raw_score"] == 80.5
    assert rows[0]["rank"] == 1


def test_ranker_makes_no_call_when_all_scores_exist(tmp_path):
    candidate = make_candidate()
    candidate["evaluation"]["promo_engagement_potential"] = 8
    provider = FixedProvider({})
    rows = rank_candidates([candidate], [make_episode()], [], provider=provider, cache_path=tmp_path / "cache.json")
    assert provider.calls == 0
    assert rows[0]["criterion_scores"]["promo_engagement_potential"] == 8


def test_ranker_skips_uncached_missing_score_calls_by_default(tmp_path):
    candidate = make_candidate()
    provider = FixedProvider({})
    settings = copy.deepcopy(get_settings())
    settings.ranking.max_missing_score_llm_calls = 0

    rows = rank_candidates(
        [candidate], [make_episode()], [], settings=settings, provider=provider,
        cache_path=tmp_path / "cache.json",
    )

    assert provider.calls == 0
    assert rows[0]["criterion_scores"]["promo_engagement_potential"] is None
    assert rows[0]["ranking_warnings"]


def test_ranker_caps_missing_score_api_calls_per_run(tmp_path):
    settings = get_settings()
    settings.ranking.max_missing_score_llm_calls = 1
    settings.ranking.request_interval_seconds = 0
    assessment = {
        name: {"score": 7, "evidence": "Supported by candidate evidence."}
        for name in (
            "hook_strength", "cliffhanger_strength", "standalone_understandability",
            "emotional_intensity", "narrative_coherence", "promo_engagement_potential",
        )
    }
    provider = FixedProvider(assessment)
    candidates = [make_candidate("P1"), make_candidate("P2", start=40, end=70)]

    rank_candidates(
        candidates, [make_episode()], [], settings=settings, provider=provider,
        cache_path=tmp_path / "cache.json",
    )

    assert provider.calls == 1


def test_request_pacer_enforces_minimum_interval(monkeypatch):
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr("src.utils.retry.time.monotonic", lambda: clock[0])

    def fake_sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr("src.utils.retry.time.sleep", fake_sleep)
    pacer = RequestPacer(12)
    pacer.wait()
    clock[0] += 3
    pacer.wait()

    assert sleeps == [9]


def test_ranking_category_overlap_episode_diversity_and_cap(tmp_path):
    settings = get_settings()
    settings.ranking.maximum_selected_promos = 2
    settings.ranking.maximum_promos_per_episode = 1
    settings.ranking.minimum_selection_score = 0
    candidates = [make_candidate("P1", 1, 1, 0, 30, 10), make_candidate("P2", 1, 1, 2, 32, 9), make_candidate("P3", 2, 1, 0, 30, 8)]
    for candidate in candidates:
        candidate["evaluation"]["promo_engagement_potential"] = 8
    rows = rank_candidates(candidates, [make_episode(1), make_episode(2)], [], settings=settings, provider=FixedProvider({}), cache_path=tmp_path / "cache.json")
    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert rows[0]["quality_category"] == "Excellent"
    assert rows[0]["duplicate_or_overlap_flags"]
    selected = [row for row in rows if row["selection_status"] == "selected"]
    assert len(selected) == 2
    assert len({row["segments"][0]["episode_id"] for row in selected}) == 2


def test_ranking_json_and_excel_csv_output(tmp_path):
    candidate = make_candidate()
    candidate["evaluation"]["promo_engagement_potential"] = 7
    rows = rank_candidates([candidate], [make_episode()], [], provider=FixedProvider({}), cache_path=tmp_path / "cache.json")
    json_path, csv_path = tmp_path / "ranked.json", tmp_path / "ranked.csv"
    write_ranking_outputs(rows, json_path, csv_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["promo_id"] == "P0001"
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        assert next(csv.DictReader(handle))["Promo ID"] == "P0001"


def test_technical_validation_accepts_good_candidate_and_rejects_bad_timestamp():
    episode = make_episode()
    candidate = make_candidate()
    result = validate_candidate(candidate, {1: episode}, candidate)
    assert result["technical_status"] == "PASS"

    bad = make_candidate(start=30, end=20)
    result = validate_candidate(bad, {1: episode})
    assert result["technical_status"] == "FAIL"
    assert any(issue["category"] == "invalid_timestamp_order" for issue in result["technical_errors"])


def test_technical_validation_flags_duration_references_and_scores():
    episode = make_episode()
    short = make_candidate(end=19)
    result = validate_candidate(short, {1: episode})
    assert any(issue["category"] == "segment_duration" for issue in result["technical_errors"])

    invalid = make_candidate(episode_id=9, scene_id=7)
    invalid["evaluation"]["hook_score"] = 11
    result = validate_candidate(invalid, {1: episode})
    categories = {issue["category"] for issue in result["technical_errors"]}
    assert "missing_episode" in categories
    assert "invalid_score" in categories

    long_candidate = make_candidate(end=61)
    result = validate_candidate(long_candidate, {1: episode})
    assert any(issue["category"] == "segment_duration" for issue in result["technical_errors"])


def test_technical_validation_duplicate_ids_and_missing_required_fields():
    duplicates = [make_candidate("SAME"), make_candidate("SAME", episode_id=2)]
    duplicate_errors = validate_unique_ids(duplicates)
    assert len(duplicate_errors) == 2

    malformed = {"promo_id": "MISSING", "segments": []}
    result = validate_candidate(malformed, {1: make_episode()})
    assert result["technical_status"] == "FAIL"
    assert any(issue["category"] == "missing_segments" for issue in result["technical_errors"])


def test_technical_validation_accepts_legacy_list_evidence_with_warning():
    candidate = make_candidate()
    candidate["hook"]["evidence"] = ["First source line", "Second source line"]
    candidate["cliffhanger"]["evidence"] = ["Final source line"]
    result = validate_candidate(candidate, {1: make_episode()})

    assert result["technical_status"] == "PASS_WITH_WARNINGS"
    assert not result["technical_errors"]
    assert sum(issue["category"] == "legacy_evidence_list" for issue in result["technical_warnings"]) == 2
    assert isinstance(candidate["hook"]["evidence"], list)


def test_module5_independent_evaluation_failures_and_high_rank_low_quality(tmp_path):
    candidate = make_candidate()
    candidate["rank"] = 1
    candidate["raw_score"] = 95
    candidate["selection_status"] = "selected"
    provider = FixedProvider(make_narrative_response(score=3))
    report, failures, records = evaluate_ranked_promos(
        [candidate], [candidate], [make_episode()], provider=provider,
        cache_path=tmp_path / "quality-cache.json",
    )
    assert provider.calls == 1
    assert report["candidates_evaluated"] == 1
    assert records[0]["quality_decision"] == "PASS_WITH_WARNINGS"
    assert records[0]["high_rank_low_quality"] is True
    assert "high_rank_low_quality" in {item["category"] for item in records[0]["failure_labels"]}
    assert failures["high_ranked_poor_quality_examples"] == ["P0001"]


def test_module5_outputs_and_empty_dataset_are_valid(tmp_path):
    report, failures, records = evaluate_ranked_promos([], [], [], provider=FixedProvider({}), cache_path=tmp_path / "cache.json")
    json_path = tmp_path / "quality.json"
    csv_path = tmp_path / "quality.csv"
    failure_path = tmp_path / "failures.json"
    write_quality_outputs(report, failures, records, json_path, csv_path, failure_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["candidates_evaluated"] == 0
    assert json.loads(failure_path.read_text(encoding="utf-8"))["candidates_processed"] == 0
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_narrative_evaluation_cache_prevents_repeat_provider_calls(tmp_path):
    candidate = make_candidate()
    candidate["selection_status"] = "selected"
    candidate["rank"] = 1
    provider = FixedProvider(make_narrative_response())
    cache = tmp_path / "quality-cache.json"
    evaluate_ranked_promos([candidate], [candidate], [make_episode()], provider=provider, cache_path=cache)
    evaluate_ranked_promos([candidate], [candidate], [make_episode()], provider=provider, cache_path=cache)
    assert provider.calls == 1


def test_module5_no_cache_forces_fresh_provider_request(tmp_path):
    candidate = make_candidate()
    candidate["selection_status"] = "selected"
    candidate["rank"] = 1
    provider = FixedProvider(make_narrative_response())
    cache_path = tmp_path / "quality-cache.json"
    args = ([candidate], [candidate], [make_episode()])

    evaluate_ranked_promos(*args, provider=provider, cache_path=cache_path)
    cached_bytes = cache_path.read_bytes()
    evaluate_ranked_promos(*args, provider=provider, cache_path=cache_path, use_cache=False)

    assert provider.calls == 2
    assert cache_path.read_bytes() == cached_bytes


def test_module5_stops_narrative_calls_after_rate_limit(tmp_path):
    class RateLimitedProvider(FixedProvider):
        def generate_json(self, prompt, response_schema=None):
            self.calls += 1
            raise RuntimeError("HTTP 429 rate limit exceeded")

    settings = copy.deepcopy(get_settings())
    settings.llm.max_retries = 3
    settings.quality.maximum_narrative_evaluations = 3
    settings.quality.concurrency = 1
    candidates = []
    for index in range(3):
        candidate = make_candidate(f"P{index + 1:04d}", start=0, end=20, score=8)
        candidate["selection_status"] = "selected"
        candidate["rank"] = index + 1
        candidate["raw_score"] = 80
        candidates.append(candidate)
    provider = RateLimitedProvider({})

    report, _, rows = evaluate_ranked_promos(
        candidates,
        candidates,
        [make_episode()],
        settings=settings,
        provider=provider,
        cache_path=tmp_path / "rate-limit-cache.json",
    )
    assert provider.calls == 1
    assert report["narrative_candidates_evaluated"] == 0
    assert rows[0]["narrative_evaluation_status"] == "unavailable"
    assert all(row["narrative_evaluation_status"] in {"unavailable", "skipped"} for row in rows)
    assert all("narrative_evaluation_unavailable" not in {label["category"] for label in row["failure_labels"]} for row in rows)


def test_module5_marks_selected_candidates_skipped_from_narrative_review(tmp_path):
    settings = copy.deepcopy(get_settings())
    settings.quality.maximum_narrative_evaluations = 0
    candidate = make_candidate()
    candidate["selection_status"] = "selected"
    candidate["rank"] = 1
    provider = FixedProvider({})
    report, _, rows = evaluate_ranked_promos(
        [candidate], [candidate], [make_episode()], settings=settings,
        provider=provider, cache_path=tmp_path / "cache.json",
    )
    labels = {finding["category"] for finding in rows[0]["failure_labels"]}
    assert provider.calls == 0
    assert rows[0]["narrative_evaluation_status"] == "skipped"
    assert "narrative_evaluation_skipped" not in labels
    assert rows[0]["quality_decision"] == "PASS_WITH_WARNINGS"


def test_module5_does_not_repeat_request_after_malformed_response_by_default(tmp_path):
    settings = copy.deepcopy(get_settings())
    settings.quality.maximum_narrative_evaluations = 1
    settings.quality.max_narrative_attempts = 1
    candidate = make_candidate()
    candidate["selection_status"] = "selected"
    candidate["rank"] = 1
    provider = FixedProvider({})

    _, _, rows = evaluate_ranked_promos(
        [candidate], [candidate], [make_episode()], settings=settings,
        provider=provider, cache_path=tmp_path / "malformed-cache.json",
    )

    assert provider.calls == 1
    assert rows[0]["narrative_evaluation_status"] == "unavailable"
    assert "schema" in rows[0]["narrative_evaluation_message"].lower()


def test_module5_narrative_prompt_requires_all_schema_fields():
    prompt = NARRATIVE_QUALITY_TEMPLATE.format(
        episode_id=1,
        scene_ids=[1],
        start_time="00:00:00",
        end_time="00:00:30",
        duration_seconds=30,
        promo_summary="A confrontation escalates.",
        source_text="00:00:01 DIALOGUE A: Stop!",
    )
    for field in (
        "opening_hook", "ending_cliffhanger", "standalone_understandability",
        "narrative_coherence", "emotional_engagement", "failure_findings", "overall_explanation",
    ):
        assert f'"{field}"' in prompt


def test_module5_flattens_nested_evidence_without_losing_text():
    payload = {
        "opening_hook": {
            "score": 7,
            "explanation": "Supported hook.",
            "evidence": ["First line", ["Second line"]],
            "weaknesses": [],
        }
    }
    normalized = _normalize_narrative_payload(payload)
    assert normalized["opening_hook"]["evidence"] == ["First line", "Second line"]
    assert payload["opening_hook"]["evidence"] == ["First line", ["Second line"]]


def test_module5_derives_missing_overall_summary_from_model_criteria():
    payload = {
        "opening_hook": {"explanation": "The opening has a direct accusation."},
        "ending_cliffhanger": {"explanation": "The promise remains unresolved."},
    }
    normalized = _normalize_narrative_payload(payload)
    assert normalized["overall_explanation_derived"] is True
    assert "direct accusation" in normalized["overall_explanation"]
    assert "promise remains unresolved" in normalized["overall_explanation"]


def test_module5_normalizes_equivalent_failure_category_labels():
    assert _normalize_failure_finding({"category": "Missing Context"})["category"] == "missing_context"
    assert _normalize_failure_finding({"category": "Abrupt Ending"})["category"] == "weak_cliffhanger"
    assert _normalize_failure_finding({"category": "Limited Hook"})["category"] == "weak_opening_hook"
