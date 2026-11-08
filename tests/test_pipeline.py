import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.config import get_settings
from src.analysis.scene_analyzer import SceneAnalyzer
from src.generation.candidate_evaluator import CandidateEvaluator
from src.generation.candidate_filter import score_candidate, select_top_candidates
from src.generation.candidate_generator import CandidateGenerator
from src.llm.factory import create_provider
from src.llm.providers.openai_compatible_provider import OpenAICompatibleProvider
from src.schemas.scene_analysis import SceneAnalysis
from src.utils.cache import SimpleCache
from src.utils.retry import retry_delay_seconds
from src.utils.timestamp import (
    duration_seconds,
    parse_timestamp,
    seconds_to_timestamp,
    validate_duration_window,
)


@pytest.fixture
def sample_episode():
    data = json.loads(Path("outputs/episodes/parsed_episodes.json").read_text(encoding="utf-8"))
    return data[0]


def test_timestamp_conversion_round_trip():
    assert seconds_to_timestamp(0) == "00:00:00"
    assert seconds_to_timestamp(90) == "00:01:30"
    assert parse_timestamp("00:01:30") == 90.0
    assert parse_timestamp("1:30") == 90.0


def test_duration_validation_and_invalid_timestamps():
    assert validate_duration_window("00:00:20", "00:00:40") is True
    assert validate_duration_window("00:00:40", "00:00:20") is False
    assert parse_timestamp("bad-time") is None
    assert parse_timestamp(None) is None


def test_scene_analysis_schema_validation():
    payload = {
        "episode_id": 1,
        "scene_id": 5,
        "event_type": "betrayal",
        "dramatic_summary": "A betrayal is exposed.",
        "key_events": ["A secret is revealed."],
        "emotional_intensity": 8,
        "suspense_level": 7,
        "hook_potential": 9,
        "cliffhanger_potential": 8,
        "standalone_potential": 7,
        "key_dialogue": [{"text": "You lied to me.", "timestamp": "00:01:25", "character": "Sarah"}],
        "important_actions": [{"description": "Sarah confronts John.", "timestamp": "00:01:30"}],
        "unresolved_questions": ["What did John hide?"],
        "context_dependencies": ["The couple's history"],
        "promo_relevance": "high",
    }
    scene = SceneAnalysis.model_validate(payload)
    assert scene.episode_id == 1
    assert 1 <= scene.emotional_intensity <= 10


def test_candidate_generation_from_episode(sample_episode):
    generator = CandidateGenerator(
        min_duration=20,
        max_duration=60,
        preferred_methods=["event_anchored"],
    )
    candidates = generator.generate_for_episode(sample_episode)
    assert isinstance(candidates, list)
    assert len(candidates) >= 1
    first = candidates[0]
    assert first["total_duration"] >= 20
    assert first["total_duration"] <= 60


def test_candidate_filter_ranks_and_caps_api_evaluations():
    high_analysis = [{
        "episode_id": 1,
        "scene_id": 1,
        "hook_potential": 9,
        "cliffhanger_potential": 9,
        "emotional_intensity": 9,
        "suspense_level": 9,
        "standalone_potential": 8,
        "promo_relevance": "high",
    }]
    low_analysis = [{
        "episode_id": 1,
        "scene_id": 2,
        "hook_potential": 2,
        "cliffhanger_potential": 2,
        "emotional_intensity": 3,
        "suspense_level": 2,
        "standalone_potential": 3,
        "promo_relevance": "low",
    }]
    high_candidate = {"segments": [{"episode_id": 1, "scene_ids": [1]}]}
    low_candidate = {"segments": [{"episode_id": 1, "scene_ids": [2]}]}
    work = [
        ({"episode_id": 1}, low_candidate, low_analysis),
        ({"episode_id": 1}, high_candidate, high_analysis),
    ]

    assert score_candidate(high_candidate, high_analysis) > score_candidate(low_candidate, low_analysis)
    assert select_top_candidates(work, 1) == [work[1]]


def test_scene_prompt_formats_and_limits_long_scene_text():
    analyzer = SceneAnalyzer(create_provider("mock"))
    scene = {
        "episode_id": 1,
        "scene_id": 2,
        "heading": "INT. HALL",
        "start_time": 0,
        "end_time": 500,
        "characters": ["A"],
        "dialogue": [{"character": "A", "start_time": 0, "text": "Opening line. " * 1000}],
        "actions": [{"start_time": 499, "text": "Final action."}],
    }

    prompt = analyzer.build_prompt(scene)
    scene_text = analyzer._scene_to_text(scene)
    assert "Episode 1, scene 2" in prompt
    assert "Final action." in prompt
    assert len(scene_text) <= 4000
    assert "middle omitted" in scene_text


def test_scene_analyzer_retries_rate_limited_call(monkeypatch):
    class RateLimitedOnceProvider:
        provider_name = "fixed-test"
        model = "test-model"
        calls = 0

        def generate_json(self, prompt, response_schema=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("429 rate_limit_exceeded; try again in 0.25s")
            return {}

        @staticmethod
        def parse_json_response(response):
            return response

    provider = RateLimitedOnceProvider()
    analyzer = SceneAnalyzer(provider)
    analyzer.settings.llm.max_retries = 2
    analyzer.settings.analysis.request_interval_seconds = 0
    sleeps = []
    monkeypatch.setattr("src.analysis.scene_analyzer.time.sleep", sleeps.append)

    result = analyzer.analyze_scene({"episode_id": 1, "scene_id": 1, "dialogue": [], "actions": []})

    assert provider.calls == 2
    assert sleeps == [0.25]
    assert result["episode_id"] == 1


def test_scene_analyzer_paces_provider_requests(monkeypatch):
    class FixedProvider:
        provider_name = "fixed-test"
        model = "test-model"

        def __init__(self):
            self.schemas = []

        def generate_json(self, prompt, response_schema=None):
            self.schemas.append(response_schema)
            return {}

        @staticmethod
        def parse_json_response(response):
            return response

    provider = FixedProvider()
    analyzer = SceneAnalyzer(provider)
    analyzer.settings.analysis.request_interval_seconds = 12
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr("src.analysis.scene_analyzer.time.monotonic", lambda: clock[0])

    def fake_sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr("src.analysis.scene_analyzer.time.sleep", fake_sleep)

    analyzer.analyze_scene({"episode_id": 1, "scene_id": 1, "dialogue": [], "actions": []})
    clock[0] = 3.0
    analyzer.analyze_scene({"episode_id": 1, "scene_id": 2, "dialogue": [], "actions": []})

    assert sleeps == [9.0]
    assert provider.schemas == [None, None]


def test_retry_delay_uses_provider_rate_limit_estimate():
    assert retry_delay_seconds(RuntimeError("try again in 17.9s"), 0, 1.0) == 17.9
    assert retry_delay_seconds(
        RuntimeError("Please try again in 2m38.112s"), 0, 1.0, maximum_seconds=300.0
    ) == 158.112


def test_candidate_evaluator_retries_rate_limited_request(monkeypatch):
    class RateLimitedThenSuccessProvider:
        provider_name = "test"

        def __init__(self):
            self.calls = 0

        def generate_json(self, prompt, response_schema=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("429 rate_limit_exceeded; try again in 0.25s")
            return {"overall_summary": "Recovered after retry."}

        @staticmethod
        def parse_json_response(response):
            return response

    sleeps = []
    monkeypatch.setattr("src.generation.candidate_evaluator.time.sleep", sleeps.append)
    provider = RateLimitedThenSuccessProvider()
    evaluator = CandidateEvaluator(provider, max_retries=3, retry_backoff_seconds=1.0)
    candidate = {
        "segments": [{"episode_id": 1, "scene_ids": [1], "start_time": "00:00:10", "end_time": "00:00:30"}],
        "total_duration": 20,
    }

    result = evaluator.evaluate({"episode_id": 1, "scenes": []}, candidate, [])

    assert provider.calls == 2
    assert sleeps == [0.25]
    assert result["overall_summary"] == "Recovered after retry."


def test_promo_prompt_includes_only_events_inside_clip_window():
    evaluator = CandidateEvaluator(create_provider("mock"))
    episode = {
        "episode_id": 1,
        "scenes": [{
            "scene_id": 1,
            "dialogue": [
                {"character": "A", "start_time": 5, "end_time": 8, "text": "outside start"},
                {"character": "B", "start_time": 15, "end_time": 18, "text": "inside clip"},
                {"character": "C", "start_time": 35, "end_time": 39, "text": "outside end"},
            ],
            "actions": [],
        }],
    }
    candidate = {
        "segments": [{"episode_id": 1, "scene_ids": [1], "start_time": "00:00:10", "end_time": "00:00:30"}],
        "total_duration": 20,
    }

    prompt = evaluator.build_prompt(episode, candidate, [])
    assert "inside clip" in prompt
    assert "outside start" not in prompt
    assert "outside end" not in prompt


def test_candidate_timestamps_come_from_python_boundaries():
    class FixedProvider:
        provider_name = "fixed-test"

        def generate_json(self, prompt, response_schema=None):
            return {
                "hook": {"description": "Opening hook", "evidence": "Opening evidence", "timestamp": "99:99:99"},
                "cliffhanger": {"description": "Ending hook", "evidence": "Ending evidence", "timestamp": "99:99:99"},
            }

        @staticmethod
        def parse_json_response(response):
            return response

    candidate = {
        "segments": [{"episode_id": 1, "scene_ids": [1], "start_time": "00:00:10", "end_time": "00:00:30"}],
        "total_duration": 20,
    }
    result = CandidateEvaluator(FixedProvider()).evaluate({"episode_id": 1, "scenes": []}, candidate, [])

    assert result["hook"]["timestamp"] == "00:00:10"
    assert result["cliffhanger"]["timestamp"] == "00:00:30"


def test_duplicate_candidates_removed():
    generated = [
        {"segments": [{"episode_id": 1, "scene_ids": [1], "start_time": "00:00:20", "end_time": "00:00:40"}], "total_duration": 20},
        {"segments": [{"episode_id": 1, "scene_ids": [1], "start_time": "00:00:20", "end_time": "00:00:40"}], "total_duration": 20},
        {"segments": [{"episode_id": 1, "scene_ids": [1], "start_time": "00:00:25", "end_time": "00:00:45"}], "total_duration": 20},
    ]
    seen = {}
    unique = []
    for item in generated:
        key = (item["segments"][0]["start_time"], item["segments"][0]["end_time"], item["segments"][0]["episode_id"])
        if key not in seen:
            unique.append(item)
            seen[key] = True
    assert len(unique) == 2


def test_llm_response_parsing():
    raw = "```json\n{\"answer\": \"ok\", \"score\": 8}\n```"
    provider = create_provider("mock")
    parsed = provider.parse_json_response(raw)
    assert parsed["answer"] == "ok"
    assert parsed["score"] == 8


def test_provider_switching():
    mock_provider = create_provider("mock")
    grok_provider = create_provider("openai_compatible")
    assert mock_provider.__class__.__name__.endswith("MockLLMProvider")
    assert grok_provider.__class__.__name__.endswith("OpenAICompatibleProvider")
    assert get_settings().llm.provider in {"mock", "openai_compatible"}


def test_openai_compatible_provider_does_not_retry_rate_limit():
    class APIError(Exception):
        status_code = 429

    class Completions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise APIError("Rate limit exceeded")

    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    provider.model = "test-model"
    provider.temperature = 0

    with pytest.raises(RuntimeError, match="Rate limit exceeded"):
        provider.generate_json("prompt", response_schema={"type": "object"})
    assert provider.client.chat.completions.calls == 1


def test_openai_compatible_provider_falls_back_only_for_unsupported_json_mode():
    class APIError(Exception):
        status_code = 400

    class Completions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if "response_format" in kwargs:
                raise APIError("response_format json_object is unsupported")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))])

    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    provider.model = "test-model"
    provider.temperature = 0

    assert provider.generate_json("prompt", response_schema={"type": "object"}) == {"ok": True}
    assert provider.client.chat.completions.calls == 2


def test_openai_compatible_provider_falls_back_for_json_validation_400():
    class APIError(Exception):
        status_code = 400

    class Completions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if "response_format" in kwargs:
                raise APIError("Failed to validate JSON. Please adjust your prompt.")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))])

    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    provider.model = "test-model"
    provider.temperature = 0

    assert provider.generate_json("prompt", response_schema={"type": "object"}) == {"ok": True}
    assert provider.client.chat.completions.calls == 2


def test_cache_functionality():
    cache = SimpleCache()
    cache.set("demo", {"value": 1})
    assert cache.get("demo") == {"value": 1}
    assert cache.get("missing") is None


def test_duration_calculation_with_float_seconds():
    assert duration_seconds("00:00:10", "00:00:40") == 30.0
    assert duration_seconds("00:01:30", "00:02:00") == 30.0
