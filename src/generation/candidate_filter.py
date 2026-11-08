"""Deterministically rank promo candidates before spending LLM evaluation calls."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

CandidateWorkItem = tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]


def score_candidate(candidate: dict[str, Any], scene_analyses: list[dict[str, Any]]) -> float:
    """Score a candidate from existing scene-analysis signals without an API call."""
    segment = candidate.get("segments", [{}])[0]
    scene_ids = {int(scene_id) for scene_id in segment.get("scene_ids", [])}
    episode_id = int(segment.get("episode_id", 0))
    relevant = [
        item
        for item in scene_analyses
        if int(item.get("episode_id", 0)) == episode_id
        and int(item.get("scene_id", 0)) in scene_ids
    ]
    if not relevant:
        return 0.0

    weights = {
        "hook_potential": 0.30,
        "cliffhanger_potential": 0.25,
        "emotional_intensity": 0.20,
        "suspense_level": 0.15,
        "standalone_potential": 0.10,
    }
    score = 0.0
    for field, weight in weights.items():
        values = [float(item.get(field, 0) or 0) for item in relevant]
        score += (sum(values) / len(values)) * weight

    relevance_bonus = {"high": 1.0, "medium": 0.5, "low": 0.0}
    score += sum(relevance_bonus.get(str(item.get("promo_relevance", "low")).lower(), 0.0) for item in relevant) / len(relevant)
    return score


def select_top_candidates(
    candidates: list[CandidateWorkItem],
    limit: int,
) -> list[CandidateWorkItem]:
    """Select the highest-ranked candidates with round-robin episode coverage."""
    if limit <= 0 or not candidates:
        return []

    by_episode: dict[int, list[tuple[float, int, CandidateWorkItem]]] = defaultdict(list)
    for position, item in enumerate(candidates):
        episode, candidate, analyses = item
        episode_id = int(episode.get("episode_id", 0))
        by_episode[episode_id].append((score_candidate(candidate, analyses), position, item))

    for items in by_episode.values():
        items.sort(key=lambda ranked: (-ranked[0], ranked[1]))

    episode_order = sorted(
        by_episode,
        key=lambda episode_id: (-by_episode[episode_id][0][0], episode_id),
    )
    selected: list[CandidateWorkItem] = []
    offsets = {episode_id: 0 for episode_id in episode_order}
    while len(selected) < limit:
        added = False
        for episode_id in episode_order:
            offset = offsets[episode_id]
            if offset < len(by_episode[episode_id]):
                selected.append(by_episode[episode_id][offset][2])
                offsets[episode_id] += 1
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
    return selected


__all__ = ["CandidateWorkItem", "score_candidate", "select_top_candidates"]
