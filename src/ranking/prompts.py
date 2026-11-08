"""Prompt template for Module 4 missing-score fallback assessment."""

MISSING_SCORE_TEMPLATE = """
Score the promo criteria listed below using only the supplied candidate metadata and screenplay evidence. Be critical, do not invent facts or timestamps, and give one short evidence-based explanation per requested score. Scores must be integers 1-10. Return all six named criteria as objects with score and evidence; for any criterion not requested, use score 1 and evidence 'not requested'.

Requested missing criteria: {missing_criteria}
Candidate: episode {episode_id}, scenes {scene_ids}, {start_time}-{end_time}, {duration_seconds} seconds
Existing summary: {summary}
Evidence:
{source_text}

Criteria names: hook_strength, cliffhanger_strength, standalone_understandability, emotional_intensity, narrative_coherence, promo_engagement_potential.
Return a JSON object with these six keys.
"""
