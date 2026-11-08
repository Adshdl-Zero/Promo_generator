"""Prompt templates for independent Module 5 quality evaluation."""

NARRATIVE_QUALITY_TEMPLATE = """
Independently and critically evaluate this promo. Do not reuse its ranking score as evidence and do not assume the previous evaluation is correct. Use ONLY the supplied screenplay text and metadata; do not invent facts or timestamps. A high score requires direct evidence. Report narrative weaknesses even if the promo appears strong.

Fixed promo duration: {duration_seconds} seconds.
Existing short promo summary (context only): {promo_summary}
Source screenplay evidence within the clip:
{source_text}

Return JSON only, with ALL of these exact root keys; do not omit any: opening_hook, ending_cliffhanger, standalone_understandability, narrative_coherence, emotional_engagement, failure_findings, overall_explanation.
Each of the five criterion keys must be an object with score (integer 1-10), explanation (short string), evidence (array of at most 3 concise source-supported strings), and weaknesses (array of at most 3 strings). Every evidence array element must be a string, never a nested array. failure_findings must be an array of objects with category, severity (low|medium|high), explanation, evidence (array of strings), suggested_improvement. overall_explanation must be a short string, even when there are no weaknesses.
Use this exact shape and keep every key:
{{"opening_hook":{{"score":1,"explanation":"...","evidence":[],"weaknesses":[]}},"ending_cliffhanger":{{"score":1,"explanation":"...","evidence":[],"weaknesses":[]}},"standalone_understandability":{{"score":1,"explanation":"...","evidence":[],"weaknesses":[]}},"narrative_coherence":{{"score":1,"explanation":"...","evidence":[],"weaknesses":[]}},"emotional_engagement":{{"score":1,"explanation":"...","evidence":[],"weaknesses":[]}},"failure_findings":[],"overall_explanation":"..."}}
Be critical about a slow first 3 seconds, weak ending, missing context, confusion, repetitive dialogue, weak emotion, or excluded payoff. Do not propose changed timestamps as if they were facts.
"""
