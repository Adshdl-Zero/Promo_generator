"""Prompt templates for Module 6 multi-segment narrative evaluation."""

MULTI_SEGMENT_EVALUATION_TEMPLATE = """
Evaluate the following ordered promo as ONE short-form story for viewers unfamiliar with the series.
The order, episodes, scenes, boundaries, and source evidence are fixed. Never invent or change timestamps, dialogue, actions, or events. Be critical; use the actual screenplay excerpts below, not only summaries.

Ordered segments and source text:
{segments_text}

Assess the sequence as a whole: first-three-second hook, understandable context, transition quality, emotional progression, narrative coherence, closing cliffhanger, standalone clarity, and promotional engagement. Return JSON with these exact integer scores from 1 to 10:
hook_strength, narrative_coherence, context_completeness, transition_quality, emotional_impact, cliffhanger_strength, standalone_understandability, overall_promotional_potential.
Also return strengths (array of concise strings), weaknesses (array), and explanation (short evidence-based paragraph). Output JSON only; do not omit fields. Evidence must be supported by the excerpts.
"""
