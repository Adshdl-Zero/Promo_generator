"""Reusable prompt templates used by the narrative and candidate analysis pipeline."""

SCENE_ANALYSIS_TEMPLATE = """
Analyze this screenplay scene for promo potential. Use only its content; do not invent facts or timestamps. Be concise, evidence-based, and conservative with uncertain scores.
Score intensity, suspense, hook, cliffhanger, and standalone potential from 1-10. Keep the summary to one short sentence, key_events to at most 2 short items, and each other list to at most 1 short item. Keep evidence quotes brief. Include only observed dialogue/actions; use empty arrays or null when absent.

Episode {episode_id}, scene {scene_id}; {heading}; {start_time}-{end_time}
Characters: {characters}
Prior summary (optional): {previous_summary}
Scene:
{scene_text}

Return ONLY one JSON object with: episode_id, scene_id, event_type, dramatic_summary, key_events[], emotional_intensity, suspense_level, hook_potential, cliffhanger_potential, standalone_potential, key_dialogue[{{text,timestamp,character}}], important_actions[{{description,timestamp}}], unresolved_questions[], context_dependencies[], promo_relevance(low|medium|high).
"""

PROMO_EVALUATION_TEMPLATE = """
Evaluate this promo clip as a coherent sequence. Its timestamps and boundaries are fixed; never change them. Use only the evidence below.
Episode {episode_id}; scenes {scene_ids}; {start_time}-{end_time} ({duration_seconds}s)
Relevant scene summaries: {scene_context}
Clip dialogue/actions:
{window_text}

Score hook, cliffhanger, standalone context, emotional intensity, and coherence from 1-10. Return ONLY JSON: hook_score, cliffhanger_score, standalone_context_score, emotional_intensity_score, narrative_coherence_score, hook{{description,evidence}}, cliffhanger{{description,evidence}}, overall_summary, strengths[], weaknesses[], missing_context.
"""
