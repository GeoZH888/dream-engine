"""Narrator prompts: the prefrontal critic switched off (dlPFC hypoactivity in REM)."""

from __future__ import annotations

from typing import Any

from dream_engine.seed import DreamSeed

SYSTEM_CRITIC_OFF = """You are the narrating cortex of a sleeping person, in a research simulation \
of how dreams are built from memory. You turn the seed you are given into a dream report.

Rules:
1. Write in the first person, present tense, as a dream report told immediately on waking.
2. Inside the dream the dreamer never questions impossibilities: nothing is noticed as \
strange, nothing is checked, nothing is explained.
3. Do not explain symbolism. Do not moralise. Do not resolve the plot; stop where the dream stops.
4. Use only the provided memory fragments and distortions. Do not invent unrelated themes, \
people or places. Small connective sensory detail is allowed.
5. Emotion is strong, but its cause can be unclear.
6. Follow the stage style exactly. NREM N3: fragments only, almost no narrative. NREM N2: \
thought-like and mundane, close to waking thought. REM: sensory detail, especially visual.

Never mention fragment ids, stages, operators or this simulation in the report. Also rate \
the report's overall emotional tone: valence from -1 (negative) to 1 (positive), arousal \
from 0 (calm) to 1 (intense)."""

LUCID_DIRECTIVE = """Partial lucidity (the critic briefly switches back on): partway through, \
the dreamer notices exactly one of the anomalies and realises this is a dream. They gain \
limited control: they try to change one thing, and it only partly works before the dream \
pulls them back in. The rest of the rules still hold."""

# Nominal stage temperature -> prompt guidance, used when the model rejects sampling params.
LOOSENESS = [(0.5, "literal and low-variance"), (0.7, "plain and predictable"),
             (0.95, "loose and associative"), (9.9, "very loose: surprising associations, fluid logic")]

NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "narrative": {"type": "string"},
        "emotional_tone": {
            "type": "object",
            "properties": {"valence": {"type": "number"}, "arousal": {"type": "number"}},
            "required": ["valence", "arousal"],
            "additionalProperties": False,
        },
    },
    "required": ["narrative", "emotional_tone"],
    "additionalProperties": False,
}

ROLE_LABEL = {"replay": "replayed", "augment": "noisy recall", "pgo_injection": "intrusion"}


def looseness(temperature: float) -> str:
    return next(text for limit, text in LOOSENESS if temperature <= limit)


def _age(seed: DreamSeed, f) -> str:
    newest = max(x.timestamp for x in seed.fragments)
    days = (newest.date() - f.timestamp.date()).days
    return "today" if days == 0 else "yesterday" if days == 1 else f"{days} days ago"


def build_user_prompt(seed: DreamSeed, profile: dict[str, Any], send_temperature: bool) -> str:
    lo, hi = profile["words"]
    lines = [f"Stage: {seed.stage} (sleep cycle {seed.cycle}).",
             f"Style: {profile['style']}.",
             f"Length: {lo}-{hi} words."]
    if not send_temperature:
        lines.append(f"Associative looseness: {looseness(profile['temperature'])}.")

    lines.append("\nMemory fragments (the only material you may use):")
    injected = {e.fragment_id for e in seed.pgo_events if e.fragment_id}
    for f in seed.fragments:
        if f.id in injected:
            continue  # introduced at its burst below
        lines.append(f"- [{f.id}] ({ROLE_LABEL[f.role]}, {_age(seed, f)}; valence {f.valence:+.1f}, "
                     f"arousal {f.arousal:.1f}) {f.text}")

    if seed.operators:
        lines.append("\nDistortions to enact, without remarking on them:")
        lines += [f"- {op.directive}" for op in seed.operators]

    if seed.pgo_events:
        by_id = {f.id: f for f in seed.fragments}
        lines.append("\nActivation bursts, in order (place each at roughly its position in the report):")
        for e in seed.pgo_events:
            where = f"~{int(e.position * 100)}% through"
            if e.kind == "fragment_injection":
                f = by_id[e.fragment_id]
                lines.append(f"- {where}: this memory suddenly intrudes: [{f.id}] {f.text}")
            else:
                lines.append(f"- {where}: abrupt scene change, no transition")

    if seed.lucid:
        lines.append("\n" + LUCID_DIRECTIVE)
    return "\n".join(lines)
