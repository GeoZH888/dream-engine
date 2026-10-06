"""Amygdala salience: valence/arousal tags for memory fragments.

Valence is in [-1, 1] (negative .. positive), arousal in [0, 1] (calm .. intense).
The LLM extractor tags fragments directly; this module supplies the clamping rules
and a small lexicon-based tagger used by the offline heuristic extractor.
"""

from __future__ import annotations

import re

# word -> (valence, arousal). A compact affective lexicon in the spirit of ANEW/NRC-VAD.
LEXICON: dict[str, tuple[float, float]] = {
    # positive, high arousal
    "excited": (0.8, 0.9), "thrilled": (0.9, 0.9), "amazing": (0.8, 0.8), "laughed": (0.7, 0.7),
    "laughing": (0.7, 0.7), "love": (0.9, 0.7), "loved": (0.9, 0.7), "celebrated": (0.8, 0.8),
    "won": (0.8, 0.8), "dancing": (0.7, 0.8), "kissed": (0.8, 0.8), "proud": (0.7, 0.6),
    # positive, low arousal
    "happy": (0.8, 0.5), "beautiful": (0.7, 0.4), "calm": (0.5, 0.1), "peaceful": (0.6, 0.1),
    "relaxed": (0.6, 0.1), "warm": (0.5, 0.3), "nice": (0.5, 0.3), "good": (0.5, 0.3),
    "kind": (0.6, 0.3), "smiled": (0.6, 0.4), "enjoyed": (0.7, 0.4), "grateful": (0.7, 0.3),
    "cozy": (0.6, 0.2), "quiet": (0.2, 0.1), "gentle": (0.5, 0.2), "sweet": (0.6, 0.3),
    # negative, high arousal
    "afraid": (-0.7, 0.8), "scared": (-0.7, 0.8), "terrified": (-0.9, 1.0), "panic": (-0.8, 1.0),
    "angry": (-0.7, 0.9), "furious": (-0.9, 1.0), "argued": (-0.6, 0.8), "argument": (-0.6, 0.8),
    "shouted": (-0.5, 0.9), "anxious": (-0.6, 0.8), "stressed": (-0.6, 0.8), "nervous": (-0.4, 0.7),
    "late": (-0.4, 0.6), "missed": (-0.5, 0.6), "lost": (-0.6, 0.6), "deadline": (-0.4, 0.7),
    "crash": (-0.7, 0.9), "fell": (-0.5, 0.7), "fight": (-0.7, 0.9), "embarrassed": (-0.6, 0.7),
    "rejected": (-0.8, 0.7), "worried": (-0.5, 0.6),
    # negative, low arousal
    "sad": (-0.7, 0.3), "tired": (-0.4, 0.1), "lonely": (-0.7, 0.3), "bored": (-0.4, 0.1),
    "disappointed": (-0.6, 0.4), "grey": (-0.2, 0.1), "gray": (-0.2, 0.1), "rain": (-0.1, 0.2),
    "cold": (-0.3, 0.3), "sick": (-0.6, 0.3), "miss": (-0.5, 0.4), "homesick": (-0.6, 0.4),
}
NEGATORS = {"not", "never", "no", "didn't", "don't", "wasn't", "isn't", "hardly"}
INTENSIFIERS = {"very", "so", "really", "extremely", "incredibly", "totally"}
TOKEN_RE = re.compile(r"[a-zA-Z']+")


def clamp_valence(v: float) -> float:
    return max(-1.0, min(1.0, float(v)))


def clamp_arousal(a: float) -> float:
    return max(0.0, min(1.0, float(a)))


def salience(valence: float, arousal: float) -> float:
    """Emotional salience in [0, 1]: how strongly the amygdala tags a memory."""
    return clamp_arousal(0.5 * (abs(clamp_valence(valence)) + clamp_arousal(arousal)))


def tag_text(text: str, default_arousal: float, intensifier_boost: float, negation_window: int) -> tuple[float, float]:
    """Lexicon tagger: mean valence of affect words (negation flips), max arousal."""
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    valences, arousals = [], []
    negate_left = 0
    for tok in tokens:
        if tok in NEGATORS:
            negate_left = negation_window
            continue
        if tok in LEXICON:
            v, a = LEXICON[tok]
            valences.append(-v if negate_left else v)
            arousals.append(a)
        negate_left = max(0, negate_left - 1)

    valence = sum(valences) / len(valences) if valences else 0.0
    arousal = max(arousals) if arousals else default_arousal
    boosts = sum(t in INTENSIFIERS for t in tokens) + text.count("!")
    arousal += intensifier_boost * min(boosts, 3)
    return clamp_valence(valence), clamp_arousal(arousal)
