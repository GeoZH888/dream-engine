"""Bizarreness rater (spec §4.7): an LLM judge using Hobson's three categories.

- discontinuity: abrupt changes of scene, person, object or plot without transition
- incongruity: elements that do not fit together or are impossible in waking life
- uncertainty: explicit vagueness about identity, place or time

The rater is *blind*: it sees only the report text, never the stage, the seed, or
the operators. Score = mean of the three category scores, each in [0, 1].
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dream_engine.llm import make_client, structured_call

CATEGORIES = ("discontinuity", "incongruity", "uncertainty")

RATER_SYSTEM = """You rate the bizarreness of dream reports for sleep research, using Hobson's \
categories. Rate each category from 0 to 1 for this report as a whole:

- discontinuity: abrupt, unexplained changes of scene, character, object or plot.
- incongruity: elements that are mismatched, impossible or improbable in waking life \
(wrong features, impossible physics, merged identities or places).
- uncertainty: explicit vagueness about who someone is, where the dreamer is, or when.

Anchors: 0 = none at all (an ordinary waking-life account); 0.3 = one mild instance; \
0.6 = several clear instances; 1 = pervasive. Judge intensity, not length: a short report \
can be very bizarre and a long one can be entirely mundane. For each category give the score \
and a short quote or note as evidence (empty if the score is 0)."""

_CAT = {"type": "object",
        "properties": {"score": {"type": "number"}, "evidence": {"type": "string"}},
        "required": ["score", "evidence"], "additionalProperties": False}
RATER_SCHEMA = {"type": "object", "properties": {c: _CAT for c in CATEGORIES},
                "required": list(CATEGORIES), "additionalProperties": False}


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class BizarrenessRater:
    def __init__(self, cfg: dict[str, Any], client: Any = None):
        rater_model = cfg["eval"].get("rater_model") or cfg["model"]
        self.cfg = {**cfg, "model": rater_model}
        self.client = client or make_client()
        self.name = f"llm:{rater_model}"

    def rate(self, narrative: str) -> dict[str, Any]:
        data = structured_call(self.client, self.cfg, system=RATER_SYSTEM, user=f"Dream report:\n{narrative}",
                               schema=RATER_SCHEMA, effort=self.cfg["llm"]["rate_effort"])
        cats = {c: {"score": round(_clamp(data[c]["score"]), 3), "evidence": data[c]["evidence"]} for c in CATEGORIES}
        score = round(sum(v["score"] for v in cats.values()) / len(CATEGORIES), 3)
        return {"score": score, "categories": cats, "rater": self.name}


def rate_night(night: dict[str, Any], rater: BizarrenessRater, concurrency: int, rerate: bool = False) -> int:
    """Fill ``bizarreness_score`` (and the rating trace) in place; returns how many were rated."""
    todo = [e for e in night["episodes"]
            if e.get("narrative") and (rerate or e.get("bizarreness_score") is None)]

    def one(e):
        try:
            r = rater.rate(e["narrative"])
            e["bizarreness_score"] = r["score"]
            e["bizarreness_rating"] = r
        except Exception as ex:
            e["bizarreness_rating"] = {"error": f"{type(ex).__name__}: {ex}"}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, todo))
    return len(todo)
