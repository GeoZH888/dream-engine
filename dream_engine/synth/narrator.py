"""Narrative construction (cortical synthesis): seed -> first-person dream report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from dream_engine.llm import make_client, structured_call
from dream_engine.memory import emotion
from dream_engine.seed import DreamSeed
from dream_engine.synth.prompts import NARRATIVE_SCHEMA, SYSTEM_CRITIC_OFF, build_user_prompt


@dataclass
class Narration:
    narrative: str
    valence: float
    arousal: float
    prompt: str
    narrator: str


class Narrator(Protocol):
    name: str

    def narrate(self, seed: DreamSeed) -> Narration: ...


class LLMNarrator:
    def __init__(self, cfg: dict[str, Any], client: Any = None):
        self.cfg = cfg
        self.client = client or make_client()
        self.name = f"llm:{cfg['model']}"

    def narrate(self, seed: DreamSeed) -> Narration:
        profile = self.cfg["stages"][seed.profile]
        prompt = build_user_prompt(seed, profile, self.cfg["llm"]["send_temperature"])
        data = structured_call(self.client, self.cfg, system=SYSTEM_CRITIC_OFF, user=prompt,
                               schema=NARRATIVE_SCHEMA, effort=self.cfg["llm"]["narrate_effort"],
                               temperature=profile["temperature"])
        tone = data["emotional_tone"]
        return Narration(data["narrative"].strip(), emotion.clamp_valence(tone["valence"]),
                         emotion.clamp_arousal(tone["arousal"]), prompt, self.name)


class DryRunNarrator:
    """No LLM: concatenates the seed, for tests and for inspecting seeds (`--dry-run`)."""

    name = "dry-run"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    def narrate(self, seed: DreamSeed) -> Narration:
        profile = self.cfg["stages"][seed.profile]
        prompt = build_user_prompt(seed, profile, self.cfg["llm"]["send_temperature"])
        parts = [f.text for f in seed.fragments] + [op.directive for op in seed.operators]
        w = [max(f.arousal, 1e-3) for f in seed.fragments]
        val = sum(f.valence * x for f, x in zip(seed.fragments, w)) / sum(w)
        return Narration(" ".join(parts), val, max(f.arousal for f in seed.fragments), prompt, self.name)
