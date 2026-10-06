"""Visual association cortex: image prompts for REM episodes (spec §4.6).

Prompts only; an image-model backend can be plugged in later via ``ImageBackend``.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from dream_engine.llm import make_client, structured_call

VISUAL_SYSTEM = """You write image-generation prompts for scenes from a dream report, for a \
research simulation of dream imagery. Each prompt depicts one concrete moment of the report: \
subject, setting, light and colour, viewpoint. Keep the dream's impossibilities exactly as \
described. No text in the image, no named real people, no symbolism that is not in the report. \
Each prompt is one sentence of at most 60 words and ends with the given style."""

VISUAL_SCHEMA = {
    "type": "object",
    "properties": {"prompts": {"type": "array", "items": {"type": "string"}}},
    "required": ["prompts"],
    "additionalProperties": False,
}


class ImageBackend(Protocol):
    """Pluggable later phase: turn a prompt into an image file path."""

    def render(self, prompt: str, out_path: str) -> str: ...


def prompt_count(profile: dict[str, Any], rng: np.random.Generator) -> int:
    lo, hi = profile["image_prompts"]
    return int(rng.integers(lo, hi + 1)) if hi > 0 else 0


class LLMVisualizer:
    def __init__(self, cfg: dict[str, Any], client: Any = None):
        self.cfg = cfg
        self.client = client or make_client()

    def prompts(self, narrative: str, n: int) -> list[str]:
        if n <= 0:
            return []
        style = self.cfg["visual"]["style"]
        data = structured_call(
            self.client, self.cfg, system=VISUAL_SYSTEM,
            user=f"Write exactly {n} prompt(s), for moments in order through the report. "
                 f"Style to end each prompt with: {style}\n\nDream report:\n{narrative}",
            schema=VISUAL_SCHEMA, effort=self.cfg["llm"]["visual_effort"])
        return [p.strip() for p in data["prompts"] if p.strip()][:n]


class DryRunVisualizer:
    def __init__(self, cfg: dict[str, Any]):
        self.style = cfg["visual"]["style"]

    def prompts(self, narrative: str, n: int) -> list[str]:
        return [f"{narrative[:80]}..., {self.style}"] * n
