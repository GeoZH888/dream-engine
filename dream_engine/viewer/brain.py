"""Brain mechanism map: how active each simulated brain system was in a dream.

This is a schematic of the simulation, not a measurement. Each region's level (0-1)
comes from the engine variable that models it, so the map explains the dream's
ingredients:

- hippocampus: memory replay (fragments replayed vs the stage's maximum)
- amygdala: emotional salience (mean arousal and |valence| of what was replayed)
- pons (brainstem): PGO bursts (count vs the per-episode cap); REM only
- visual cortex: imagery (image prompts in REM; faint in NREM)
- prefrontal cortex: the critic (on in NREM, off in REM, partly back in lucid REM)
- thalamus: sensory gating and spindles (N2 high, N3 lower, REM low)

The REM / NREM pattern follows the textbook picture of sleep neuroimaging
(e.g. Maquet 1996; Braun 1997): limbic and visual areas up and dorsolateral
prefrontal cortex down in REM.
"""

from __future__ import annotations

from typing import Any

# The fixed parts of the textbook pattern; everything else is computed per dream.
PREFRONTAL = {"N2": 0.75, "N3": 0.45, "REM": 0.12}
PREFRONTAL_LUCID = 0.55
THALAMUS = {"N2": 0.8, "N3": 0.5, "REM": 0.3}
VISUAL_NREM = {"N2": 0.2, "N3": 0.08}

REGIONS = {
    "prefrontal": ("Prefrontal cortex", "The critic: switched off in REM, so the dreamer never questions anything"),
    "visual": ("Visual cortex", "Imagery: scene images (REM image prompts)"),
    "hippocampus": ("Hippocampus", "Memory replay: how many day-residue fragments were replayed"),
    "amygdala": ("Amygdala", "Emotional salience: arousal and |valence| of the replayed memories"),
    "thalamus": ("Thalamus", "Sensory gating and sleep spindles (strongest in N2)"),
    "pons": ("Pons (brainstem)", "PGO bursts: random activations that cut scenes and inject memories"),
}


def brain_activity(ep: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stage = ep["stage"]
    t = ep["trace"]
    replayed = [f for f in t["fragments"] if f["role"] == "replay"]
    max_frags = max(s["fragments"][1] for s in cfg["stages"].values())
    arousal = sum(f["arousal"] for f in replayed) / len(replayed) if replayed else 0.0
    valence = sum(abs(f["valence"]) for f in replayed) / len(replayed) if replayed else 0.0
    n_img = len(ep.get("image_prompts") or [])
    max_img = max(s["image_prompts"][1] for s in cfg["stages"].values()) or 1

    levels = {
        "prefrontal": (PREFRONTAL_LUCID if ep.get("lucid") else PREFRONTAL[stage],
                       "lucid: critic partly back" if ep.get("lucid") else f"{stage}: critic {'off' if stage == 'REM' else 'on'}"),
        "visual": ((0.35 + 0.65 * n_img / max_img) if stage == "REM" else VISUAL_NREM[stage],
                   f"{n_img} image prompt{'s' if n_img != 1 else ''}" if stage == "REM" else "little imagery"),
        "hippocampus": (len(replayed) / max_frags, f"{len(replayed)} of up to {max_frags} memories replayed"),
        "amygdala": (0.5 * arousal + 0.5 * valence, f"arousal {arousal:.2f}, |valence| {valence:.2f}"),
        "thalamus": (THALAMUS[stage], f"{stage} gating"),
        "pons": (len(t["pgo_events"]) / cfg["noise"]["pgo_max_events"] if stage == "REM" else 0.0,
                 (f"{len(t['pgo_events'])} PGO burst{'s' if len(t['pgo_events']) != 1 else ''}"
                  if stage == "REM" else "no PGO bursts in NREM")),
    }
    return {k: {"level": round(min(1.0, max(0.0, v)), 3), "why": why, "name": REGIONS[k][0], "role": REGIONS[k][1]}
            for k, (v, why) in levels.items()}
