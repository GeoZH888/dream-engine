"""PGO waves (activation-synthesis, Hobson & McCarley): semi-random bursts during REM.

Bursts follow a Poisson process over the REM segment's minutes. Each burst triggers
either an abrupt **scene transition** or the **injection of a random fragment**
(uniformly random, not salience-weighted: the brainstem does not choose).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from dream_engine.memory.store import Fragment
from dream_engine.seed import DreamSeed, PGOEvent, SeedFragment


def burst_times(start: float, end: float, rate_per_min: float, rng: np.random.Generator,
                max_events: int) -> list[float]:
    """Poisson process on [start, end) (exponential inter-arrival times).

    If it yields more than ``max_events`` bursts, a random subset is kept, so the
    survivors still spread over the whole segment instead of clustering at its start.
    """
    times, t = [], start
    if rate_per_min <= 0:
        return times
    while True:
        t += rng.exponential(1.0 / rate_per_min)
        if t >= end:
            break
        times.append(t)
    if len(times) > max_events:
        keep = rng.choice(len(times), size=max_events, replace=False)
        times = [times[i] for i in sorted(keep)]
    return times


def apply_pgo(seed: DreamSeed, seg_start: float, seg_end: float, pool: list[Fragment],
              noise_cfg: dict[str, Any], rng: np.random.Generator) -> DreamSeed:
    times = burst_times(seg_start, seg_end, noise_cfg["pgo_rate_per_min"], rng, noise_cfg["pgo_max_events"])
    events, injected = [], list(seed.fragments)
    used = {f.id for f in seed.fragments}
    for t in times:
        position = (t - seg_start) / max(seg_end - seg_start, 1e-9)
        candidates = [f for f in pool if f.id not in used]
        if candidates and rng.random() < noise_cfg["pgo_injection_prob"]:
            f = candidates[int(rng.integers(len(candidates)))]
            used.add(f.id)
            injected.append(SeedFragment.from_fragment(
                f, "pgo_injection", {"minute": round(t, 1), "chosen": "uniformly at random",
                                     "candidates": len(candidates)}))
            events.append(PGOEvent(t, position, "fragment_injection", f.id))
        else:
            events.append(PGOEvent(t, position, "scene_transition"))
    return replace(seed, fragments=tuple(injected), pgo_events=tuple(events))
