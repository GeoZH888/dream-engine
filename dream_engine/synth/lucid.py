"""Lucidity (Voss et al.): the dlPFC partially re-activates in late REM.

Optional mode. A late-REM episode turns lucid with probability ``lucid.prob_late_rem``;
the narrator then lets the dreamer notice one anomaly and gain limited control.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from dream_engine.seed import DreamSeed


def maybe_lucid(seed: DreamSeed, cfg: dict[str, Any], rng: np.random.Generator, enabled: bool) -> DreamSeed:
    if enabled and seed.profile == "REM_late" and seed.operators and rng.random() < cfg["lucid"]["prob_late_rem"]:
        return replace(seed, lucid=True)
    return seed
