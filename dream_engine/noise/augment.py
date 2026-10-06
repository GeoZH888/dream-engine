"""Overfitted Brain Hypothesis (Hoel, 2021): noise in replay aids generalisation.

Perturb a seed fragment's embedding with Gaussian noise and retrieve the stored
fragment nearest to the perturbed point: a slightly-off, "corrupted" recall.
``sigma`` is the expected norm of the perturbation (per-dimension std = sigma/√d),
so its meaning does not depend on the embedding dimension.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from dream_engine.memory.store import Fragment
from dream_engine.seed import DreamSeed, SeedFragment


def perturb(vec: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma / np.sqrt(len(vec)), size=len(vec))
    out = vec + noise
    return out / np.linalg.norm(out)


def augment_seed(seed: DreamSeed, pool: list[Fragment], count: int, sigma: float,
                 rng: np.random.Generator) -> DreamSeed:
    by_id = {f.id: f for f in pool}
    frags = list(seed.fragments)
    for _ in range(count):
        sources = [f for f in frags if f.role == "replay" and f.id in by_id]
        used = {f.id for f in frags}
        candidates = [f for f in pool if f.id not in used]
        if not sources or not candidates:
            break
        src = sources[int(rng.integers(len(sources)))]
        src_vec = by_id[src.id].embedding
        point = perturb(src_vec, sigma, rng)
        sims = np.stack([c.embedding for c in candidates]) @ point
        k = int(np.argmax(sims))
        hit = candidates[k]
        frags.append(SeedFragment.from_fragment(hit, "augment", {
            "source_fragment": src.id, "sigma": sigma,
            "perturbation_cosine": round(float(point @ src_vec), 4),
            "distance_from_source": round(float(1.0 - hit.embedding @ src_vec), 4),
        }))
    return replace(seed, fragments=tuple(frags))
