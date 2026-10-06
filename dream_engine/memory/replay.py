"""Hippocampal replay sampler (spec §4.2).

Sampling weight for fragment i::

    w_i = α·recency_i + β·arousal_i + γ·|valence_i| + δ·novelty_i − ε·replay_count_i
    recency_i = exp(−Δt_i / τ)

- Coefficients are modulated per stage: NREM favors recency, REM favors arousal.
- The first fragment of an episode is drawn by weight. Each following fragment is an
  *association* from the previous one, drawn by weight from a band of embedding
  distances: NREM stays near (episodic), REM takes a *medium*-distance band —
  not the nearest neighbour, not a random pick — producing loose association chains.
- With probability ``old_memory_prob`` a slot pulls a memory older than
  ``old_memory_days`` instead (the dream-lag effect).

Every pick carries a trace explaining why it was chosen.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any

import numpy as np

from dream_engine.memory.store import Fragment

COEFFS = ("alpha", "beta", "gamma", "delta", "epsilon")


def stage_family(stage: str) -> str:
    """N1/N2/N3 -> NREM; REM, REM_early, REM_late -> REM."""
    s = stage.upper()
    if s.startswith("REM"):
        return "REM"
    if s in {"N1", "N2", "N3", "NREM"}:
        return "NREM"
    raise ValueError(f"Unknown sleep stage: {stage}")


def stage_coefficients(replay_cfg: dict[str, Any], stage: str) -> dict[str, float]:
    mult = replay_cfg["stage_multipliers"][stage_family(stage)]
    return {k: replay_cfg[k] * mult.get(k, 1.0) for k in COEFFS}


def recency(ts: datetime, now: datetime, tau_days: float) -> float:
    dt_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return math.exp(-dt_days / tau_days)


def novelty_scores(embeddings: np.ndarray) -> np.ndarray:
    """Distinctiveness: 1 − max cosine similarity to any other fragment, in [0, 1]."""
    n = len(embeddings)
    if n <= 1:
        return np.ones(n)
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)
    return np.clip(1.0 - sims.max(axis=1), 0.0, 1.0)


@dataclass
class WeightTerms:
    recency: float
    arousal: float
    abs_valence: float
    novelty: float
    replay_count: int
    weight: float

    def explain(self, c: dict[str, float]) -> str:
        return (f"w={self.weight:.3f} = {c['alpha']:.2f}·rec {self.recency:.2f} + {c['beta']:.2f}·ar "
                f"{self.arousal:.2f} + {c['gamma']:.2f}·|val| {self.abs_valence:.2f} + {c['delta']:.2f}·nov "
                f"{self.novelty:.2f} − {c['epsilon']:.2f}·replays {self.replay_count}")


def compute_weights(frags: list[Fragment], now: datetime, coeffs: dict[str, float], tau_days: float,
                    min_weight: float, novelty: np.ndarray | None = None) -> list[WeightTerms]:
    if novelty is None:
        novelty = novelty_scores(np.stack([f.embedding for f in frags])) if frags else np.array([])
    out = []
    for f, nov in zip(frags, novelty):
        rec = recency(f.timestamp, now, tau_days)
        raw = (coeffs["alpha"] * rec + coeffs["beta"] * f.arousal + coeffs["gamma"] * abs(f.valence)
               + coeffs["delta"] * float(nov) - coeffs["epsilon"] * f.replay_count)
        out.append(WeightTerms(rec, f.arousal, abs(f.valence), float(nov), f.replay_count, max(min_weight, raw)))
    return out


@dataclass
class Pick:
    fragment: Fragment
    step: int
    mode: str                       # seed | association | dream_lag
    terms: WeightTerms
    probability: float              # probability of this pick among the candidates at that step
    candidates: int
    distance_from_prev: float | None = None
    distance_band: tuple[float, float] | None = None
    distance_rank: int | None = None  # 0 = nearest candidate to the previous link

    def trace(self, coeffs: dict[str, float]) -> dict[str, Any]:
        t = {
            "step": self.step, "fragment_id": self.fragment.id, "mode": self.mode,
            "weight": round(self.terms.weight, 4), "weight_terms": {k: round(v, 4) if isinstance(v, float) else v
                                                                    for k, v in asdict(self.terms).items()},
            "explanation": self.terms.explain(coeffs),
            "probability": round(self.probability, 4), "candidates": self.candidates,
        }
        if self.distance_from_prev is not None:
            t["distance_from_prev"] = round(self.distance_from_prev, 4)
            t["distance_band"] = [round(x, 4) for x in self.distance_band]
            t["distance_rank"] = self.distance_rank
        return t


@dataclass
class ReplayResult:
    stage: str
    now: datetime
    coefficients: dict[str, float]
    picks: list[Pick] = field(default_factory=list)

    @property
    def fragments(self) -> list[Fragment]:
        return [p.fragment for p in self.picks]

    def traces(self) -> list[dict[str, Any]]:
        return [p.trace(self.coefficients) for p in self.picks]


class ReplaySampler:
    def __init__(self, fragments: list[Fragment], replay_cfg: dict[str, Any], seed: int | None = None):
        self.cfg = replay_cfg
        self.rng = np.random.default_rng(seed)
        # private copies: replay counts change during a night without touching the caller's objects
        self.fragments = [replace(f) for f in fragments]
        self.emb = (np.stack([f.embedding for f in self.fragments]) if self.fragments
                    else np.zeros((0, 1), dtype=np.float32))
        self.novelty = novelty_scores(self.emb)

    def note_replayed(self, fragment_ids: list[str]) -> None:
        """Count a replay within this sampler's lifetime (e.g. one simulated night)."""
        ids = set(fragment_ids)
        for f in self.fragments:
            if f.id in ids:
                f.replay_count += 1

    # -- helpers
    def _draw(self, idx: list[int], weights: np.ndarray) -> tuple[int, float]:
        w = weights[idx]
        p = w / w.sum()
        k = int(self.rng.choice(len(idx), p=p))
        return idx[k], float(p[k])

    def _band(self, prev: int, cand: list[int], quantiles: tuple[float, float]) -> tuple[list[int], np.ndarray, list[int]]:
        """Candidates whose distance rank from ``prev`` lies in the quantile band."""
        d = 1.0 - self.emb[cand] @ self.emb[prev]
        order = np.argsort(d, kind="stable")
        m = len(cand)
        q_lo, q_hi = quantiles
        lo = int(math.floor(q_lo * m))
        if q_lo > 0 and m > 1:
            lo = max(lo, 1)          # a positive lower quantile always excludes the nearest neighbour
        lo = min(lo, m - 1)
        hi = max(lo + 1, int(math.ceil(q_hi * m)))
        chosen = order[lo:hi]
        ranks = [int(r) for r in range(lo, hi)]
        return [cand[i] for i in chosen], d, ranks

    # -- main entry point
    def sample(self, stage: str, n: int, now: datetime) -> ReplayResult:
        family = stage_family(stage)
        coeffs = stage_coefficients(self.cfg, stage)
        result = ReplayResult(stage=stage, now=now, coefficients=coeffs)

        pool = [i for i, f in enumerate(self.fragments) if f.timestamp <= now]
        if not pool or n <= 0:
            return result
        terms = compute_weights(self.fragments, now, coeffs, self.cfg["tau_days"], self.cfg["min_weight"],
                                novelty=self.novelty)
        weights = np.array([t.weight for t in terms])
        old_cut = self.cfg["old_memory_days"] * 86400.0
        is_old = {i: (now - self.fragments[i].timestamp).total_seconds() > old_cut for i in pool}
        quantiles = tuple(self.cfg["rem_distance_quantiles" if family == "REM" else "nrem_distance_quantiles"])

        chosen: list[int] = []
        for step in range(min(n, len(pool))):
            available = [i for i in pool if i not in chosen]
            old = [i for i in available if is_old[i]]
            recent = [i for i in available if not is_old[i]]

            if old and self.rng.random() < self.cfg["old_memory_prob"]:
                i, p = self._draw(old, weights)
                pick = Pick(self.fragments[i], step, "dream_lag", terms[i], p, len(old))
            elif not chosen:
                cand = recent or available
                i, p = self._draw(cand, weights)
                pick = Pick(self.fragments[i], step, "seed", terms[i], p, len(cand))
            else:
                prev = chosen[-1]
                band, d_all, ranks = self._band(prev, available, quantiles)
                i, p = self._draw(band, weights)
                d_band = 1.0 - self.emb[band] @ self.emb[prev]
                pick = Pick(self.fragments[i], step, "association", terms[i], p, len(band),
                            distance_from_prev=float(1.0 - self.emb[i] @ self.emb[prev]),
                            distance_band=(float(d_band.min()), float(d_band.max())),
                            distance_rank=ranks[band.index(i)])
            chosen.append(i)
            result.picks.append(pick)
        return result


def chain_distance_stats(sampler: ReplaySampler, stage: str, n: int, now: datetime, chains: int) -> dict[str, float]:
    """Compare association step distances against nearest-neighbour and random-pair baselines.

    For REM, the acceptance criterion is: nearest < REM association < random.
    """
    emb = sampler.emb
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    nn_dist = float(np.mean(1.0 - sims.max(axis=1)))
    iu = np.triu_indices(len(emb), k=1)
    rand_dist = float(np.mean(1.0 - (emb @ emb.T)[iu]))

    steps, ranks = [], []
    for _ in range(chains):
        for pick in sampler.sample(stage, n, now).picks:
            if pick.mode == "association":
                steps.append(pick.distance_from_prev)
                ranks.append(pick.distance_rank)
    return {
        "stage": stage,
        "chains": chains,
        "association_steps": len(steps),
        "mean_nearest_neighbour_distance": nn_dist,
        "mean_association_distance": float(np.mean(steps)) if steps else float("nan"),
        "mean_random_pair_distance": rand_dist,
        "share_steps_to_nearest_neighbour": float(np.mean([r == 0 for r in ranks])) if ranks else float("nan"),
    }
