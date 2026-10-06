"""Phase 1 acceptance: sampling weights unit-tested; REM chains show medium-distance associations."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pytest

from conftest import LOGS, frag, unit
from dream_engine.ingest.extract import HeuristicExtractor
from dream_engine.ingest.parse_log import parse_log
from dream_engine.memory.embed import HashingEmbedder
from dream_engine.memory.replay import (
    ReplaySampler, chain_distance_stats, compute_weights, novelty_scores, recency, stage_coefficients,
    stage_family,
)
from dream_engine.memory.store import MemoryStore

NOW = datetime(2026, 10, 5, 23, 0)


# ---------------------------------------------------------------- weight formula

def test_recency_decay():
    assert recency(NOW, NOW, 2) == pytest.approx(1.0)
    assert recency(NOW - timedelta(days=2), NOW, 2) == pytest.approx(math.exp(-1))
    assert recency(NOW - timedelta(days=4), NOW, 2) == pytest.approx(math.exp(-2))
    assert recency(NOW + timedelta(hours=1), NOW, 2) == pytest.approx(1.0)  # no boost from the future


def test_weight_matches_formula(cfg):
    r = cfg["replay"]
    c = stage_coefficients(r, "REM")
    f = frag("a", NOW - timedelta(days=1), valence=-0.6, arousal=0.8, replay_count=2)
    nov = np.array([0.4])
    (t,) = compute_weights([f], NOW, c, r["tau_days"], r["min_weight"], novelty=nov)
    expected = (c["alpha"] * math.exp(-1 / r["tau_days"]) + c["beta"] * 0.8 + c["gamma"] * 0.6
                + c["delta"] * 0.4 - c["epsilon"] * 2)
    assert t.weight == pytest.approx(expected)
    assert (t.recency, t.arousal, t.abs_valence, t.novelty, t.replay_count) == pytest.approx(
        (math.exp(-0.5), 0.8, 0.6, 0.4, 2))


def test_stage_coefficients_follow_config(cfg):
    r = cfg["replay"]
    for stage, fam in [("N2", "NREM"), ("N3", "NREM"), ("REM_late", "REM")]:
        assert stage_family(stage) == fam
        c = stage_coefficients(r, stage)
        for k in ("alpha", "beta", "gamma", "delta", "epsilon"):
            assert c[k] == pytest.approx(r[k] * r["stage_multipliers"][fam][k])
    nrem, rem = stage_coefficients(r, "N2"), stage_coefficients(r, "REM")
    assert nrem["alpha"] > rem["alpha"]   # NREM favors recency
    assert rem["beta"] > nrem["beta"]     # REM favors arousal


def test_replay_count_penalty_and_floor(cfg):
    r = cfg["replay"]
    c = stage_coefficients(r, "N2")
    ws = [compute_weights([frag("a", NOW, arousal=0.5, replay_count=k)], NOW, c, r["tau_days"],
                          r["min_weight"], novelty=np.array([0.0]))[0].weight for k in range(0, 30, 3)]
    assert all(a >= b for a, b in zip(ws, ws[1:]))
    assert ws[0] > ws[1]
    assert ws[-1] == pytest.approx(r["min_weight"])


def test_novelty():
    e = np.stack([unit([1, 0, 0]), unit([1, 0, 0]), unit([0, 1, 0]), unit([0, 0, 1])])
    nov = novelty_scores(e)
    assert nov[0] == pytest.approx(0.0) and nov[1] == pytest.approx(0.0)  # duplicates are not novel
    assert nov[2] == pytest.approx(1.0) and nov[3] == pytest.approx(1.0)


# ---------------------------------------------------------------- sampling behaviour

def test_seed_frequencies_proportional_to_weights(cfg):
    cfg["replay"]["old_memory_prob"] = 0.0
    frags = [frag(f"f{i}", NOW - timedelta(hours=6 * i), valence=0.2 * i - 0.4, arousal=0.15 * i,
                  emb=np.eye(5)[i]) for i in range(5)]
    sampler = ReplaySampler(frags, cfg["replay"], seed=1)
    c = stage_coefficients(cfg["replay"], "REM")
    w = np.array([t.weight for t in compute_weights(frags, NOW, c, cfg["replay"]["tau_days"],
                                                    cfg["replay"]["min_weight"])])
    trials = 20000
    counts = Counter(sampler.sample("REM", 1, NOW).picks[0].fragment.id for _ in range(trials))
    for i, f in enumerate(frags):
        assert counts[f.id] / trials == pytest.approx(w[i] / w.sum(), abs=0.015)


def test_nrem_favors_recent_rem_favors_arousal(cfg):
    cfg["replay"]["old_memory_prob"] = 0.0
    recent_calm = frag("recent_calm", NOW - timedelta(hours=2), valence=0.1, arousal=0.1, emb=[1, 0])
    older_intense = frag("older_intense", NOW - timedelta(days=5), valence=-0.8, arousal=0.9, emb=[0, 1])
    sampler = ReplaySampler([recent_calm, older_intense], cfg["replay"], seed=3)

    def p_recent(stage):
        picks = [sampler.sample(stage, 1, NOW).picks[0].fragment.id for _ in range(4000)]
        return picks.count("recent_calm") / len(picks)

    assert p_recent("N2") > 0.5 > p_recent("REM")


def test_old_memory_probability(cfg):
    r = cfg["replay"]
    frags = [frag(f"new{i}", NOW - timedelta(hours=i + 1), arousal=0.5, emb=np.eye(8)[i]) for i in range(4)]
    frags += [frag(f"old{i}", NOW - timedelta(days=45 + i), arousal=0.5, emb=np.eye(8)[4 + i]) for i in range(4)]
    sampler = ReplaySampler(frags, r, seed=7)
    trials = 6000
    modes = [sampler.sample("REM", 1, NOW).picks[0].mode for _ in range(trials)]
    assert modes.count("dream_lag") / trials == pytest.approx(r["old_memory_prob"], abs=0.02)

    r0 = dict(r, old_memory_prob=0.0)
    s0 = ReplaySampler(frags, r0, seed=7)
    picks = [p.fragment.id for _ in range(500) for p in s0.sample("REM", 1, NOW).picks]
    assert not any(fid.startswith("old") for fid in picks)


def test_future_fragments_are_never_replayed(cfg):
    frags = [frag("past", NOW - timedelta(hours=3), emb=[1, 0]),
             frag("future", NOW + timedelta(hours=3), arousal=1.0, emb=[0, 1])]
    sampler = ReplaySampler(frags, cfg["replay"], seed=0)
    for _ in range(200):
        assert [f.id for f in sampler.sample("REM", 2, NOW).fragments] == ["past"]


def test_same_seed_same_replay(cfg):
    frags = [frag(f"f{i}", NOW - timedelta(hours=i), arousal=0.1 * i, emb=np.eye(10)[i]) for i in range(10)]
    a = [f.id for f in ReplaySampler(frags, cfg["replay"], seed=42).sample("REM", 5, NOW).fragments]
    b = [f.id for f in ReplaySampler(frags, cfg["replay"], seed=42).sample("REM", 5, NOW).fragments]
    assert a == b and len(set(a)) == 5


# ---------------------------------------------------------------- medium-distance associations

def _fan(n: int) -> np.ndarray:
    """Unit vectors whose cosine to e0 decreases strictly: distances to e0 are distinct and ordered."""
    out = []
    for k in range(n):
        theta = (k + 1) * (math.pi / 2) / (n + 1)
        v = np.zeros(n + 2)
        v[0], v[k + 2] = math.cos(theta), math.sin(theta)
        out.append(v)
    return np.array(out)


def test_rem_association_skips_nearest_and_far(cfg):
    """With the anchor seeded first, every REM association lands in the medium distance band."""
    r = dict(cfg["replay"], old_memory_prob=0.0)
    n_cand = 20
    anchor = frag("anchor", NOW - timedelta(minutes=1), valence=-1.0, arousal=1.0,
                  emb=np.eye(n_cand + 2)[0])
    cands = [frag(f"c{k:02d}", NOW - timedelta(days=3), emb=v) for k, v in enumerate(_fan(n_cand))]
    sampler = ReplaySampler([anchor] + cands, r, seed=11)
    q_lo, q_hi = r["rem_distance_quantiles"]
    ranks = []
    for _ in range(400):
        res = sampler.sample("REM", 2, NOW)
        if res.picks[0].fragment.id != "anchor":
            continue
        p = res.picks[1]
        assert p.mode == "association"
        ranks.append(p.distance_rank)
        lo, hi = p.distance_band
        assert lo <= p.distance_from_prev <= hi
    assert ranks, "anchor should usually be seeded"
    m = n_cand
    assert min(ranks) >= max(1, math.floor(q_lo * m))   # never the nearest neighbour
    assert max(ranks) < math.ceil(q_hi * m)             # never the far, random-looking tail


def test_nrem_association_stays_near(cfg):
    r = dict(cfg["replay"], old_memory_prob=0.0)
    n_cand = 20
    anchor = frag("anchor", NOW - timedelta(minutes=1), valence=-1.0, arousal=1.0, emb=np.eye(n_cand + 2)[0])
    cands = [frag(f"c{k:02d}", NOW - timedelta(days=3), emb=v) for k, v in enumerate(_fan(n_cand))]
    sampler = ReplaySampler([anchor] + cands, r, seed=5)
    ranks = [res.picks[1].distance_rank for res in (sampler.sample("N2", 2, NOW) for _ in range(300))
             if res.picks[0].fragment.id == "anchor"]
    assert ranks and max(ranks) < math.ceil(r["nrem_distance_quantiles"][1] * n_cand)


@pytest.fixture
def sample_memory(cfg, tmp_path):
    """The bundled day logs, ingested offline (heuristic extractor + hashing embedder)."""
    ext, emb = HeuristicExtractor(cfg), HashingEmbedder(cfg["embedding"]["hashing_dim"], cfg["embedding"]["hashing_ngram_weight"])
    store = MemoryStore(tmp_path / "memory.db")
    for log in sorted(LOGS.iterdir()):
        frags = ext.extract(parse_log(log, cfg["ingest"]["default_entry_time"]))
        store.add(frags, emb.embed([f.text for f in frags]), emb.name)
    yield store
    store.close()


def test_rem_chains_medium_distance_on_day_logs(cfg, sample_memory):
    """Acceptance: on real day-log content, REM chain steps are farther than the nearest
    neighbour, closer than a random pair, and NREM steps stay closer than REM steps."""
    frags = sample_memory.all()
    assert len(frags) >= 20
    rem = chain_distance_stats(ReplaySampler(frags, cfg["replay"], seed=42), "REM_late", 5, NOW, 300)
    nrem = chain_distance_stats(ReplaySampler(frags, cfg["replay"], seed=42), "N2", 2, NOW, 300)

    assert rem["association_steps"] > 500
    assert rem["mean_nearest_neighbour_distance"] < rem["mean_association_distance"] < rem["mean_random_pair_distance"]
    assert rem["share_steps_to_nearest_neighbour"] == 0.0
    assert nrem["mean_association_distance"] < rem["mean_association_distance"]
