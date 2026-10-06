"""A simulated night: schedule → replay → noise → narrator → visual prompts.

Planning (all stochastic choices) is sequential and seeded, so a (date, seed) pair
always yields the same seeds. Narration is the only LLM step and runs in parallel.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

from dream_engine.memory.replay import ReplaySampler
from dream_engine.memory.store import Fragment
from dream_engine.noise.augment import augment_seed
from dream_engine.noise.bizarre import apply_operators, structural_bizarreness
from dream_engine.noise.pgo import apply_pgo
from dream_engine.seed import DreamSeed, SeedFragment
from dream_engine.sleep.scheduler import EpisodeSlot, Segment, build_night, episode_slots, sleep_onset
from dream_engine.synth.lucid import maybe_lucid
from dream_engine.synth.narrator import Narrator
from dream_engine.visual.prompts import prompt_count


@dataclass
class PlannedEpisode:
    slot: EpisodeSlot
    seed: DreamSeed
    clock: datetime
    replay_traces: list[dict[str, Any]]
    replay_coefficients: dict[str, float]
    n_images: int


@dataclass
class NightPlan:
    night_id: str
    seed: int
    onset: datetime
    segments: list[Segment]
    episodes: list[PlannedEpisode]
    hypnogram_source: str = "simulated"


def _range(rng: np.random.Generator, lo_hi: list[int]) -> int:
    lo, hi = lo_hi
    return int(rng.integers(lo, hi + 1))


def plan_night(cfg: dict[str, Any], fragments: list[Fragment], night: date, seed: int,
               lucid: bool = False, segments: list[Segment] | None = None,
               hypnogram_source: str | None = None) -> NightPlan:
    """Plan every episode. ``segments`` replaces the simulated timeline with a real hypnogram."""
    r_sched, r_replay, r_count, r_noise, r_ops, r_misc = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6))
    segments = segments if segments is not None else build_night(cfg, r_sched)
    onset = sleep_onset(cfg, night)
    sampler = ReplaySampler(fragments, cfg["replay"], seed=int(r_replay.integers(2**63)))

    planned = []
    for slot in episode_slots(cfg, segments):
        profile = cfg["stages"][slot.profile]
        clock = onset + timedelta(minutes=slot.minute_start)
        result = sampler.sample(slot.profile, _range(r_count, profile["fragments"]), clock)
        sampler.note_replayed([f.id for f in result.fragments])
        traces = result.traces()

        seed_ = DreamSeed(slot.stage, slot.profile, slot.cycle, slot.minute_start, fragments=tuple(
            SeedFragment.from_fragment(p.fragment, "replay", {"mode": t["mode"], "weight": t["weight"]})
            for p, t in zip(result.picks, traces)))
        pool = [f for f in sampler.fragments if f.timestamp <= clock]
        seed_ = augment_seed(seed_, pool, profile["augment"], cfg["noise"]["embedding_sigma"], r_noise)
        if slot.stage == "REM":
            seed_ = apply_pgo(seed_, slot.segment.minute_start, slot.segment.minute_end, pool, cfg["noise"], r_noise)
        seed_ = apply_operators(seed_, _range(r_ops, profile["bizarre_ops"]), r_ops, cfg["bizarre"])
        seed_ = maybe_lucid(seed_, cfg, r_misc, lucid)
        planned.append(PlannedEpisode(slot, seed_, clock, traces, result.coefficients,
                                      prompt_count(profile, r_misc)))
    return NightPlan(night.isoformat(), seed, onset, segments, planned, hypnogram_source or "simulated")


def narrate_episode(cfg: dict[str, Any], night_id: str, ep: PlannedEpisode, narrator: Narrator,
                    visualizer: Any) -> dict[str, Any]:
    seed, profile = ep.seed, cfg["stages"][ep.seed.profile]
    out: dict[str, Any] = {
        "night_id": night_id,
        "cycle": seed.cycle,
        "stage": seed.stage,
        "profile": seed.profile,
        "minute_start": seed.minute_start,
        "clock_time": ep.clock.strftime("%H:%M"),
        "seed_fragments": [f.id for f in seed.fragments],
        "operators_applied": [op.name for op in seed.operators],
        "lucid": seed.lucid,
        "narrative": None,
        "word_count": 0,
        "image_prompts": [],
        "bizarreness_score": None,              # filled by `dream eval` (LLM rater)
        "bizarreness_prior": structural_bizarreness(seed, cfg["bizarre"]),
        "emotional_tone": None,
        "trace": {
            "fragments": [f.to_dict() for f in seed.fragments],
            "replay": ep.replay_traces,
            "replay_coefficients": ep.replay_coefficients,
            "operators": [op.to_dict() for op in seed.operators],
            "pgo_events": [e.to_dict() for e in seed.pgo_events],
            "stage_profile": {k: profile[k] for k in ("fragments", "bizarre_ops", "words", "augment", "style")},
            "temperature_nominal": profile["temperature"],
            "temperature_sent": bool(cfg["llm"]["send_temperature"]),
        },
    }
    try:
        n = narrator.narrate(seed)
        out.update(narrative=n.narrative, word_count=len(n.narrative.split()),
                   emotional_tone={"valence": round(n.valence, 3), "arousal": round(n.arousal, 3)})
        out["trace"].update(narrator=n.narrator, prompt=n.prompt)
        if visualizer and ep.n_images:
            out["image_prompts"] = visualizer.prompts(n.narrative, ep.n_images)
    except Exception as e:  # one failed episode must not lose the night
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def run_night(cfg: dict[str, Any], plan: NightPlan, narrator: Narrator, visualizer: Any = None,
              episodes: list[PlannedEpisode] | None = None) -> dict[str, Any]:
    eps = plan.episodes if episodes is None else episodes
    with ThreadPoolExecutor(max_workers=cfg["llm"]["concurrency"]) as pool:
        results = list(pool.map(lambda ep: narrate_episode(cfg, plan.night_id, ep, narrator, visualizer), eps))
    return {
        "night_id": plan.night_id,
        "seed": plan.seed,
        "model": cfg["model"],
        "narrator": getattr(narrator, "name", "?"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sleep_onset": plan.onset.isoformat(timespec="minutes"),
        "cycles": max(s.cycle for s in plan.segments),
        "hypnogram_source": plan.hypnogram_source,
        "hypnogram": [s.to_dict() for s in plan.segments],
        "config": {k: cfg[k] for k in ("replay", "noise", "bizarre", "stages", "sleep")},
        "episodes": results,
    }
