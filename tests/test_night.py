"""Phase 2: scheduler + noise + narrator."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from conftest import LOGS, frag, unit
from dream_engine.cli import main
from dream_engine.night import plan_night, run_night
from dream_engine.noise import bizarre
from dream_engine.noise.augment import augment_seed, perturb
from dream_engine.noise.pgo import apply_pgo, burst_times
from dream_engine.report import night_markdown, write_night
from dream_engine.seed import DreamSeed, SeedFragment
from dream_engine.sleep.scheduler import build_night, episode_slots, hypnogram_text
from dream_engine.synth.lucid import maybe_lucid
from dream_engine.synth.narrator import DryRunNarrator, LLMNarrator
from dream_engine.synth.prompts import SYSTEM_CRITIC_OFF, build_user_prompt
from dream_engine.visual.prompts import DryRunVisualizer

NOW = datetime(2026, 10, 5, 23, 0)


# ---------------------------------------------------------------- scheduler

@pytest.mark.parametrize("seed", range(10))
def test_night_has_five_cycles_with_growing_rem(cfg, seed):
    segs = build_night(cfg, np.random.default_rng(seed))
    assert {s.cycle for s in segs} == set(range(1, cfg["cycles"] + 1))
    assert segs[0].minute_start == 0 and segs[-1].stage == "W"
    assert all(a.minute_end == b.minute_start for a, b in zip(segs, segs[1:]))  # contiguous
    assert abs(segs[-1].minute_start - cfg["night_hours"] * 60) <= 1
    rem = [s.minutes for s in segs if s.stage == "REM"]
    assert len(rem) == cfg["cycles"]
    jit = cfg["sleep"]["jitter_minutes"]
    lo, hi = cfg["sleep"]["rem_minutes"]
    assert abs(rem[0] - lo) <= jit + 1 and abs(rem[-1] - hi) <= jit + 1
    assert rem[-1] > rem[0] + 15
    n3 = [sum(s.minutes for s in segs if s.stage == "N3" and s.cycle == c) for c in range(1, 6)]
    assert n3[0] > n3[-1] and n3[-1] <= jit  # slow-wave sleep fades


def test_episode_slots(cfg):
    segs = build_night(cfg, np.random.default_rng(0))
    slots = episode_slots(cfg, segs)
    rem = [s for s in slots if s.stage == "REM"]
    assert len(rem) == 5 and len([s for s in slots if s.stage == "N2"]) == 5
    late = cfg["sleep"]["rem_late_from_cycle"]
    assert all(s.profile == ("REM_late" if s.cycle >= late else "REM_early") for s in rem)
    assert [s.minute_start for s in slots] == sorted(s.minute_start for s in slots)


def test_hypnogram_text(cfg):
    txt = hypnogram_text(build_night(cfg, np.random.default_rng(0)))
    rows = txt.splitlines()
    assert rows[1].startswith("REM") and "█" in rows[1] and "(hours)" in rows[-1]


# ---------------------------------------------------------------- noise

def _sf(fid, text, kinds, ts=NOW, role="replay"):
    return SeedFragment(fid, text, tuple(kinds), tuple(sorted(kinds.items())), 0.0, 0.5, ts, role)


SEED = DreamSeed("REM", "REM_late", 4, 300, fragments=(
    _sf("f1", "Met my supervisor at the station", {"my supervisor": "person", "station": "place"}),
    _sf("f2", "Video call with grandmother in the courtyard", {"grandmother": "person", "courtyard": "place"}),
    _sf("f3", "Laptop bag soaked in the rain", {"laptop bag": "object"}, ts=NOW - timedelta(days=45)),
))


def test_burst_times_poisson(cfg):
    rng = np.random.default_rng(0)
    counts = [len(burst_times(0, 30, 0.2, rng, 1000)) for _ in range(3000)]
    assert np.mean(counts) == pytest.approx(6.0, rel=0.05)    # rate × minutes
    assert np.var(counts) == pytest.approx(6.0, rel=0.15)     # Poisson: variance = mean
    assert max(len(burst_times(0, 30, 0.2, rng, 3)) for _ in range(200)) == 3
    capped = [t for _ in range(500) for t in burst_times(0, 40, 0.5, rng, 3)]
    assert np.mean(capped) == pytest.approx(20, abs=1.5)  # capped bursts still span the segment


def test_pgo_injects_random_fragments(cfg):
    pool = [frag(f"p{i}", NOW - timedelta(hours=i), emb=np.eye(10)[i]) for i in range(10)]
    noise = dict(cfg["noise"], pgo_rate_per_min=1.0, pgo_injection_prob=1.0, pgo_max_events=4)
    out = apply_pgo(SEED, 300, 330, pool, noise, np.random.default_rng(1))
    assert len(out.pgo_events) == 4 and all(e.kind == "fragment_injection" for e in out.pgo_events)
    injected = [f for f in out.fragments if f.role == "pgo_injection"]
    assert [f.id for f in injected] == [e.fragment_id for e in out.pgo_events]
    assert all(0 <= e.position < 1 for e in out.pgo_events)
    assert SEED.pgo_events == ()  # pure: input seed unchanged


def test_operators_target_the_right_entity_kinds(cfg):
    rng, p = np.random.default_rng(0), cfg["bizarre"]
    merged = bizarre.identity_merge(SEED, rng, p).operators[-1]
    assert {"my supervisor", "grandmother"} <= set(merged.involves)
    fused = bizarre.place_fusion(SEED, rng, p).operators[-1]
    assert {"station", "courtyard"} <= set(fused.involves)
    obj = bizarre.object_transformation(SEED, rng, p).operators[-1]
    assert "laptop bag" in obj.directive
    td = bizarre.time_distortion(SEED, rng, p).operators[-1]
    assert td.involves == ("f3",)  # the 45-day-old fragment overlaps the present
    assert SEED.operators == ()


def test_operators_skip_when_not_applicable(cfg):
    lone = DreamSeed("REM", "REM_early", 1, 0, fragments=(_sf("f1", "Read a paper", {}),))
    rng, p = np.random.default_rng(0), cfg["bizarre"]
    assert bizarre.identity_merge(lone, rng, p) is None
    assert bizarre.place_fusion(lone, rng, p) is None
    assert bizarre.scene_jump(lone, rng, p) is None
    assert bizarre.object_transformation(lone, rng, p) is None
    out = bizarre.apply_operators(lone, 2, rng, p)
    assert len(out.operators) == 2 and len({o.name for o in out.operators}) == 2


def test_structural_prior_increases(cfg):
    p = cfg["bizarre"]
    rng = np.random.default_rng(0)
    assert bizarre.structural_bizarreness(SEED, p) == 0.0
    assert bizarre.structural_bizarreness(bizarre.apply_operators(SEED, 3, rng, p), p) > \
           bizarre.structural_bizarreness(bizarre.apply_operators(SEED, 1, rng, p), p)


def test_augment_retrieves_near_the_source():
    rng = np.random.default_rng(0)
    v = unit([1, 0, 0, 0])
    assert perturb(v, 0.0, rng) @ v == pytest.approx(1.0)
    pool = [frag("src", NOW, emb=[1, 0, 0, 0]), frag("near", NOW, emb=[0.9, 0.1, 0, 0]),
            frag("far", NOW, emb=[0, 0, 0, 1])]
    seed = DreamSeed("REM", "REM_early", 1, 0, fragments=(SeedFragment.from_fragment(pool[0], "replay"),))
    out = augment_seed(seed, pool, 1, 0.15, rng)
    added = out.fragments[-1]
    assert added.id == "near" and added.role == "augment"
    assert dict(added.trace)["source_fragment"] == "src"


# ---------------------------------------------------------------- narrator prompts

def test_prompt_contents(cfg):
    seed = bizarre.apply_operators(SEED, 2, np.random.default_rng(0), cfg["bizarre"])
    seed = maybe_lucid(seed, dict(cfg, lucid={"prob_late_rem": 1.0}), np.random.default_rng(0), True)
    prompt = build_user_prompt(seed, cfg["stages"]["REM_late"], send_temperature=False)
    assert "250-400 words" in prompt and "[f1]" in prompt
    assert all(op.directive in prompt for op in seed.operators)
    assert "Associative looseness: very loose" in prompt
    assert "realises this is a dream" in prompt
    assert "never questions" in SYSTEM_CRITIC_OFF and "Do not explain symbolism" in SYSTEM_CRITIC_OFF


def test_lucid_only_when_enabled_and_late_rem(cfg):
    c = dict(cfg, lucid={"prob_late_rem": 1.0})
    seed = bizarre.apply_operators(SEED, 1, np.random.default_rng(0), cfg["bizarre"])
    assert not maybe_lucid(seed, c, np.random.default_rng(0), False).lucid
    assert maybe_lucid(seed, c, np.random.default_rng(0), True).lucid
    early = DreamSeed("REM", "REM_early", 1, 0, seed.fragments, seed.operators)
    assert not maybe_lucid(early, c, np.random.default_rng(0), True).lucid


class FakeLLM:
    """Answers narrator calls with a report whose length follows the requested range."""

    def __init__(self):
        self.calls = []
        self.messages = self.beta = self
        self.beta.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        prompt = kw["messages"][0]["content"]
        lo = int(prompt.split("Length: ")[1].split("-")[0])
        payload = {"narrative": " ".join(["word"] * lo), "emotional_tone": {"valence": -0.2, "arousal": 0.8}}
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(payload))])


def test_llm_narrator_request(cfg):
    fake = FakeLLM()
    n = LLMNarrator(cfg, client=fake).narrate(SEED)
    call = fake.calls[0]
    assert "temperature" not in call            # Sonnet 5.5 rejects non-default sampling params
    assert call["system"] == SYSTEM_CRITIC_OFF
    assert len(n.narrative.split()) == cfg["stages"]["REM_late"]["words"][0]
    assert (n.valence, n.arousal) == (-0.2, 0.8)
    cfg["llm"]["send_temperature"] = True
    LLMNarrator(cfg, client=fake).narrate(SEED)
    assert fake.calls[-1]["temperature"] == cfg["stages"]["REM_late"]["temperature"]


# ---------------------------------------------------------------- full night

@pytest.fixture
def memory(cfg):
    from dream_engine.ingest.extract import HeuristicExtractor
    from dream_engine.ingest.parse_log import parse_log
    from dream_engine.memory.embed import HashingEmbedder
    from dream_engine.memory.store import Fragment

    ext = HeuristicExtractor(cfg)
    emb = HashingEmbedder(cfg["embedding"]["hashing_dim"], cfg["embedding"]["hashing_ngram_weight"])
    out = []
    for log in sorted(LOGS.iterdir()):
        for i, f in enumerate(ext.extract(parse_log(log))):
            out.append(Fragment(f"{log.stem}_{i}", f.text, f.type, f.entities, f.timestamp, f.valence, f.arousal,
                                emb.embed([f.text])[0], entity_kinds=f.entity_kinds))
    return out


def test_plan_is_deterministic_and_seed_dependent(cfg, memory):
    def sig(plan):
        return [(e.seed.profile, [f.id for f in e.seed.fragments], [o.name for o in e.seed.operators])
                for e in plan.episodes]

    a = plan_night(cfg, memory, date(2026, 10, 6), 42)
    assert sig(a) == sig(plan_night(cfg, memory, date(2026, 10, 6), 42))
    assert sig(a) != sig(plan_night(cfg, memory, date(2026, 10, 6), 43))


def test_plan_respects_stage_profiles(cfg, memory):
    for seed in range(5):
        plan = plan_night(cfg, memory, date(2026, 10, 6), seed)
        for ep in plan.episodes:
            prof = cfg["stages"][ep.seed.profile]
            replayed = [f for f in ep.seed.fragments if f.role == "replay"]
            assert prof["fragments"][0] <= len(replayed) <= prof["fragments"][1]
            assert prof["bizarre_ops"][0] <= len(ep.seed.operators) <= prof["bizarre_ops"][1]
            if ep.seed.stage != "REM":
                assert not ep.seed.pgo_events and not ep.seed.operators
            assert all(f.timestamp <= ep.clock for f in ep.seed.fragments)


def test_replays_within_a_night_are_penalised(cfg, memory):
    plan = plan_night(cfg, memory, date(2026, 10, 6), 42)
    counts = {}
    for ep in plan.episodes:
        for f in ep.seed.fragments:
            if f.role == "replay":
                counts[f.id] = counts.get(f.id, 0) + 1
    assert max(counts.values()) <= 4  # the replay_count penalty spreads replay over memories
    assert memory[0].replay_count == 0  # caller's fragments untouched


def test_run_night_and_report(cfg, memory, tmp_path):
    plan = plan_night(cfg, memory, date(2026, 10, 6), 42)
    night = run_night(cfg, plan, LLMNarrator(cfg, client=FakeLLM()), DryRunVisualizer(cfg))
    eps = night["episodes"]
    assert len(eps) == len(plan.episodes) and not any(e.get("error") for e in eps)
    assert {e["cycle"] for e in eps} == {1, 2, 3, 4, 5}
    for e in eps:
        lo, hi = cfg["stages"][e["profile"]]["words"]
        assert lo <= e["word_count"] <= hi
        assert e["seed_fragments"] and e["trace"]["replay"]
        n_img = len(e["image_prompts"])
        lo_i, hi_i = cfg["stages"][e["profile"]]["image_prompts"]
        assert lo_i <= n_img <= hi_i
    jp, mp = write_night(night, tmp_path)
    md = mp.read_text(encoding="utf-8")
    assert "## Hypnogram" in md and "Why these elements appeared" in md and "w=" in md
    assert json.loads(jp.read_text(encoding="utf-8"))["night_id"] == "2026-10-06"


def test_failed_episode_is_recorded_not_fatal(cfg, memory):
    class Boom:
        name = "boom"

        def narrate(self, seed):
            raise RuntimeError("api down")

    plan = plan_night(cfg, memory, date(2026, 10, 6), 1)
    night = run_night(cfg, plan, Boom())
    assert all("api down" in e["error"] for e in night["episodes"])
    assert "Generation failed" in night_markdown(night)


def test_cli_night_dry_run(tmp_path):
    db, out = str(tmp_path / "m.db"), tmp_path / "nights"
    r = CliRunner()
    for log in ["2026-08-20.md", "2026-10-04.md", "2026-10-05.md"]:
        assert r.invoke(main, ["--db", db, "ingest", str(LOGS / log), "--extractor", "heuristic",
                               "--embedder", "hashing"]).exit_code == 0
    res = r.invoke(main, ["--db", db, "night", "--date", "2026-10-06", "--dry-run", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "2026-10-06.dry.json").exists() and (out / "2026-10-06.dry.md").exists()
    res = r.invoke(main, ["--db", db, "episode", "--stage", "REM", "--cycle", "4", "--date", "2026-10-06",
                          "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "Cycle 4 · REM_late" in res.output
