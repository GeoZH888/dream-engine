"""Phase 4: EEG features, hypnogram → night (synthetic data; the real run is `dream eeg train`)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from dream_engine.eeg.features import epoch_features, spindle_density
from dream_engine.eeg.hypnogram import segments_from_labels, smooth
from dream_engine.eeg.load import EpochedRecording
from dream_engine.night import plan_night
from dream_engine.sleep.scheduler import episode_slots

SF = 100.0
N = int(30 * SF)


def _sine(freq, amp, n=N, phase=0.0):
    t = np.arange(n) / SF
    return amp * np.sin(2 * np.pi * freq * t + phase)


def test_spindle_density_counts_bursts(cfg):
    rng = np.random.default_rng(0)
    epochs = rng.normal(0, 5, size=(6, N))
    for k in range(3):                       # epoch 2 gets 3 one-second 13 Hz bursts
        a = int((5 + 8 * k) * SF)
        epochs[2, a:a + int(SF)] += _sine(13, 40, int(SF))
    d = spindle_density(epochs, SF, cfg["eeg"]["spindle"])
    assert d[2] == 3 and d.sum() == 3


def test_features_separate_slow_waves_from_alpha(cfg):
    rng = np.random.default_rng(1)
    slow = np.stack([_sine(1.5, 80) + rng.normal(0, 3, N) for _ in range(20)])
    alpha = np.stack([_sine(10, 30) + rng.normal(0, 3, N) for _ in range(20)])
    eeg = np.concatenate([slow, alpha])[:, None, :].repeat(2, axis=1)
    eog = rng.normal(0, 5, size=(40, N))
    rec = EpochedRecording("synthetic", 0, SF, eeg, eog, ["N3"] * 20 + ["W"] * 20)
    X, names = epoch_features(rec, cfg)
    k = len(cfg["eeg"]["context_epochs"] * 2 * [0]) + 1
    assert X.shape == (40, len(names)) and len(names) % k == 0
    rd, ra = names.index("Fpz-Cz_rel_delta"), names.index("Fpz-Cz_rel_alpha")
    assert X[:20, rd].mean() > X[20:, rd].mean() and X[20:, ra].mean() > X[:20, ra].mean()
    assert "spindle_density@+1" in names


def test_smooth_removes_single_epoch_flicker():
    assert smooth(["N2", "N2", "REM", "N2", "N2"], 3) == ["N2"] * 5
    assert smooth(["N2", "REM", "REM", "REM", "N2"], 3)[1:4] == ["REM"] * 3


def _labels(spec):
    return [s for s, minutes in spec for _ in range(minutes * 2)]  # 2 epochs per minute


def test_segments_and_cycles_from_hypnogram(cfg):
    labels = _labels([("W", 20), ("N1", 5), ("N2", 20), ("N3", 30), ("N2", 15), ("REM", 10),
                      ("N2", 10), ("REM", 3), ("N2", 3), ("REM", 6),          # gaps < 15 min: same REM period
                      ("N2", 30), ("N3", 10), ("N2", 20), ("REM", 30),
                      ("N2", 10), ("W", 10)])                                 # short tail joins cycle 2
    segs = segments_from_labels(labels, cfg)
    assert segs[0].minute_start == 0 and segs[0].stage == "N1"               # trimmed to sleep onset
    assert all(a.minute_end == b.minute_start for a, b in zip(segs, segs[1:]))
    assert max(s.cycle for s in segs) == 2
    assert [s.cycle for s in segs if s.stage == "REM"] == [1, 1, 1, 2]
    assert segs[-1].stage == "W" and segs[-2].stage == "N2" and segs[-2].cycle == 2

    slots = episode_slots(cfg, segs)
    c1 = [s for s in slots if s.cycle == 1]
    assert [s.stage for s in c1] == ["N3", "N2", "REM"]
    assert c1[2].segment.minutes == 10                                       # longest REM in the cycle
    assert c1[1].segment.minute_end == c1[2].segment.minute_start            # the N2 right before it


def test_plan_night_follows_real_hypnogram(cfg):
    from conftest import frag
    from datetime import datetime, timedelta

    frags = [frag(f"f{i}", datetime(2026, 10, 5, 8) + timedelta(hours=i), arousal=0.5, emb=np.eye(12)[i])
             for i in range(12)]
    labels = _labels([("N2", 20), ("N3", 40), ("N2", 10), ("REM", 12), ("N2", 50), ("REM", 25), ("N2", 5)])
    segs = segments_from_labels(labels, cfg)
    plan = plan_night(cfg, frags, date(2026, 10, 6), 7, segments=segs, hypnogram_source="test")
    assert plan.segments is segs and plan.hypnogram_source == "test"
    rem = [e for e in plan.episodes if e.seed.stage == "REM"]
    assert [e.seed.minute_start for e in rem] == [s.minute_start for s in segs if s.stage == "REM"]
    for e in rem:
        assert all(e.slot.segment.minute_start <= p.minute < e.slot.segment.minute_end for p in e.seed.pgo_events)
