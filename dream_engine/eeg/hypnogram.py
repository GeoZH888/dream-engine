"""Real hypnogram → night timeline, so dreams follow an actual recorded night.

Epoch labels are smoothed (mode filter), trimmed to sleep onset, turned into
segments, and grouped into cycles: a cycle ends with each REM period (REM separated
by less than ``rem_cycle_gap_minutes`` counts as one period).
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from dream_engine.sleep.scheduler import Segment


def read_hypnogram(path: str | Path, cfg: dict[str, Any]) -> tuple[list[str], str]:
    """Epoch labels from a hypnogram CSV (ours), a Sleep-EDF Hypnogram EDF, or a PSG EDF
    (staged with the trained model). Returns (labels, description of the source)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8") as f:
            return [row["stage"] for row in csv.DictReader(f)], f"hypnogram CSV {path.name}"
    from dream_engine.eeg.load import Recording, SC_NAME, find_recordings, load_recording

    if path.name.endswith("Hypnogram.edf"):
        m = SC_NAME.search(path.name)
        recs = [r for r in find_recordings(path.parent) if m and r.name == m.group(0)[2:]]
        if not recs:
            raise ValueError(f"No PSG next to {path} to align the hypnogram with")
        labels = load_recording(recs[0], cfg).labels
        return [s or "W" for s in labels], f"expert hypnogram {path.name}"
    if path.name.endswith("PSG.edf"):
        import joblib

        from dream_engine.eeg.stager import predict_recording

        model_path = Path(cfg["eeg"]["data_dir"]) / "stager.joblib"
        if not model_path.exists():
            raise ValueError("No trained stager; run `dream eeg train` first.")
        pred, _ = predict_recording(joblib.load(model_path), Recording(path.stem, -1, -1, path, None), cfg)
        return pred, f"LightGBM staging of {path.name}"
    raise ValueError(f"Unsupported hypnogram file: {path}")


def smooth(labels: list[str], window: int) -> list[str]:
    """Mode filter; ties keep the original label."""
    if window <= 1:
        return list(labels)
    h = window // 2
    out = []
    for i, lab in enumerate(labels):
        counts = Counter(labels[max(0, i - h): i + h + 1])
        top = max(counts.values())
        out.append(lab if counts[lab] == top else counts.most_common(1)[0][0])
    return out


def segments_from_labels(labels: list[str], cfg: dict[str, Any]) -> list[Segment]:
    e = cfg["eeg"]
    ep_min = e["epoch_seconds"] / 60
    labs = smooth(labels, e["smoothing_epochs"])
    sleep = [i for i, s in enumerate(labs) if s != "W"]
    if not sleep:
        raise ValueError("The hypnogram contains no sleep")
    labs = labs[sleep[0]: sleep[-1] + 1]          # from sleep onset to the final awakening

    runs: list[list] = []                         # [stage, first_epoch, last_epoch_exclusive]
    for i, s in enumerate(labs):
        if runs and runs[-1][0] == s:
            runs[-1][2] = i + 1
        else:
            runs.append([s, i, i + 1])

    # cycle boundaries: end of each REM period (REM runs closer than the gap are one period)
    gap = e["rem_cycle_gap_minutes"] / ep_min
    rem_runs = [r for r in runs if r[0] == "REM"]
    period_ends: list[int] = []
    for r in rem_runs:
        if period_ends and r[1] - period_ends[-1] < gap:
            period_ends[-1] = r[2]
        else:
            period_ends.append(r[2])
    if period_ends and len(labs) - period_ends[-1] < e["min_final_cycle_minutes"] / ep_min:
        period_ends[-1] = len(labs)               # short tail joins the last cycle
    period_ends = period_ends or [len(labs)]

    segs = []
    for stage, a, b in runs:
        cycle = next((k + 1 for k, end in enumerate(period_ends) if a < end), len(period_ends) + 1)
        segs.append(Segment(int(round(a * ep_min)), int(round(b * ep_min)), stage, cycle))
    segs = [s for s in segs if s.minute_end > s.minute_start]
    for a, b in zip(segs, segs[1:]):              # keep contiguity after rounding
        b.minute_start = a.minute_end
    segs = [s for s in segs if s.minute_end > s.minute_start]
    segs.append(Segment(segs[-1].minute_end, segs[-1].minute_end + 2, "W", segs[-1].cycle))
    return segs
