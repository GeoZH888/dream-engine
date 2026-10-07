"""Real EEG for nights driven by a Sleep-EDF recording, packaged for the viewer.

For each dream: a 30-s window of the recorded signals from the middle of the dream's
sleep segment (EEG Fpz-Cz, EEG Pz-Oz, horizontal EOG), and the segment's relative
band power. For the night: a spectrogram of Fpz-Cz, one column per 30-s epoch,
aligned with the hypnogram (minute 0 = sleep onset).

The dream *content* is not derived from these signals; the recording sets only when
each dream happens and which stage it is.
"""

from __future__ import annotations

import base64
import csv
import io
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import decimate, welch

from dream_engine.eeg.hypnogram import smooth
from dream_engine.eeg.load import find_recordings, load_recording

SPEC_FMAX = 30.0


def recording_for(night: dict[str, Any], cfg: dict[str, Any]) -> tuple[Any, Path] | None:
    """The Sleep-EDF recording and hypnogram CSV behind a night, if it has one on disk."""
    src = night.get("hypnogram_source", "simulated")
    m = re.search(r"(\d{4})\.csv", src)
    if not m:
        return None
    data_dir = Path(cfg["eeg"]["data_dir"])
    csv_path = data_dir / "heldout" / f"{m.group(1)}.csv"
    recs = [r for r in find_recordings(data_dir) if r.name == m.group(1)]
    return (recs[0], csv_path) if recs and csv_path.exists() else None


def onset_epoch(csv_path: Path, cfg: dict[str, Any]) -> int:
    """Epoch index (in the cropped recording) that the night's minute 0 corresponds to."""
    with open(csv_path, encoding="utf-8") as f:
        labels = [row["stage"] for row in csv.DictReader(f)]
    labs = smooth(labels, cfg["eeg"]["smoothing_epochs"])
    return next(i for i, s in enumerate(labs) if s != "W")


def _png(arr: np.ndarray) -> str:
    """2-D array in [0, 1] (rows = frequency, top = high) → base64 PNG with a perceptual colour map."""
    from matplotlib import colormaps
    from PIL import Image

    rgb = (colormaps["magma"](np.clip(arr, 0, 1))[..., :3] * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def night_eeg(night: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any] | None:
    found = recording_for(night, cfg)
    if not found:
        return None
    rec, csv_path = found
    e = cfg["eeg"]
    er = load_recording(rec, cfg)
    epoch_s = e["epoch_seconds"]
    ep_per_min = 60 / epoch_s
    first = onset_epoch(csv_path, cfg)
    total_min = night["hypnogram"][-1]["minute_end"]
    last = min(len(er.labels), first + int(total_min * ep_per_min))
    sf = er.sfreq

    # night spectrogram (Fpz-Cz), log power normalised per frequency row for contrast
    f, p = welch(er.eeg[first:last, 0], fs=sf, nperseg=int(4 * sf), axis=-1)
    keep = (f >= 0.5) & (f <= SPEC_FMAX)
    logp = np.log10(p[:, keep] + 1e-12).T[::-1]          # rows: high → low frequency
    lo, hi = np.percentile(logp, [5, 99], axis=1, keepdims=True)
    spec = (logp - lo) / (hi - lo + 1e-9)

    bands = e["bands"]
    episodes = []
    for ep in night["episodes"]:
        seg = next((s for s in night["hypnogram"]
                    if s["minute_start"] == ep["minute_start"] and s["stage"] == ep["stage"]), None)
        if seg is None:
            episodes.append(None)
            continue
        a = first + int(seg["minute_start"] * ep_per_min)
        b = min(last, first + int(seg["minute_end"] * ep_per_min))
        mid = (a + b) // 2
        q = 2 if sf >= 100 else 1                          # 100 Hz → 50 Hz keeps spindles (12–15 Hz)
        chans = {
            "EEG Fpz-Cz": er.eeg[mid, 0], "EEG Pz-Oz": er.eeg[mid, 1], "EOG horizontal": er.eog[mid],
        }
        traces = {k: [int(round(v)) for v in (decimate(x, q, zero_phase=True) if q > 1 else x)] for k, x in chans.items()}
        fs, ps = welch(er.eeg[a:max(b, a + 1), 0], fs=sf, nperseg=int(4 * sf), axis=-1)
        ps = ps.mean(axis=0)
        absb = {name: float(ps[(fs >= lo_) & (fs < hi_)].sum()) for name, (lo_, hi_) in bands.items()}
        tot = sum(absb.values()) or 1.0
        episodes.append({
            "window_minute": round((mid - first) / ep_per_min, 1), "fs": sf / q, "seconds": epoch_s,
            "traces": traces, "band_rel": {k: round(v / tot, 3) for k, v in absb.items()},
            "expert_stage": er.labels[mid],
        })
    return {
        "recording": f"Sleep-EDF SC{rec.name}", "channels": list(episodes[0]["traces"]) if any(episodes) else [],
        "spectrogram": {"png": _png(spec), "fmin": 0.5, "fmax": SPEC_FMAX, "minutes": round((last - first) / ep_per_min, 1)},
        "episodes": episodes,
    }
