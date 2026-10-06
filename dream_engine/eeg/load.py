"""Sleep-EDF loading: PSG signals + expert hypnogram → 30-s epochs with stage labels.

Stages follow AASM: Rechtschaffen & Kales stages 3 and 4 are merged into N3.
Movement time and unscored epochs are dropped from training (label ``None``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}
ANNOTATION_STAGE = {
    "Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM",
}
SC_NAME = re.compile(r"SC4(\d\d)(\d)")  # SC4ssN: subject ss, night N


@dataclass
class Recording:
    name: str               # e.g. SC4001
    subject: int
    night: int
    psg: Path
    hypnogram: Path | None


@dataclass
class EpochedRecording:
    name: str
    subject: int
    sfreq: float
    eeg: np.ndarray         # (n_epochs, n_eeg_channels, n_samples)
    eog: np.ndarray         # (n_epochs, n_samples)
    labels: list[str | None]  # per epoch; None = unscored / movement


def find_recordings(data_dir: str | Path) -> list[Recording]:
    """Pair every *PSG.edf with its *Hypnogram.edf (same subject and night)."""
    data_dir = Path(data_dir)
    hyps = {SC_NAME.search(p.name).group(0): p for p in data_dir.rglob("*Hypnogram.edf") if SC_NAME.search(p.name)}
    out = []
    for psg in sorted(data_dir.rglob("*PSG.edf")):
        m = SC_NAME.search(psg.name)
        if m:
            out.append(Recording(m.group(0)[2:], int(m.group(1)), int(m.group(2)), psg, hyps.get(m.group(0))))
    return out


def hypnogram_labels(hyp_path: str | Path, n_epochs: int, epoch_s: float, offset_s: float = 0.0) -> list[str | None]:
    """Expert labels per epoch from a Hypnogram EDF, for epochs starting at ``offset_s``."""
    import mne

    ann = mne.read_annotations(str(hyp_path))
    labels: list[str | None] = [None] * n_epochs
    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        stage = ANNOTATION_STAGE.get(desc)
        first = int(round((onset - offset_s) / epoch_s))
        for i in range(max(first, 0), min(first + int(round(duration / epoch_s)), n_epochs)):
            labels[i] = stage
    return labels


def _sleep_bounds(labels: list[str | None]) -> tuple[int, int] | None:
    sleep = [i for i, s in enumerate(labels) if s not in (None, "W")]
    return (sleep[0], sleep[-1]) if sleep else None


def load_recording(rec: Recording, cfg: dict[str, Any], crop_to_sleep: bool = True) -> EpochedRecording:
    """Band-pass filter, cut into epochs, attach labels, and crop long wake at the ends."""
    import mne

    e = cfg["eeg"]
    raw = mne.io.read_raw_edf(str(rec.psg), include=e["eeg_channels"] + [e["eog_channel"]], preload=True,
                              verbose="error")
    raw.filter(*e["bandpass"], verbose="error")
    sfreq = raw.info["sfreq"]
    n_per = int(round(e["epoch_seconds"] * sfreq))
    data = raw.get_data(picks=e["eeg_channels"] + [e["eog_channel"]])
    n_epochs = data.shape[1] // n_per
    data = data[:, : n_epochs * n_per].reshape(data.shape[0], n_epochs, n_per).transpose(1, 0, 2)

    labels: list[str | None] = [None] * n_epochs
    if rec.hypnogram:
        labels = hypnogram_labels(rec.hypnogram, n_epochs, e["epoch_seconds"])
    lo, hi = 0, n_epochs - 1
    if crop_to_sleep and (b := _sleep_bounds(labels)):
        margin = int(e["wake_margin_minutes"] * 60 / e["epoch_seconds"])
        lo, hi = max(0, b[0] - margin), min(n_epochs - 1, b[1] + margin)
    sl = slice(lo, hi + 1)
    n_eeg = len(e["eeg_channels"])
    return EpochedRecording(rec.name, rec.subject, sfreq, data[sl, :n_eeg] * 1e6, data[sl, n_eeg] * 1e6,
                            labels[sl])
