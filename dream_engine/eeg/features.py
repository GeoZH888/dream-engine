"""Per-epoch features for sleep staging (spec §4.8).

- EEG (each channel): log absolute and relative band power δ/θ/α/σ/β (Welch),
  log total power, Hjorth mobility and complexity.
- Spindle density: σ-band RMS-envelope events of plausible duration per epoch.
- EOG: log variance, relative δ power (slow rolling eye movements), and correlation
  with frontal EEG (rapid eye movements in REM are EOG-specific; N3 slow waves leak
  into EOG and correlate with EEG).
- Context: the features of ±k neighbouring epochs, since stages depend on context.

Features are z-scored within each recording (unsupervised), which removes
between-subject amplitude differences.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from dream_engine.eeg.load import EpochedRecording

EPS = 1e-12


def _band_powers(x: np.ndarray, sfreq: float, bands: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """x: (n_epochs, n_samples) → (abs band power (n, b), total power (n,))."""
    f, p = welch(x, fs=sfreq, nperseg=int(4 * sfreq), axis=-1)
    total_mask = (f >= min(b[0] for b in bands.values())) & (f < max(b[1] for b in bands.values()))
    df = f[1] - f[0]
    total = p[:, total_mask].sum(axis=1) * df
    absp = np.stack([p[:, (f >= lo) & (f < hi)].sum(axis=1) * df for lo, hi in bands.values()], axis=1)
    return absp, total


def _hjorth(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d1, d2 = np.diff(x, axis=-1), np.diff(x, n=2, axis=-1)
    v0, v1, v2 = x.var(axis=-1) + EPS, d1.var(axis=-1) + EPS, d2.var(axis=-1) + EPS
    mobility = np.sqrt(v1 / v0)
    return mobility, np.sqrt(v2 / v1) / mobility


def spindle_density(x: np.ndarray, sfreq: float, sp: dict[str, Any]) -> np.ndarray:
    """Spindle events per epoch on one channel; x: (n_epochs, n_samples)."""
    n_ep, n = x.shape
    sos = butter(4, sp["band"], btype="bandpass", fs=sfreq, output="sos")
    sig = sosfiltfilt(sos, x.reshape(-1))
    w = max(1, int(sp["rms_window_s"] * sfreq))
    env = np.sqrt(np.convolve(sig ** 2, np.ones(w) / w, mode="same"))
    above = env > env.mean() + sp["threshold_sd"] * env.std()
    edges = np.diff(np.concatenate([[0], above.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    dur = (ends - starts) / sfreq
    lo, hi = sp["duration_s"]
    ok = (dur >= lo) & (dur <= hi)
    counts = np.bincount((starts[ok] // n).clip(0, n_ep - 1), minlength=n_ep)
    return counts.astype(float)


def epoch_features(rec: EpochedRecording, cfg: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    e = cfg["eeg"]
    bands = e["bands"]
    cols, names = [], []
    for ci, ch in enumerate(e["eeg_channels"]):
        x = rec.eeg[:, ci]
        absp, total = _band_powers(x, rec.sfreq, bands)
        rel = absp / (total[:, None] + EPS)
        mob, comp = _hjorth(x)
        tag = ch.split()[-1]
        cols += [np.log(absp + EPS), rel, np.log(total + EPS)[:, None], mob[:, None], comp[:, None]]
        names += [f"{tag}_log_{b}" for b in bands] + [f"{tag}_rel_{b}" for b in bands] + \
                 [f"{tag}_log_total", f"{tag}_hjorth_mobility", f"{tag}_hjorth_complexity"]
    cols.append(spindle_density(rec.eeg[:, 0], rec.sfreq, e["spindle"])[:, None])
    names.append("spindle_density")

    eog = rec.eog
    absp, total = _band_powers(eog, rec.sfreq, bands)
    eeg0 = rec.eeg[:, 0]
    xc = ((eog - eog.mean(1, keepdims=True)) * (eeg0 - eeg0.mean(1, keepdims=True))).mean(1)
    corr = xc / (eog.std(1) * eeg0.std(1) + EPS)
    cols += [np.log(eog.var(axis=1) + EPS)[:, None], (absp[:, 0] / (total + EPS))[:, None], corr[:, None]]
    names += ["eog_log_var", "eog_rel_delta", "eog_eeg_corr"]

    X = np.concatenate(cols, axis=1)
    X = (X - X.mean(0)) / (X.std(0) + EPS)          # per-recording standardisation
    return _with_context(X, names, e["context_epochs"])


def _with_context(X: np.ndarray, names: list[str], k: int) -> tuple[np.ndarray, list[str]]:
    parts, all_names = [X], list(names)
    n = len(X)
    for off in [o for o in range(-k, k + 1) if o != 0]:
        idx = np.clip(np.arange(n) + off, 0, n - 1)
        parts.append(X[idx])
        all_names += [f"{nm}@{off:+d}" for nm in names]
    return np.concatenate(parts, axis=1), all_names
