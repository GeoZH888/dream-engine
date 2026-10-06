"""Sleep stager: LightGBM on epoch features, evaluated with subject-grouped CV.

Cross-validation folds never mix a subject's nights between train and test, so the
reported accuracy is for unseen people. Out-of-fold predicted hypnograms are saved:
they are honest held-out stagings that can drive a night (`dream night --hypnogram`).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dream_engine.eeg.features import epoch_features
from dream_engine.eeg.load import STAGE_INDEX, STAGES, Recording, load_recording


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray            # stage index
    subject: np.ndarray
    recording: np.ndarray    # recording name per epoch
    feature_names: list[str]


def build_dataset(recs: list[Recording], cfg: dict[str, Any], cache: Path | None = None,
                  log: Callable[[str], None] = print) -> Dataset:
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["recordings_used"]) == [r.name for r in recs]:
            return Dataset(z["X"], z["y"], z["subject"], z["recording"], list(z["feature_names"]))
    Xs, ys, subs, names_rec, names = [], [], [], [], None
    for i, rec in enumerate(recs, 1):
        er = load_recording(rec, cfg)
        X, names = epoch_features(er, cfg)
        keep = np.array([lab is not None for lab in er.labels])
        Xs.append(X[keep])
        ys.append(np.array([STAGE_INDEX[lab] for lab in er.labels if lab is not None]))
        subs.append(np.full(keep.sum(), rec.subject))
        names_rec.append(np.full(keep.sum(), rec.name))
        log(f"  [{i}/{len(recs)}] {rec.name}: {keep.sum()} scored epochs")
    ds = Dataset(np.concatenate(Xs), np.concatenate(ys), np.concatenate(subs), np.concatenate(names_rec), names)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=ds.X, y=ds.y, subject=ds.subject, recording=ds.recording,
                            feature_names=np.array(names), recordings_used=np.array([r.name for r in recs]))
    return ds


def make_model(cfg: dict[str, Any]):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(objective="multiclass", random_state=cfg["seed"], verbose=-1, **cfg["eeg"]["lightgbm"])


def cross_validate(ds: Dataset, cfg: dict[str, Any], log: Callable[[str], None] = print) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
    from sklearn.model_selection import GroupKFold

    folds = min(cfg["eeg"]["cv_folds"], len(set(ds.subject)))
    oof = np.empty_like(ds.y)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(ds.X, ds.y, ds.subject), 1):
        model = make_model(cfg).fit(ds.X[tr], ds.y[tr])
        oof[te] = model.predict(ds.X[te])
        log(f"  fold {k}/{folds}: subjects {sorted(set(ds.subject[te].tolist()))} "
            f"accuracy {accuracy_score(ds.y[te], oof[te]):.3f}")
    per_rec = {r: round(float(accuracy_score(ds.y[ds.recording == r], oof[ds.recording == r])), 4)
               for r in sorted(set(ds.recording.tolist()))}
    return {
        "epochs": int(len(ds.y)), "subjects": int(len(set(ds.subject))), "recordings": len(per_rec),
        "folds": folds, "grouped_by": "subject",
        "accuracy": round(float(accuracy_score(ds.y, oof)), 4),
        "macro_f1": round(float(f1_score(ds.y, oof, average="macro")), 4),
        "cohen_kappa": round(float(cohen_kappa_score(ds.y, oof)), 4),
        "per_stage_f1": dict(zip(STAGES, [round(float(x), 4) for x in
                                          f1_score(ds.y, oof, average=None, labels=range(len(STAGES)))])),
        "confusion_matrix": {"labels": STAGES,
                             "rows_true_cols_pred": confusion_matrix(ds.y, oof, labels=range(len(STAGES))).tolist()},
        "per_recording_accuracy": per_rec,
        "class_counts": dict(zip(STAGES, np.bincount(ds.y, minlength=len(STAGES)).tolist())),
        "_oof": oof,
    }


def save_hypnogram_csv(path: Path, labels: list[str], truth: list[str | None] | None = None,
                       epoch_s: float = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "seconds", "stage"] + (["expert"] if truth else []))
        for i, s in enumerate(labels):
            w.writerow([i, int(i * epoch_s), s] + ([truth[i] or ""] if truth else []))


def predict_recording(model: Any, rec: Recording, cfg: dict[str, Any]) -> tuple[list[str], list[str | None]]:
    """Stage a whole (cropped) recording; returns (predicted, expert-or-None) per epoch."""
    er = load_recording(rec, cfg)
    X, _ = epoch_features(er, cfg)
    return [STAGES[i] for i in model.predict(X)], er.labels


def staging_markdown(res: dict[str, Any], target: float) -> str:
    cm = res["confusion_matrix"]["rows_true_cols_pred"]
    L = ["# Sleep staging (Sleep-EDF, LightGBM)", "",
         f"{res['recordings']} recordings · {res['subjects']} subjects · {res['epochs']} scored 30-s epochs · "
         f"{res['folds']}-fold CV grouped by subject", "",
         f"- **Accuracy {res['accuracy']:.3f}** (target ≥ {target}: "
         f"**{'PASS' if res['accuracy'] >= target else 'FAIL'}**)",
         f"- Macro F1 {res['macro_f1']:.3f} · Cohen's κ {res['cohen_kappa']:.3f}", "",
         "| stage | epochs | F1 |", "|---|---|---|"]
    L += [f"| {s} | {res['class_counts'][s]} | {res['per_stage_f1'][s]:.3f} |" for s in STAGES]
    L += ["", "Confusion matrix (rows = expert, columns = predicted):", "",
          "| | " + " | ".join(STAGES) + " |", "|---" * (len(STAGES) + 1) + "|"]
    L += [f"| **{s}** | " + " | ".join(str(v) for v in row) + " |" for s, row in zip(STAGES, cm)]
    L += ["", "Per-recording accuracy: " + ", ".join(f"{k} {v:.2f}" for k, v in res["per_recording_accuracy"].items()), ""]
    return "\n".join(L)
