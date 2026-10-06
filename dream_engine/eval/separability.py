"""Stage separability (spec §4.7) and the REM-vs-NREM bizarreness comparison.

A logistic-regression classifier on narrative embeddings should tell NREM-generated
from REM-generated reports (target accuracy ≥ 0.75). Folds are grouped by night, so a
night's reports are never split between train and test. Two baselines are reported
next to it: the majority class, and a classifier on word count alone (REM reports are
longer by design, so this shows how much the embeddings add beyond length).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def labelled_episodes(nights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for n in nights:
        for e in n["episodes"]:
            if e.get("narrative"):
                out.append({**e, "label": int(e["stage"] == "REM"), "group": f"{n['night_id']}#{n['seed']}"})
    return out


def _cv_predict(X: np.ndarray, y: np.ndarray, groups: list[str], folds: int, seed: int) -> np.ndarray:
    n_groups = len(set(groups))
    if n_groups >= folds:
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        split = list(cv.split(X, y, groups))
    else:
        cv = StratifiedKFold(n_splits=min(folds, int(min(np.bincount(y)))), shuffle=True, random_state=seed)
        split = list(cv.split(X, y))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return cross_val_predict(model, X, y, cv=split)


def separability(nights: list[dict[str, Any]], embedder: Any, folds: int, seed: int) -> dict[str, Any]:
    eps = labelled_episodes(nights)
    y = np.array([e["label"] for e in eps])
    if len(set(y)) < 2 or min(np.bincount(y)) < 2:
        return {"error": "need at least 2 REM and 2 NREM reports"}
    groups = [e["group"] for e in eps]
    X = embedder.embed([e["narrative"] for e in eps])
    pred = _cv_predict(X, y, groups, folds, seed)
    words = np.array([[np.log1p(e["word_count"])] for e in eps])
    pred_len = _cv_predict(words, y, groups, folds, seed)
    majority = max(np.mean(y), 1 - np.mean(y))
    return {
        "n_reports": len(eps), "n_rem": int(y.sum()), "n_nrem": int(len(y) - y.sum()),
        "folds": folds, "grouped_by_night": len(set(groups)) >= folds, "embedder": embedder.name,
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "baseline_majority_accuracy": round(float(majority), 4),
        "baseline_wordcount_accuracy": round(float(accuracy_score(y, pred_len)), 4),
    }


def bizarreness_comparison(nights: list[dict[str, Any]]) -> dict[str, Any]:
    eps = [e for e in labelled_episodes(nights) if e.get("bizarreness_score") is not None]
    rem = [e["bizarreness_score"] for e in eps if e["label"] == 1]
    nrem = [e["bizarreness_score"] for e in eps if e["label"] == 0]
    by_profile: dict[str, list[float]] = {}
    for e in eps:
        by_profile.setdefault(e["profile"], []).append(e["bizarreness_score"])

    per_night = []
    for n in nights:
        r = [e["bizarreness_score"] for e in n["episodes"] if e["stage"] == "REM" and e.get("bizarreness_score") is not None]
        nr = [e["bizarreness_score"] for e in n["episodes"] if e["stage"] != "REM" and e.get("bizarreness_score") is not None]
        if r and nr:
            per_night.append({"night": f"{n['night_id']}#{n['seed']}", "rem_mean": round(float(np.mean(r)), 3),
                              "nrem_mean": round(float(np.mean(nr)), 3), "rem_higher": bool(np.mean(r) > np.mean(nr))})
    out = {
        "rated_reports": len(eps),
        "rem_mean": round(float(np.mean(rem)), 4) if rem else None,
        "nrem_mean": round(float(np.mean(nrem)), 4) if nrem else None,
        "by_profile": {k: {"n": len(v), "mean": round(float(np.mean(v)), 4)} for k, v in sorted(by_profile.items())},
        "nights": len(per_night),
        "nights_rem_higher": sum(p["rem_higher"] for p in per_night),
        "per_night": per_night,
    }
    if len(rem) >= 2 and len(nrem) >= 2:
        out["mann_whitney_p_rem_greater"] = float(mannwhitneyu(rem, nrem, alternative="greater").pvalue)
    return out
