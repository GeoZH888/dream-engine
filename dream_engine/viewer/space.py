"""3D memory space: every fragment placed by meaning, each dream drawn as a path.

Fragments are projected from their embedding (e.g. 384-d MiniLM) to 3D with metric
MDS on cosine distance, so nearby points are memories with related meaning. The
projection is approximate; its faithfulness (rank correlation between the original
and the 3D distances) is reported on the page. Step distances shown for each dream
are measured in the full embedding space, not in the projection.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from dream_engine.memory.store import Fragment
from dream_engine.viewer.build import GROUP_NAMES, STANDALONE_HEAD, STANDALONE_TAIL, night_label

TEMPLATE = Path(__file__).with_name("space.html")
PLACEHOLDER = "/*__SPACE__*/{}"


def project_3d(emb: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    """Metric MDS on cosine distance → (coords scaled to radius ~10, rank correlation)."""
    from sklearn.manifold import MDS

    dist = np.clip(1.0 - emb @ emb.T, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:  # scikit-learn >= 1.8 names the precomputed input `metric`
            mds = MDS(n_components=3, metric="precomputed", init="random", n_init=8, random_state=seed)
        except TypeError:
            mds = MDS(n_components=3, dissimilarity="precomputed", n_init=8, random_state=seed)
        xyz = mds.fit_transform(dist)
    xyz -= xyz.mean(axis=0)
    xyz *= 10.0 / (np.abs(xyz).max() or 1.0)
    iu = np.triu_indices(len(emb), k=1)
    d3 = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    rho = float(spearmanr(dist[iu], d3[iu]).statistic) if len(emb) > 2 else 1.0
    return xyz, rho


def _episode(e: dict[str, Any], emb: dict[str, np.ndarray]) -> dict[str, Any]:
    t = e["trace"]
    replay = sorted(t["replay"], key=lambda r: r["step"])
    chain = [r["fragment_id"] for r in replay]
    steps = [round(float(1.0 - emb[a] @ emb[b]), 3) for a, b in zip(chain, chain[1:])]
    extras = [{"id": f["id"], "role": f["role"], "from": f["trace"].get("source_fragment")}
              for f in t["fragments"] if f["role"] != "replay"]
    return {
        "cycle": e["cycle"], "stage": e["stage"], "profile": e["profile"], "clock": e["clock_time"],
        "minute": e["minute_start"], "chain": chain, "modes": [r["mode"] for r in replay], "steps": steps,
        "extras": extras, "bizarreness": e.get("bizarreness_score"), "ops": e["operators_applied"],
        "first": (e.get("narrative") or "").split(". ")[0][:160],
    }


def build_space(fragments: list[Fragment], night_files: list[Path], seed: int, embedder: str) -> dict[str, Any]:
    emb = np.stack([f.embedding for f in fragments]).astype(np.float64)
    xyz, rho = project_3d(emb, seed)
    by_id = {f.id: v for f, v in zip(fragments, emb)}
    nights = []
    for p in night_files:
        n = json.loads(p.read_text(encoding="utf-8"))
        nights.append({"label": night_label(n), "group": GROUP_NAMES.get(p.parent.name, p.parent.name),
                       "source": n.get("hypnogram_source", "simulated"),
                       "episodes": [_episode(e, by_id) for e in n["episodes"]]})
    return {
        "embedder": embedder,
        "projection": {"method": "metric MDS on cosine distance", "rank_correlation": round(rho, 3)},
        "fragments": [{"id": f.id, "text": f.text, "type": f.type, "date": f.timestamp.strftime("%Y-%m-%d %H:%M"),
                       "valence": round(f.valence, 2), "arousal": round(f.arousal, 2),
                       "entities": f.entities, "xyz": [round(float(c), 3) for c in v]}
                      for f, v in zip(fragments, xyz)],
        "nights": nights,
    }


def write_space(fragments: list[Fragment], night_files: list[Path], out: Path, seed: int, embedder: str) -> Path:
    data = build_space(fragments, night_files, seed, embedder)
    html = TEMPLATE.read_text(encoding="utf-8")
    assert PLACEHOLDER in html
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(STANDALONE_HEAD + html.replace(PLACEHOLDER, payload) + STANDALONE_TAIL, encoding="utf-8")
    return out
