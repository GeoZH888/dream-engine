"""Readable night report (spec §7): hypnogram strip, then each episode with its trace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dream_engine.sleep.scheduler import Segment, hypnogram_text

ROLE = {"replay": "replay", "augment": "noisy recall", "pgo_injection": "PGO intrusion"}


def night_markdown(night: dict[str, Any]) -> str:
    segs = [Segment(**s) for s in night["hypnogram"]]
    eps = night["episodes"]
    L = [f"# Night {night['night_id']}",
         "",
         f"Sleep onset {night['sleep_onset'].replace('T', ' ')} · seed {night['seed']} · "
         f"narrator `{night['narrator']}` · {len(eps)} episodes over {night['cycles']} cycles · "
         f"hypnogram: {night.get('hypnogram_source', 'simulated')}",
         "",
         "*Simulated dream-like content built from day-log memories. Not a real dream, not an interpretation.*",
         "",
         "## Hypnogram",
         "",
         "```",
         hypnogram_text(segs),
         "```",
         ""]
    rem = [s.minutes for s in segs if s.stage == "REM"]
    L += [f"REM periods (min): {', '.join(map(str, rem))}", ""]

    for i, e in enumerate(eps, 1):
        t = e["trace"]
        title = f"## {i}. Cycle {e['cycle']} · {e['profile']} · {e['clock_time']} (minute {e['minute_start']})"
        L += [title, ""]
        if e.get("error"):
            L += [f"> **Generation failed:** {e['error']}", ""]
        elif e["narrative"]:
            L += ["> " + e["narrative"].replace("\n", "\n> "), ""]
            tone = e["emotional_tone"]
            meta = [f"{e['word_count']} words", f"tone valence {tone['valence']:+.2f}, arousal {tone['arousal']:.2f}",
                    f"bizarreness prior {e['bizarreness_prior']:.2f}"]
            if e.get("bizarreness_score") is not None:
                meta.append(f"**rated bizarreness {e['bizarreness_score']:.2f}**")
            if e.get("lucid"):
                meta.append("**lucid**")
            L += ["*" + " · ".join(meta) + "*", ""]

        L += ["**Why these elements appeared**", ""]
        replay_by_id = {r["fragment_id"]: r for r in t["replay"]}
        for f in t["fragments"]:
            why = ROLE[f["role"]]
            r = replay_by_id.get(f["id"])
            if r:
                why = f"replay ({r['mode']}, p={r['probability']:.2f}"
                if "distance_from_prev" in r:
                    why += f", distance {r['distance_from_prev']:.2f} from previous"
                why += f"; {r['explanation']})"
            elif f["role"] == "augment":
                tr = f["trace"]
                why = (f"noisy recall: embedding of {tr['source_fragment']} perturbed (σ={tr['sigma']}), "
                       f"nearest stored memory")
            elif f["role"] == "pgo_injection":
                why = f"PGO burst at minute {f['trace']['minute']}: random fragment"
            L.append(f"- `{f['id']}` ({f['timestamp'][:10]}) {f['text']}  \n  ↳ {why}")
        if t["operators"]:
            L += ["", "Distortions applied:"]
            L += [f"- `{op['name']}`: {op['directive']}" for op in t["operators"]]
        if t["pgo_events"]:
            L += ["", "PGO bursts: " + ", ".join(
                f"min {p['minute']} → {p['kind'].replace('_', ' ')}" for p in t["pgo_events"])]
        if e["image_prompts"]:
            L += ["", "Image prompts:"]
            L += [f"{k}. {p}" for k, p in enumerate(e["image_prompts"], 1)]
        L.append("")
    return "\n".join(L)


def write_night(night: dict[str, Any], out_dir: str | Path, stem: str | None = None) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or night["night_id"]
    jp, mp = out / f"{stem}.json", out / f"{stem}.md"
    jp.write_text(json.dumps(night, indent=2, ensure_ascii=False), encoding="utf-8")
    mp.write_text(night_markdown(night), encoding="utf-8")
    return jp, mp


def eval_markdown(summary: dict[str, Any]) -> str:
    b, s, t = summary["bizarreness"], summary["separability"], summary["targets"]
    L = ["# Dream Engine evaluation", "",
         f"{summary['nights']} nights · {b['rated_reports']} rated reports · rater `{summary['rater']}`", "",
         "## Bizarreness (blind LLM rater, Hobson categories)", "",
         "| | n | mean |", "|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['mean']:.3f} |" for k, v in b["by_profile"].items()]
    L += ["", f"- REM mean **{b['rem_mean']}** vs NREM mean **{b['nrem_mean']}**",
          f"- REM > NREM in **{b['nights_rem_higher']} / {b['nights']}** nights"]
    if "mann_whitney_p_rem_greater" in b:
        L.append(f"- Mann–Whitney U (REM > NREM), one-sided p = {b['mann_whitney_p_rem_greater']:.2e}")
    L += ["", "## Stage separability (NREM vs REM from narrative embeddings)", ""]
    if "error" in s:
        L.append(f"Not computed: {s['error']}")
    else:
        L += [f"- Logistic regression on `{s['embedder']}` embeddings, {s['folds']}-fold CV"
              f"{' grouped by night' if s['grouped_by_night'] else ''}: "
              f"**accuracy {s['accuracy']:.3f}** (balanced {s['balanced_accuracy']:.3f})",
              f"- Baselines: majority class {s['baseline_majority_accuracy']:.3f}, "
              f"word count only {s['baseline_wordcount_accuracy']:.3f}"]
    L += ["", "## Phase 3 acceptance", "",
          f"- Bizarreness REM > NREM on every night: **{'PASS' if t['bizarreness_pass'] else 'FAIL'}**",
          f"- Separability ≥ {t['separability_target']}: **{'PASS' if t['separability_pass'] else 'FAIL'}**", ""]
    return "\n".join(L)
