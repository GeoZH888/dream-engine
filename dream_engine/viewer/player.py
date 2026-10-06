"""Dream player: plays each dream like a short film (text over generative visuals).

Only what the film needs is embedded: narratives, stage, timing, emotional tone and
the PGO burst positions (which become hard scene cuts). No external services.
Dreams rendered with `dream video` also get their MP4, copied next to the page.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from dream_engine.video.render import clip_name
from dream_engine.viewer.build import GROUP_NAMES, STANDALONE_HEAD, STANDALONE_TAIL, night_label

TEMPLATE = Path(__file__).with_name("player.html")
PLACEHOLDER = "/*__PLAYER__*/[]"


def _episode(e: dict[str, Any], videos: dict[str, str]) -> dict[str, Any] | None:
    if not e.get("narrative"):
        return None
    t = e["trace"]
    return {
        "cycle": e["cycle"], "stage": e["stage"], "profile": e["profile"], "clock": e["clock_time"],
        "minute": e["minute_start"], "narrative": e["narrative"],
        "tone": e.get("emotional_tone") or {"valence": 0.0, "arousal": 0.3},
        "pgo": [{"position": p["position"], "kind": p["kind"]} for p in t.get("pgo_events", [])],
        "ops": e.get("operators_applied", []), "lucid": bool(e.get("lucid")),
        "bizarreness": e.get("bizarreness_score"),
        "video": videos.get(clip_name(e)),
    }


def _videos(video_dir: Path | None, stem: str, page_dir: Path | None) -> dict[str, str]:
    """Rendered clips of one night, copied next to the page: {clip name: relative URL}."""
    src = video_dir / stem if video_dir else None
    if not src or not src.is_dir():
        return {}
    out = {}
    for clip in sorted(src.glob("c*_*.mp4")):
        rel = f"videos/{stem}/{clip.name}"
        if page_dir is not None:
            dst = page_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or dst.stat().st_mtime < clip.stat().st_mtime:
                shutil.copy2(clip, dst)
        out[clip.stem] = rel
    return out


def build_player(night_files: list[Path], video_dir: Path | None = None,
                 page_dir: Path | None = None) -> list[dict[str, Any]]:
    nights = []
    for p in night_files:
        n = json.loads(p.read_text(encoding="utf-8"))
        videos = _videos(video_dir, p.stem, page_dir)
        eps = [x for x in (_episode(e, videos) for e in n["episodes"]) if x]
        nights.append({
            "label": night_label(n), "group": GROUP_NAMES.get(p.parent.name, p.parent.name),
            "onset": n["sleep_onset"],
            "hypnogram": [[s["minute_start"], s["minute_end"], s["stage"]] for s in n["hypnogram"]],
            "episodes": eps,
        })
    return nights


def write_player(night_files: list[Path], out: Path, video_dir: Path | None = None) -> Path:
    html = TEMPLATE.read_text(encoding="utf-8")
    assert PLACEHOLDER in html
    payload = json.dumps(build_player(night_files, video_dir, out.parent), ensure_ascii=False).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(STANDALONE_HEAD + html.replace(PLACEHOLDER, payload) + STANDALONE_TAIL, encoding="utf-8")
    return out
