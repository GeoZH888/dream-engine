"""Phase 5: bundle night JSON files into one self-contained HTML viewer.

The page has no server and no external data: the nights are embedded in the HTML,
so the file can be opened locally or published as a shareable page.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

TEMPLATE = Path(__file__).with_name("template.html")
PLACEHOLDER = "/*__NIGHTS__*/[]"


def night_label(night: dict[str, Any]) -> str:
    src = night.get("hypnogram_source", "simulated")
    if src == "simulated":
        return f"{night['night_id']} · simulated · seed {night['seed']}"
    m = re.search(r"(SC)?4\d{3}", src)
    return f"{night['night_id']} · EEG {('SC' + m.group(0).removeprefix('SC')) if m else 'recording'}"


GROUP_NAMES = {"nights": "Featured nights", "study": "20-night study"}


STANDALONE_HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                   '</head>\n<body>\n')
STANDALONE_TAIL = "\n</body>\n</html>\n"


def build_viewer(nights: list[dict[str, Any]], groups: list[str] | None = None, standalone: bool = True,
                 cfg: dict[str, Any] | None = None, log=lambda s: None) -> str:
    """``groups`` (one per night) sorts nights into sections of the night picker.

    ``standalone`` wraps the page in a full HTML document (for opening locally or static
    hosting); ``False`` returns the bare fragment, for hosts that add their own skeleton.
    With ``cfg``, each dream gets its brain mechanism map, and nights driven by a
    Sleep-EDF recording get their real EEG (when the recording is on disk).
    """
    slim = []
    for i, n in enumerate(nights):
        n = {k: v for k, v in n.items() if k != "config"}  # tuning snapshot is not shown
        n["label"] = night_label(n)
        n["group"] = groups[i] if groups else "Nights"
        if cfg is not None:
            from dream_engine.viewer.brain import brain_activity
            from dream_engine.viewer.eeg_data import night_eeg

            for ep in n["episodes"]:
                ep["brain"] = brain_activity(ep, cfg)
            n["eeg"] = night_eeg(n, cfg)
            if n["eeg"]:
                log(f"  real EEG attached to {n['label']} ({n['eeg']['recording']})")
        slim.append(n)
    data = json.dumps(slim, ensure_ascii=False).replace("</", "<\\/")  # never close the <script> early
    html = TEMPLATE.read_text(encoding="utf-8")
    assert PLACEHOLDER in html
    html = html.replace(PLACEHOLDER, data)
    return STANDALONE_HEAD + html + STANDALONE_TAIL if standalone else html


def write_viewer(night_files: list[Path], out: Path, standalone: bool = True, cfg: dict[str, Any] | None = None,
                 log=lambda s: None) -> Path:
    nights = [json.loads(p.read_text(encoding="utf-8")) for p in night_files]
    groups = [GROUP_NAMES.get(p.parent.name, p.parent.name) for p in night_files]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_viewer(nights, groups, standalone, cfg, log), encoding="utf-8")
    return out
