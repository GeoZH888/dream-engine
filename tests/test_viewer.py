"""Phase 5: self-contained viewer."""

from __future__ import annotations

import json

from dream_engine.viewer.build import build_viewer, night_label


def _night(src="simulated"):
    return {"night_id": "2026-10-06", "seed": 42, "hypnogram_source": src, "config": {"big": 1},
            "episodes": [{"narrative": "a </script><script>alert(1)</script> b"}]}


def test_viewer_embeds_nights_safely():
    html = build_viewer([_night(), _night("hypnogram CSV 4001.csv")])
    assert "/*__NIGHTS__*/" not in html and "</script><script>alert" not in html
    data = html.split("const NIGHTS = ", 1)[1].split(";\n", 1)[0]
    nights = json.loads(data)  # JSON decodes the escaped "<\/" back to "</"
    assert len(nights) == 2 and "config" not in nights[0]
    assert nights[0]["episodes"][0]["narrative"].startswith("a </script>")


def test_night_labels():
    assert night_label(_night()) == "2026-10-06 · simulated · seed 42"
    assert night_label(_night("hypnogram CSV 4001.csv")) == "2026-10-06 · EEG SC4001"
