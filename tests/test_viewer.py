"""Phase 5: self-contained viewer."""

from __future__ import annotations

import json

import pytest

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


def test_standalone_and_fragment():
    full = build_viewer([_night()])
    assert full.startswith("<!doctype html>") and full.rstrip().endswith("</html>")
    frag = build_viewer([_night()], standalone=False)
    assert frag.startswith("<title>") and "<!doctype" not in frag


def test_space_projection_and_paths(tmp_path):
    from datetime import datetime

    import numpy as np

    from conftest import frag
    from dream_engine.viewer.space import build_space, project_3d

    rng = np.random.default_rng(0)
    emb = rng.normal(size=(12, 16))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    xyz, rho = project_3d(emb, seed=0)
    assert xyz.shape == (12, 3) and np.abs(xyz).max() == pytest.approx(10.0) and rho > 0.5

    frags = [frag(f"f{i}", datetime(2026, 10, 5, 9 + i), emb=emb[i]) for i in range(3)]
    ep = {"cycle": 1, "stage": "REM", "profile": "REM_early", "clock_time": "00:30", "minute_start": 90,
          "operators_applied": [], "bizarreness_score": 0.5, "narrative": "I run. Then I fly.",
          "trace": {"replay": [{"step": 1, "fragment_id": "f1", "mode": "association"},
                               {"step": 0, "fragment_id": "f0", "mode": "seed"}],
                    "fragments": [{"id": "f0", "role": "replay", "trace": {}}, {"id": "f1", "role": "replay", "trace": {}},
                                  {"id": "f2", "role": "augment", "trace": {"source_fragment": "f0"}}]}}
    p = tmp_path / "nights" / "2026-10-06.json"
    p.parent.mkdir()
    p.write_text(json.dumps({"night_id": "2026-10-06", "seed": 1, "episodes": [ep]}), encoding="utf-8")
    d = build_space(frags, [p], seed=0, embedder="test")
    e = d["nights"][0]["episodes"][0]
    assert e["chain"] == ["f0", "f1"]                       # replay order, not trace order
    assert e["steps"] == [pytest.approx(round(float(1 - emb[0] @ emb[1]), 3))]
    assert e["extras"] == [{"id": "f2", "role": "augment", "from": "f0"}]
    assert d["nights"][0]["group"] == "Featured nights" and len(d["fragments"]) == 3


def test_player_data(tmp_path):
    from dream_engine.viewer.player import build_player, write_player

    ep = {"cycle": 2, "stage": "REM", "profile": "REM_early", "clock_time": "01:56", "minute_start": 176,
          "narrative": "I run. I fly.", "emotional_tone": {"valence": -0.2, "arousal": 0.8}, "operators_applied": ["scene_jump"],
          "lucid": False, "bizarreness_score": 0.6,
          "trace": {"pgo_events": [{"minute": 180.0, "position": 0.4, "kind": "scene_transition"}]}}
    failed = {**ep, "narrative": None}
    p = tmp_path / "study" / "2026-10-06_s1.json"
    p.parent.mkdir()
    p.write_text(json.dumps({"night_id": "2026-10-06", "seed": 1, "sleep_onset": "2026-10-05T23:00",
                             "hypnogram": [{"minute_start": 0, "minute_end": 200, "stage": "N2", "cycle": 1}],
                             "episodes": [ep, failed]}), encoding="utf-8")
    (n,) = build_player([p])
    assert n["group"] == "20-night study" and n["hypnogram"] == [[0, 200, "N2"]]
    assert len(n["episodes"]) == 1                          # failed episodes are skipped
    assert n["episodes"][0]["pgo"] == [{"position": 0.4, "kind": "scene_transition"}]
    html = write_player([p], tmp_path / "player.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and "/*__PLAYER__*/" not in html
    assert n["episodes"][0]["video"] is None

    clips = tmp_path / "video" / "2026-10-06_s1"
    clips.mkdir(parents=True)
    (clips / "c2_REM_early_0156.mp4").write_bytes(b"mp4")
    page = tmp_path / "site" / "player.html"
    write_player([p], page, tmp_path / "video")
    (n,) = build_player([p], tmp_path / "video")
    assert n["episodes"][0]["video"] == "videos/2026-10-06_s1/c2_REM_early_0156.mp4"
    assert (page.parent / "videos" / "2026-10-06_s1" / "c2_REM_early_0156.mp4").read_bytes() == b"mp4"
