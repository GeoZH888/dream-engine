"""Dream videos: sentence timing, scene layout, PGO cuts, and a short end-to-end encode."""

from __future__ import annotations

import shutil
import subprocess

import pytest
from PIL import Image

from dream_engine.video.images import CachedImages
from dream_engine.video.render import encode, frames, pgo_cut_sentences, plan, split_sentences


class StubImages:
    name = "stub"

    def __init__(self):
        self.calls = 0

    def render(self, prompt, seed):
        self.calls += 1
        return Image.new("RGB", (64, 36), (seed % 255, 80, 140))


EP = {"cycle": 3, "stage": "REM", "profile": "REM_late", "clock_time": "03:25", "minute_start": 205,
      "bizarreness_score": 0.6,
      "narrative": "I run across the hall. The board flickers orange. Grandmother waits at the gate. "
                   "The gate is water. I swim through it. Then I am on the bridge.",
      "image_prompts": ["a station hall", "a gate made of water"],
      "trace": {"pgo_events": [{"position": 0.5, "kind": "scene_transition"}]}}


def test_split_sentences_and_long_sentences():
    assert split_sentences("I run. Then I fly! Do I?") == ["I run.", "Then I fly!", "Do I?"]
    long = ", ".join(["the river turns gold and the violin keeps playing"] * 6) + "."
    parts = split_sentences(long)
    assert len(parts) > 1 and all(len(p.split()) <= 40 for p in parts)


def test_pgo_cuts_map_to_sentences():
    assert pgo_cut_sentences([{"position": 0.5}, {"position": 0.01}, {"position": 0.99}], 6) == {3, 1, 5}
    assert pgo_cut_sentences([{"position": 0.5}], 1) == set()


def test_plan_layout(cfg, tmp_path):
    cfg["video"]["frame_size"] = [160, 90]
    cfg["video"]["image_size"] = [64, 36]
    stub = StubImages()
    tl, audio = plan(EP, cfg, CachedImages(stub, tmp_path / "cache"), 7, tmp_path / "work", voice=False, log=lambda s: None)
    assert audio is None and tl.meta == {"sentences": 6, "scenes": 2, "pgo_cuts": 1, "narrated": False}
    assert [s.start for s in tl.shots][0] == 0 and tl.shots[-1].end == pytest.approx(tl.duration)
    assert tl.shots[1].start == pytest.approx(tl.subs[3].start)          # scene 2 starts at sentence 4 of 6
    assert tl.shots[1].hard_in                                            # ...where the PGO burst lands
    assert tl.flashes == [pytest.approx(tl.subs[3].start)]
    assert all(a.end == pytest.approx(b.start) for a, b in zip(tl.subs, tl.subs[1:]))
    # cached: planning again does not regenerate images
    plan(EP, cfg, CachedImages(stub, tmp_path / "cache"), 7, tmp_path / "work", voice=False, log=lambda s: None)
    assert stub.calls == 2


def test_nrem_uses_ambient_field(cfg, tmp_path):
    cfg["video"]["frame_size"] = [160, 90]
    cfg["video"]["image_size"] = [64, 36]
    nrem = {**EP, "stage": "N3", "profile": "N3", "narrative": "A maze. Grey page.", "image_prompts": [],
            "trace": {"pgo_events": []}}
    tl, _ = plan(nrem, cfg, CachedImages(StubImages(), tmp_path / "c"), 1, tmp_path / "w", voice=False, log=lambda s: None)
    assert len(tl.shots) == 1 and tl.meta["scenes"] == 0
    first = next(frames(tl, cfg))
    assert first.size == (160, 90)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_encode_small_clip(cfg, tmp_path):
    v = cfg["video"]
    v.update(frame_size=[160, 96], image_size=[64, 36], fps=8, title_s=0.5, tail_s=0.5, reading_wps=20)
    tl, audio = plan(EP, cfg, CachedImages(StubImages(), tmp_path / "c"), 3, tmp_path / "w", voice=False, log=lambda s: None)
    out = encode(tl, audio, cfg, tmp_path / "clip.mp4")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True).stdout.split()
    assert probe == ["h264", "aac"]
