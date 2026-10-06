"""Render a dream report as an MP4: scene images, camera motion, subtitles, narration.

Timeline of one dream:
  title card → sentences (each subtitle timed to its narration) → fade to black.
REM dreams show their image prompts as scenes, spread evenly over the sentences, with a
slow push-in on each. Scenes crossfade, except where a PGO burst falls: there the cut is
hard, with a white flash (the same burst positions the narrator was given). NREM dreams
have no images; they play over a dim, stage-coloured field.
"""

from __future__ import annotations

import math
import re
import subprocess
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from dream_engine.video import tts
from dream_engine.video.images import CachedImages

PROFILE_TITLE = {"N2": "N2 · light sleep", "N3": "N3 · deep sleep", "REM_early": "REM · early night",
                 "REM_late": "REM · late night"}
STAGE_RGB = {"N2": (91, 127, 176), "N3": (61, 94, 168), "REM": (207, 63, 98)}


def split_sentences(text: str, max_words: int = 34) -> list[str]:
    raw = re.split(r"(?<=[.!?…])\s+(?=[A-Z“\"'‘(])", re.sub(r"\s+", " ", text).strip())
    out: list[str] = []
    for s in raw:
        words = s.split(" ")
        if len(words) <= max_words:
            out.append(s)
            continue
        buf: list[str] = []
        for w in words:
            buf.append(w)
            if len(buf) >= 18 and re.search(r"[,;:]$", w):
                out.append(" ".join(buf))
                buf = []
        if buf:
            out.append(" ".join(buf))
    return [s for s in out if s]


def pgo_cut_sentences(pgo_events: list[dict[str, Any]], n_sentences: int) -> set[int]:
    """Sentence indices where a PGO burst lands (same mapping as the browser player)."""
    if n_sentences < 2:
        return set()
    return {min(n_sentences - 1, max(1, round(p["position"] * n_sentences))) for p in pgo_events}


@dataclass
class Shot:
    start: float
    end: float
    image: Image.Image
    pan: tuple[float, float, float, float]   # start centre (x, y), end centre, as fractions
    hard_in: bool = False                     # entered by a hard cut (PGO) instead of a crossfade


@dataclass
class Subtitle:
    start: float
    end: float
    layer: Image.Image                         # RGBA, cropped to the text
    pos: tuple[int, int]


@dataclass
class Timeline:
    duration: float
    title: tuple[Image.Image, tuple[int, int]]
    slate: tuple[Image.Image, tuple[int, int]]
    shots: list[Shot]
    subs: list[Subtitle]
    flashes: list[float]
    body_start: float
    meta: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------- drawing helpers

def _font(cfg_v: dict[str, Any], size: int, italic: bool = False) -> ImageFont.FreeTypeFont:
    path = cfg_v["font_italic" if italic else "font"]
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        import matplotlib

        fallback = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / ("DejaVuSerif-Italic.ttf" if italic else "DejaVuSerif.ttf")
        return ImageFont.truetype(str(fallback), size)


def _wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        trial = f"{cur} {w}".strip()
        if font.getlength(trial) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    # balance the last two lines so a single word is not left alone
    if len(lines) >= 2 and len(lines[-1].split()) == 1 and len(lines[-2].split()) > 2:
        prev = lines[-2].split()
        lines[-2], lines[-1] = " ".join(prev[:-1]), f"{prev[-1]} {lines[-1]}"
    return lines


def _text_layer(lines: list[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int]]], gap: int) -> Image.Image:
    """Centered lines with a soft shadow, as a tight RGBA layer."""
    widths = [int(math.ceil(f.getlength(t))) for t, f, _ in lines]
    heights = [f.getbbox("Ay")[3] for _, f, _ in lines]
    pad = 24
    w, h = max(widths) + 2 * pad, sum(heights) + gap * (len(lines) - 1) + 2 * pad
    shadow = Image.new("L", (w, h), 0)
    text = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ds, dt = ImageDraw.Draw(shadow), ImageDraw.Draw(text)
    y = pad
    for (t, f, col), lw, lh in zip(lines, widths, heights):
        x = (w - lw) // 2
        ds.text((x, y + 2), t, font=f, fill=235)
        dt.text((x, y), t, font=f, fill=col + (255,))
        y += lh + gap
    shadow = shadow.filter(ImageFilter.GaussianBlur(7))
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.putalpha(shadow.point(lambda v: int(v * 0.85)))
    out.alpha_composite(text)
    return out


def ambient_field(stage: str, size: tuple[int, int], seed: int) -> Image.Image:
    """A dim, stage-coloured field of soft light for dreams without scene images."""
    w, h = size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32) + np.array([7, 9, 13], np.float32)
    base = np.array(STAGE_RGB.get(stage, STAGE_RGB["N2"]), np.float32)
    strength = 0.22 if stage == "N3" else 0.32
    for _ in range(4):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.3, 0.6) * w
        fall = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))[..., None]
        img += fall * base * strength * rng.uniform(0.6, 1.0)
    img += rng.normal(0, 2.0, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))


# ----------------------------------------------------------------------------- planning

def plan(ep: dict[str, Any], cfg: dict[str, Any], images: CachedImages | None, seed: int, work: Path,
         voice: bool, log: Callable[[str], None] = print) -> tuple[Timeline, Path | None]:
    v = cfg["video"]
    W, H = v["frame_size"]
    sentences = split_sentences(ep["narrative"])
    n_s = len(sentences)
    cuts = pgo_cut_sentences(ep["trace"].get("pgo_events", []), n_s)

    # sentence timing: narration when available, else reading speed
    wavs = tts.synthesize(sentences, work / "tts", v["voice"], v["voice_rate"]) if voice else None
    if wavs:
        durs = [tts.wav_seconds(p) + v["sentence_gap_s"] for p in wavs]
    else:
        durs = [max(2.2, len(s.split()) / v["reading_wps"] + 0.7) for s in sentences]
    t0 = v["title_s"]
    starts = list(np.cumsum([t0] + durs[:-1]))
    body_end = t0 + sum(durs)
    duration = body_end + v["tail_s"]

    audio = None
    if wavs:
        audio = work / "narration.wav"
        parts: list[tuple[Path | None, float]] = [(None, t0)]
        for p in wavs:
            parts += [(p, 0.0), (None, v["sentence_gap_s"])]
        tts.assemble(parts + [(None, v["tail_s"] + 0.5)], audio)

    # scenes
    prompts = ep.get("image_prompts") or []
    shots: list[Shot] = []
    rng = np.random.default_rng(seed)
    if prompts and images is not None:
        k = len(prompts)
        bounds = [round(i * n_s / k) for i in range(k)] + [n_s]
        for i, prompt in enumerate(prompts):
            log(f"    scene {i + 1}/{k}: {prompt[:70]}…")
            img = Image.open(images.get(prompt, seed * 101 + i)).convert("RGB")
            s = starts[bounds[i]] if i else 0.0
            e = starts[bounds[i + 1]] if bounds[i + 1] < n_s else duration
            a, b = rng.uniform(0.4, 0.6, 2), rng.uniform(0.35, 0.65, 2)
            shots.append(Shot(s, e, img, (a[0], a[1], b[0], b[1]), hard_in=i > 0 and bounds[i] in cuts))
    else:
        field_img = ambient_field(ep["stage"], tuple(v["image_size"]), seed)
        shots.append(Shot(0.0, duration, field_img, (0.45, 0.5, 0.55, 0.5)))

    # subtitles
    italic = ep["stage"] != "REM"
    font = _font(v, v["subtitle_px"], italic)
    color = (226, 230, 238) if italic else (246, 247, 250)
    subs = []
    for s_text, st, d in zip(sentences, starts, durs):
        layer = _text_layer([(ln, font, color) for ln in _wrap(s_text, font, int(W * 0.78))], gap=6)
        subs.append(Subtitle(st, st + d, layer, ((W - layer.width) // 2, int(H * 0.88) - layer.height)))

    # title card and corner slate
    title = PROFILE_TITLE.get(ep["profile"], ep["profile"])
    sub = f"cycle {ep['cycle']} · {ep['clock_time']}" + \
          (f" · bizarreness {ep['bizarreness_score']:.2f}" if ep.get("bizarreness_score") is not None else "")
    tl = _text_layer([(title, _font(v, int(v["subtitle_px"] * 1.6), True), (246, 247, 250)),
                      (sub, _font(v, int(v["subtitle_px"] * 0.6)), (170, 180, 196))], gap=14)
    sl = _text_layer([(f"{title} · cycle {ep['cycle']} · {ep['clock_time']}", _font(v, 20), (200, 206, 216))], gap=0)

    flashes = [starts[i] for i in sorted(cuts)]
    return Timeline(duration, (tl, ((W - tl.width) // 2, (H - tl.height) // 2)), (sl, (14, 10)), shots, subs, flashes,
                    t0, {"sentences": n_s, "scenes": len(prompts), "pgo_cuts": len(cuts), "narrated": bool(wavs)}), audio


# ----------------------------------------------------------------------------- frames

def _shot_frame(shot: Shot, t: float, size: tuple[int, int], zoom: float, span_pad: float) -> Image.Image:
    W, H = size
    iw, ih = shot.image.size
    p = min(1.0, max(0.0, (t - shot.start + span_pad) / max(shot.end - shot.start + 2 * span_pad, 1e-6)))
    p = p * p * (3 - 2 * p)                           # ease in-out
    s = 1.0 + zoom * p
    cw, ch = iw / s, ih / s
    cx = (shot.pan[0] + (shot.pan[2] - shot.pan[0]) * p) * iw
    cy = (shot.pan[1] + (shot.pan[3] - shot.pan[1]) * p) * ih
    x0 = min(max(cx - cw / 2, 0), iw - cw)
    y0 = min(max(cy - ch / 2, 0), ih - ch)
    return shot.image.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))


def _paste(frame: Image.Image, layer: Image.Image, pos: tuple[int, int], alpha: float) -> None:
    if alpha <= 0.01:
        return
    mask = layer.getchannel("A")
    if alpha < 0.99:
        mask = mask.point(lambda v, a=alpha: int(v * a))
    frame.paste(layer.convert("RGB"), pos, mask)


def frames(tl: Timeline, cfg: dict[str, Any]):
    v = cfg["video"]
    W, H = v["frame_size"]
    fps, cf, fl, fade = v["fps"], v["crossfade_s"], v["flash_s"], v["subtitle_fade_s"]
    shade = Image.new("L", (W, H))
    shade.putdata([int(min(185, max(0, (y / H - 0.5) * 2 * 185))) for y in range(H) for _ in range(W)])
    black, white = Image.new("RGB", (W, H)), Image.new("RGB", (W, H), (255, 255, 255))
    n = int(math.ceil(tl.duration * fps))
    for i in range(n):
        t = i / fps
        k = next((j for j, s in enumerate(tl.shots) if s.start <= t < s.end), len(tl.shots) - 1)
        shot = tl.shots[k]
        frame = _shot_frame(shot, t, (W, H), v["ken_burns_zoom"], cf / 2)
        # crossfade into the next shot, or out of the previous one (no fade across a PGO cut)
        if k + 1 < len(tl.shots) and not tl.shots[k + 1].hard_in and t > shot.end - cf / 2:
            a = (t - (shot.end - cf / 2)) / cf
            frame = Image.blend(frame, _shot_frame(tl.shots[k + 1], t, (W, H), v["ken_burns_zoom"], cf / 2), a)
        elif k > 0 and not shot.hard_in and t < shot.start + cf / 2:
            a = 0.5 + (t - shot.start) / cf
            frame = Image.blend(_shot_frame(tl.shots[k - 1], t, (W, H), v["ken_burns_zoom"], cf / 2), frame, a)
        frame.paste(black, (0, 0), shade)              # darker lower half for legible subtitles

        if t < tl.body_start:                          # title card over a darkened first scene
            frame = Image.blend(frame, black, 0.72)
            a = min(1.0, t / 0.6, (tl.body_start - t) / 0.5)
            _paste(frame, *tl.title, a)
        else:
            _paste(frame, *tl.slate, 0.55)
            for sub in tl.subs:
                if sub.start - 0.05 <= t < sub.end:
                    a = min(1.0, (t - sub.start) / fade, (sub.end - t) / fade)
                    _paste(frame, sub.layer, sub.pos, a)
        for tf in tl.flashes:
            if tf <= t < tf + fl:
                frame = Image.blend(frame, white, 0.55 * (1 - (t - tf) / fl))
        bright = min(1.0, t / 0.5, (tl.duration - t) / v["tail_s"])
        if bright < 1:
            frame = Image.blend(black, frame, max(0.0, bright))
        yield frame


def encode(tl: Timeline, audio: Path | None, cfg: dict[str, Any], out: Path) -> Path:
    v = cfg["video"]
    W, H = v["frame_size"]
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(v["fps"]), "-i", "-"]
    if audio and audio.exists():
        cmd += ["-i", str(audio)]
    else:   # a silent track keeps every clip concat-compatible
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=22050"]
    cmd += ["-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "fast", "-crf", str(v["crf"]),
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-ar", "22050", "-ac", "1",
            "-t", f"{tl.duration:.3f}", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for fr in frames(tl, cfg):
            proc.stdin.write(fr.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {err.strip()[:400]}")
    return out


def concat(clips: list[Path], out: Path) -> Path:
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{c.resolve().as_posix()}'\n" for c in clips), encoding="utf-8")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", "-movflags", "+faststart", str(out)], capture_output=True, text=True)
    lst.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {r.stderr.strip()[:400]}")
    return out


def episode_seed(night_seed: int, ep: dict[str, Any]) -> int:
    return zlib.crc32(f"{night_seed}:{ep['cycle']}:{ep['profile']}:{ep['minute_start']}".encode()) % (2**31)


def clip_name(ep: dict[str, Any]) -> str:
    """File stem of a dream's video, shared by `dream video` and the player."""
    return f"c{ep['cycle']}_{ep['profile']}_{ep['clock_time'].replace(':', '')}"
