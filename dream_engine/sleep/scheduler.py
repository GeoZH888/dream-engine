"""Night timeline (spec §4.3): ~8 h, ~5 NREM→REM cycles of ~90 min.

Each cycle is N2 → N3 → N2 → REM (the first starts with N1). REM lengthens across
the night (≈10 → 40 min) while N3 (slow-wave sleep) fades; N2 fills the rest.
The night ends with a brief wake segment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np


@dataclass
class Segment:
    minute_start: int
    minute_end: int
    stage: str          # W | N1 | N2 | N3 | REM
    cycle: int

    @property
    def minutes(self) -> int:
        return self.minute_end - self.minute_start

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeSlot:
    """Where in the night a dream episode is generated, and with which stage profile."""
    cycle: int
    stage: str          # N2 | N3 | REM
    profile: str        # N2 | N3 | REM_early | REM_late
    minute_start: int
    segment: Segment


def _lerp(a: float, b: float, i: int, n: int) -> float:
    return a if n <= 1 else a + (b - a) * i / (n - 1)


def build_night(cfg: dict[str, Any], rng: np.random.Generator) -> list[Segment]:
    s = cfg["sleep"]
    total = int(cfg["night_hours"] * 60)
    n_cycles = cfg["cycles"]
    cycle_len = total / n_cycles

    def jit(x: float) -> int:
        return max(0, int(round(x + rng.uniform(-s["jitter_minutes"], s["jitter_minutes"])))) if x > 0 else 0

    segs: list[Segment] = []
    t = 0
    for c in range(n_cycles):
        cycle = c + 1
        rem = max(1, jit(_lerp(*s["rem_minutes"], c, n_cycles)))
        n3 = jit(_lerp(*s["n3_minutes"], c, n_cycles))
        n1 = jit(s["n1_minutes"]) if c == 0 else 0
        end = total if c == n_cycles - 1 else int(round((c + 1) * cycle_len))
        n2 = max(2, end - t - rem - n3 - n1)
        n2a = int(round(n2 * s["n2_split"])) if n3 else n2
        n2b = n2 - n2a
        for stage, mins in [("N1", n1), ("N2", n2a), ("N3", n3), ("N2", n2b), ("REM", rem)]:
            if mins > 0:
                segs.append(Segment(t, t + mins, stage, cycle))
                t += mins
    segs.append(Segment(t, t + 2, "W", n_cycles))  # waking: when the report is told
    return segs


def episode_slots(cfg: dict[str, Any], segments: list[Segment]) -> list[EpisodeSlot]:
    """Per cycle: one REM episode (the longest REM segment), one N3 episode (the longest
    N3 segment, if any) and one N2 episode (the last N2 segment before that REM).

    A simulated cycle has one REM and one N3 segment; real hypnograms have several, and
    segments shorter than ``min_episode_minutes`` (brief stage flickers) host no episode.
    """
    s = cfg["sleep"]
    slots = []
    for cycle in sorted({seg.cycle for seg in segments}):
        segs = [x for x in segments if x.cycle == cycle and x.minutes >= s["min_episode_minutes"]]
        rem = max((x for x in segs if x.stage == "REM"), key=lambda x: x.minutes, default=None)
        n3 = max((x for x in segs if x.stage == "N3"), key=lambda x: x.minutes, default=None)
        n2 = [x for x in segs if x.stage == "N2" and (rem is None or x.minute_start < rem.minute_start)]
        picks = [x for x in (n2[-1] if n2 else None, n3, rem) if x is not None]
        for seg in sorted(picks, key=lambda x: x.minute_start):
            if seg.stage == "REM":
                profile = "REM_late" if cycle >= s["rem_late_from_cycle"] else "REM_early"
            else:
                profile = seg.stage
            slots.append(EpisodeSlot(cycle, seg.stage, profile, seg.minute_start, seg))
    return slots


STAGE_ROWS = ["W", "REM", "N1", "N2", "N3"]


def hypnogram_text(segments: list[Segment], minutes_per_char: int = 5) -> str:
    """A multi-row text hypnogram, one column per ``minutes_per_char`` minutes."""
    end = segments[-1].minute_end
    cols = (end + minutes_per_char - 1) // minutes_per_char

    def stage_at(minute: int) -> str:
        for seg in segments:
            if seg.minute_start <= minute < seg.minute_end:
                return seg.stage
        return "W"

    col_stage = [stage_at(i * minutes_per_char + minutes_per_char // 2) for i in range(cols)]
    lines = [f"{row:>3} │" + "".join("█" if st == row else " " for st in col_stage) for row in STAGE_ROWS]
    axis = "    └" + "".join("┴" if (i * minutes_per_char) % 60 == 0 else "─" for i in range(cols))
    labels = "     " + "".join(
        str(i * minutes_per_char // 60).ljust(60 // minutes_per_char) for i in range(0, cols, 60 // minutes_per_char))
    return "\n".join(lines + [axis, labels.rstrip() + "  (hours)"])


def sleep_onset(cfg: dict[str, Any], night: date) -> datetime:
    """The night with id D (the wake date) starts on the evening of D−1."""
    h, m = map(int, cfg["sleep_onset"].split(":"))
    return datetime.combine(night - timedelta(days=1), time(h, m))
