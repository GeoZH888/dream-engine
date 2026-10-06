"""Parse a day log (markdown, plain text, or JSON) into timestamped entries.

Supported formats
-----------------
Markdown / text::

    # 2026-10-05
    - 08:30 Coffee at the bar near Santa Maria Novella, the barista remembered me
    - [19:10] Walked across Ponte Vecchio at dusk, a violinist playing
    Paragraphs separated by blank lines are also entries.

JSON: a list of strings, or of objects ``{"text": ..., "time": "HH:MM"}`` /
``{"text": ..., "timestamp": "ISO-8601"}``.

The date comes from a ``# YYYY-MM-DD`` heading, else the file name, else ``--date``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")
BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
TIME_RE = re.compile(r"^\[?(\d{1,2}):(\d{2})\]?\s*(?:[-–—:|]\s*)?")


@dataclass
class LogEntry:
    text: str
    timestamp: datetime
    source: str


def _parse_time(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def _split_time(text: str) -> tuple[time | None, str]:
    m = TIME_RE.match(text)
    if not m:
        return None, text
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 23 or mi > 59:
        return None, text
    return time(h, mi), text[m.end():].strip()


def _resolve_date(path: Path, text: str, override: date | None) -> date:
    if override:
        return override
    for line in text.splitlines():
        h = HEADING_RE.match(line.strip())
        if h and (d := DATE_RE.search(h.group(1))):
            return date.fromisoformat(d.group(1))
    if d := DATE_RE.search(path.stem):
        return date.fromisoformat(d.group(1))
    raise ValueError(f"Cannot determine the date of {path}; name it YYYY-MM-DD.md or pass --date")


def _text_blocks(text: str) -> list[str]:
    """Bullets are one entry each; other non-heading lines group into paragraphs."""
    blocks: list[str] = []
    para: list[str] = []

    def flush():
        if para:
            blocks.append(" ".join(para).strip())
            para.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or HEADING_RE.match(line):
            flush()
            continue
        if BULLET_RE.match(line):
            flush()
            blocks.append(BULLET_RE.sub("", line).strip())
        elif blocks and not para and raw[:1].isspace():
            blocks[-1] += " " + line  # continuation of an indented bullet
        else:
            para.append(line)
    flush()
    return [b for b in blocks if b]


def parse_log(path: str | Path, default_time: str = "12:00", on_date: date | None = None) -> list[LogEntry]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    fallback_time = _parse_time(default_time)

    if path.suffix.lower() == ".json":
        return _parse_json(path, raw, fallback_time, on_date)

    day = _resolve_date(path, raw, on_date)
    entries: list[LogEntry] = []
    last_time = fallback_time
    for block in _text_blocks(raw):
        t, body = _split_time(block)
        if not body:
            continue
        # Untimed entries inherit the previous entry's time, keeping the day's order.
        last_time = t or last_time
        entries.append(LogEntry(body, datetime.combine(day, last_time), str(path)))
    return entries


def _parse_json(path: Path, raw: str, fallback_time: time, on_date: date | None) -> list[LogEntry]:
    items = json.loads(raw)
    if not isinstance(items, list):
        raise ValueError(f"{path}: JSON day log must be a list")
    day = None
    entries = []
    for item in items:
        if isinstance(item, str):
            item = {"text": item}
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if "timestamp" in item:
            ts = datetime.fromisoformat(item["timestamp"])
        else:
            day = day or _resolve_date(path, "", on_date)
            t = _parse_time(item["time"]) if "time" in item else fallback_time
            ts = datetime.combine(day, t)
        entries.append(LogEntry(text, ts, str(path)))
    return entries
