"""SQLite memory store (``memory.db``): fragments with emotion tags and embeddings."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from dream_engine.ingest.extract import ExtractedFragment

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS fragments (
    num           INTEGER PRIMARY KEY AUTOINCREMENT,
    id            TEXT UNIQUE NOT NULL,
    text          TEXT NOT NULL,
    type          TEXT NOT NULL,
    entities      TEXT NOT NULL,
    entity_kinds  TEXT NOT NULL DEFAULT '{}',
    timestamp     TEXT NOT NULL,
    valence       REAL NOT NULL,
    arousal       REAL NOT NULL,
    embedding     BLOB NOT NULL,
    replay_count  INTEGER NOT NULL DEFAULT 0,
    source        TEXT,
    extractor     TEXT,
    UNIQUE (text, timestamp)
);
CREATE TABLE IF NOT EXISTS replay_events (
    num          INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment_id  TEXT NOT NULL REFERENCES fragments(id),
    replayed_at  TEXT NOT NULL,
    stage        TEXT NOT NULL,
    trace        TEXT NOT NULL
);
"""


@dataclass
class Fragment:
    id: str
    text: str
    type: str
    entities: list[str]
    timestamp: datetime
    valence: float
    arousal: float
    embedding: np.ndarray = field(repr=False)
    replay_count: int = 0
    source: str | None = None
    extractor: str | None = None
    entity_kinds: dict[str, str] = field(default_factory=dict)

    def to_dict(self, with_embedding: bool = False) -> dict:
        d = {
            "id": self.id, "text": self.text, "type": self.type, "entities": self.entities,
            "entity_kinds": self.entity_kinds,
            "timestamp": self.timestamp.isoformat(timespec="minutes"),
            "valence": round(self.valence, 3), "arousal": round(self.arousal, 3),
            "replay_count": self.replay_count, "source": self.source, "extractor": self.extractor,
        }
        if with_embedding:
            d["embedding"] = self.embedding.tolist()
        return d


class EmbedderMismatch(RuntimeError):
    pass


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(fragments)")}
        if "entity_kinds" not in cols:  # databases created before entity kinds existed
            self.conn.execute("ALTER TABLE fragments ADD COLUMN entity_kinds TEXT NOT NULL DEFAULT '{}'")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- embedder bookkeeping: all vectors in one DB must come from the same embedder
    @property
    def embedder_name(self) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
        return row["value"] if row else None

    def _check_embedder(self, name: str) -> None:
        current = self.embedder_name
        if current is None:
            self.conn.execute("INSERT INTO meta VALUES ('embedder', ?)", (name,))
        elif current != name:
            raise EmbedderMismatch(
                f"{self.path} holds '{current}' embeddings; cannot add '{name}' embeddings. "
                f"Use the same embedding backend or ingest into a new database.")

    # -- writes
    def add(self, frags: list[ExtractedFragment], embeddings: np.ndarray, embedder_name: str) -> list[str]:
        """Insert fragments; returns ids of the ones added (duplicates are skipped)."""
        if len(frags) != len(embeddings):
            raise ValueError("one embedding per fragment required")
        self._check_embedder(embedder_name)
        added = []
        with self.conn:
            for f, vec in zip(frags, embeddings):
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO fragments (id, text, type, entities, entity_kinds, timestamp, valence,"
                    " arousal, embedding, source, extractor) VALUES ('pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (f.text, f.type, json.dumps(f.entities, ensure_ascii=False),
                     json.dumps(f.entity_kinds, ensure_ascii=False), f.timestamp.isoformat(),
                     f.valence, f.arousal, np.asarray(vec, dtype=np.float32).tobytes(), f.source, f.extractor),
                )
                if cur.rowcount:
                    fid = f"frag_{cur.lastrowid:04d}"
                    self.conn.execute("UPDATE fragments SET id=? WHERE num=?", (fid, cur.lastrowid))
                    added.append(fid)
        return added

    def record_replay(self, fragment_ids: Iterable[str], stage: str, traces: Iterable[dict],
                      at: datetime) -> None:
        with self.conn:
            for fid, trace in zip(fragment_ids, traces):
                self.conn.execute("UPDATE fragments SET replay_count = replay_count + 1 WHERE id=?", (fid,))
                self.conn.execute(
                    "INSERT INTO replay_events (fragment_id, replayed_at, stage, trace) VALUES (?, ?, ?, ?)",
                    (fid, at.isoformat(), stage, json.dumps(trace)))

    # -- reads
    def all(self) -> list[Fragment]:
        rows = self.conn.execute("SELECT * FROM fragments ORDER BY num").fetchall()
        return [self._row(r) for r in rows]

    def get(self, fid: str) -> Fragment | None:
        row = self.conn.execute("SELECT * FROM fragments WHERE id=?", (fid,)).fetchone()
        return self._row(row) if row else None

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]

    @staticmethod
    def _row(r: sqlite3.Row) -> Fragment:
        return Fragment(
            id=r["id"], text=r["text"], type=r["type"], entities=json.loads(r["entities"]),
            timestamp=datetime.fromisoformat(r["timestamp"]), valence=r["valence"], arousal=r["arousal"],
            embedding=np.frombuffer(r["embedding"], dtype=np.float32), replay_count=r["replay_count"],
            source=r["source"], extractor=r["extractor"], entity_kinds=json.loads(r["entity_kinds"]),
        )
