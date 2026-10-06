"""The dream seed: everything the narrator is allowed to use, plus why each piece is there."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from dream_engine.memory.store import Fragment


@dataclass(frozen=True)
class SeedFragment:
    id: str
    text: str
    entities: tuple[str, ...]
    entity_kinds: tuple[tuple[str, str], ...]
    valence: float
    arousal: float
    timestamp: datetime
    role: str                       # replay | augment | pgo_injection
    trace: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_fragment(cls, f: Fragment, role: str, trace: dict[str, Any] | None = None) -> "SeedFragment":
        return cls(f.id, f.text, tuple(f.entities), tuple(sorted(f.entity_kinds.items())), f.valence, f.arousal,
                   f.timestamp, role, tuple((trace or {}).items()))

    def kind(self, entity: str) -> str:
        return dict(self.entity_kinds).get(entity, "other")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "role": self.role,
                "timestamp": self.timestamp.isoformat(timespec="minutes"),
                "valence": round(self.valence, 3), "arousal": round(self.arousal, 3),
                "trace": dict(self.trace)}


@dataclass(frozen=True)
class OperatorApplication:
    name: str
    directive: str                  # what the narrator must enact
    involves: tuple[str, ...] = ()  # fragment ids / entity names it acts on

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "directive": self.directive, "involves": list(self.involves)}


@dataclass(frozen=True)
class PGOEvent:
    minute: float                   # minute of the night
    position: float                 # 0..1 position within the REM segment
    kind: str                       # scene_transition | fragment_injection
    fragment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"minute": round(self.minute, 1), "position": round(self.position, 2), "kind": self.kind}
        if self.fragment_id:
            d["fragment_id"] = self.fragment_id
        return d


@dataclass(frozen=True)
class DreamSeed:
    stage: str                      # N2 | N3 | REM
    profile: str                    # N2 | N3 | REM_early | REM_late
    cycle: int
    minute_start: int
    fragments: tuple[SeedFragment, ...]
    operators: tuple[OperatorApplication, ...] = ()
    pgo_events: tuple[PGOEvent, ...] = ()
    lucid: bool = False
    notes: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def entities_of(self, kind: str) -> list[tuple[str, str]]:
        """(entity, fragment_id) pairs of one kind, de-duplicated by name."""
        seen, out = set(), []
        for f in self.fragments:
            for e in f.entities:
                if f.kind(e) == kind and e.lower() not in seen:
                    seen.add(e.lower())
                    out.append((e, f.id))
        return out
