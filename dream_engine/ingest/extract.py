"""Turn day-log entries into memory fragments (day residue).

Two extractors share one interface:
- ``LLMExtractor``: Claude extracts fragments, typed entities, valence and arousal
  (structured JSON output, schema-validated).
- ``HeuristicExtractor``: offline fallback (one fragment per entry + lexicon tagging),
  used for tests and when no API credentials are available.

Every fragment records which extractor produced it, so the trace is explainable.
Entity kinds (person / place / object / other) let the bizarreness operators act on
the right things: two *people* merge, two *places* fuse, an *object* transforms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from dream_engine.ingest.parse_log import LogEntry
from dream_engine.llm import LLMError, has_credentials, make_client, structured_call
from dream_engine.memory import emotion

FRAGMENT_TYPES = ["episodic", "semantic", "person", "place", "object", "emotion"]
ENTITY_KINDS = ["person", "place", "object", "other"]

ExtractionError = LLMError


@dataclass
class ExtractedFragment:
    text: str
    type: str
    entities: list[str]
    valence: float
    arousal: float
    timestamp: datetime
    source: str
    extractor: str
    entity_kinds: dict[str, str] = field(default_factory=dict)


class Extractor(Protocol):
    name: str

    def extract(self, entries: list[LogEntry]) -> list[ExtractedFragment]: ...


# --------------------------------------------------------------------------- heuristic

CAPITALIZED_RE = re.compile(r"\b([A-Z][\w'’-]+(?:\s+(?:[A-Z][\w'’-]+|di|de|del|della|of|the))*)")
PERSON_WORDS = {
    "mom", "mum", "mother", "dad", "father", "grandmother", "grandma", "grandfather", "grandpa",
    "sister", "brother", "friend", "supervisor", "boss", "colleague", "professor", "teacher",
    "barista", "violinist", "stranger", "doctor", "neighbor", "neighbour", "child", "baby", "waiter",
    "landlord", "cousin", "cousins",
}
PLACE_WORDS = {
    "station", "bridge", "courtyard", "piazza", "market", "kitchen", "lab", "airport", "river",
    "balcony", "trattoria", "flat", "room", "street", "house", "home", "sea", "museum", "bar",
    "garden", "office", "school", "beach", "church", "hall", "city",
}
STOP_CAPS = {"I", "The", "A", "An", "We", "My", "It", "This", "That", "Then", "After", "Before",
             "At", "In", "On", "Today", "Tonight", "Later", "Still", "And", "But", "So"}


def heuristic_kind(entity: str) -> str:
    toks = entity.lower().split()
    if entity.lower() in PERSON_WORDS:
        return "person"
    if any(t in PLACE_WORDS for t in toks) or len(toks) > 1 and entity[:1].isupper():
        return "place"
    return "other"


def _entities(text: str) -> list[str]:
    found: list[str] = []
    for m in CAPITALIZED_RE.finditer(text):
        words = m.group(1).strip().split()
        while words and words[0] in STOP_CAPS:
            words = words[1:]
        if words:
            found.append(" ".join(words))
    for tok in re.findall(r"[a-zA-Z]+", text.lower()):
        if tok in PERSON_WORDS or tok in PLACE_WORDS:
            found.append(tok)
    seen, out = set(), []
    for e in found:
        if e.lower() not in seen:
            seen.add(e.lower())
            out.append(e)
    return out


class HeuristicExtractor:
    name = "heuristic"

    def __init__(self, cfg: dict[str, Any]):
        self.emo = cfg["emotion"]

    def extract(self, entries: list[LogEntry]) -> list[ExtractedFragment]:
        """One episodic fragment per entry: without an LLM, splitting loses the context."""
        out = []
        for entry in entries:
            v, a = emotion.tag_text(entry.text, self.emo["default_arousal"], self.emo["intensifier_boost"],
                                    self.emo["negation_window"])
            ents = _entities(entry.text)
            out.append(ExtractedFragment(
                text=entry.text, type="episodic", entities=ents, valence=v, arousal=a,
                timestamp=entry.timestamp, source=entry.source, extractor=self.name,
                entity_kinds={e: heuristic_kind(e) for e in ents},
            ))
        return out


# --------------------------------------------------------------------------- LLM

EXTRACT_SYSTEM = """You extract memory fragments from a personal day log, for a research \
simulation of how day residue enters dreams.

For each numbered entry, produce 1 to {max_per_entry} fragments. A fragment is a single \
self-contained memory trace, phrased briefly in the past tense as the person would recall it \
(keep their concrete sensory details; do not add details that are not in the log).

Fields:
- entry_index: the number of the entry the fragment comes from.
- type: one of episodic (an event), semantic (a fact learned), person, place, object, \
emotion (a feeling state).
- entities: people, places and salient concrete objects named or clearly implied, each with a \
kind: person, place, object, or other. Use short names as they appear ("my supervisor", \
"Ponte Vecchio", "laptop bag").
- valence: -1 (very negative) to 1 (very positive); 0 if neutral.
- arousal: 0 (calm) to 1 (intense).

Rate emotion from the person's own perspective. Do not interpret or diagnose."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "fragments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entry_index": {"type": "integer"},
                    "text": {"type": "string"},
                    "type": {"type": "string", "enum": FRAGMENT_TYPES},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"},
                                           "kind": {"type": "string", "enum": ENTITY_KINDS}},
                            "required": ["name", "kind"],
                            "additionalProperties": False,
                        },
                    },
                    "valence": {"type": "number"},
                    "arousal": {"type": "number"},
                },
                "required": ["entry_index", "text", "type", "entities", "valence", "arousal"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["fragments"],
    "additionalProperties": False,
}


class LLMExtractor:
    name = "llm"

    def __init__(self, cfg: dict[str, Any], client: Any = None):
        self.cfg = cfg
        self.max_per_entry = cfg["ingest"]["max_fragments_per_entry"]
        self.client = client or make_client()

    def extract(self, entries: list[LogEntry]) -> list[ExtractedFragment]:
        if not entries:
            return []
        listing = "\n".join(f"[{i}] ({e.timestamp:%Y-%m-%d %H:%M}) {e.text}" for i, e in enumerate(entries))
        data = structured_call(
            self.client, self.cfg, system=EXTRACT_SYSTEM.format(max_per_entry=self.max_per_entry),
            user=f"Day log entries:\n{listing}", schema=EXTRACT_SCHEMA, effort=self.cfg["llm"]["extract_effort"])

        out = []
        for f in data["fragments"]:
            idx = f["entry_index"]
            if not 0 <= idx < len(entries) or not f["text"].strip():
                continue
            entry = entries[idx]
            kinds = {e["name"].strip(): e["kind"] for e in f["entities"] if e["name"].strip()}
            out.append(ExtractedFragment(
                text=f["text"].strip(),
                type=f["type"] if f["type"] in FRAGMENT_TYPES else "episodic",
                entities=list(kinds),
                valence=emotion.clamp_valence(f["valence"]),
                arousal=emotion.clamp_arousal(f["arousal"]),
                timestamp=entry.timestamp, source=entry.source, extractor=f"llm:{self.cfg['model']}",
                entity_kinds=kinds,
            ))
        return out


def make_extractor(cfg: dict[str, Any], kind: str | None = None) -> tuple[Extractor, str | None]:
    """Return (extractor, warning). Falls back to the heuristic extractor without credentials."""
    kind = kind or cfg["ingest"]["extractor"]
    if kind == "heuristic":
        return HeuristicExtractor(cfg), None
    if kind == "llm":
        if not has_credentials():
            return HeuristicExtractor(cfg), (
                "No Anthropic credentials found (ANTHROPIC_API_KEY); using the heuristic extractor.")
        return LLMExtractor(cfg), None
    raise ValueError(f"Unknown extractor: {kind}")
