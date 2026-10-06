from __future__ import annotations

import json
from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import LOGS
from dream_engine.ingest.extract import ExtractionError, ExtractedFragment, HeuristicExtractor, LLMExtractor
from dream_engine.ingest.parse_log import LogEntry, parse_log
from dream_engine.memory import emotion
from dream_engine.memory.embed import HashingEmbedder
from dream_engine.memory.store import EmbedderMismatch, MemoryStore


# ---------------------------------------------------------------- parse_log

def test_parse_markdown_log():
    entries = parse_log(LOGS / "2026-10-05.md")
    assert len(entries) == 12
    assert entries[0].timestamp == datetime(2026, 10, 5, 7, 40)
    assert entries[8].text.startswith("Walked across Ponte Vecchio")
    assert entries[8].timestamp == datetime(2026, 10, 5, 19, 10)


def test_parse_untimed_and_paragraphs(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# Saturday 2026-09-12\n\n- 09:00 Breakfast\n- Untimed thing after breakfast\n\n"
                 "A paragraph that\nspans two lines.\n", encoding="utf-8")
    e = parse_log(p)
    assert [x.text for x in e] == ["Breakfast", "Untimed thing after breakfast", "A paragraph that spans two lines."]
    assert e[1].timestamp == datetime(2026, 9, 12, 9, 0)


def test_parse_json_log():
    e = parse_log(LOGS / "2026-10-03.json")
    assert len(e) == 3 and e[1].timestamp == datetime(2026, 10, 3, 13, 15)


def test_parse_requires_date(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("- something", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_log(p)
    assert parse_log(p, on_date=date(2026, 1, 2))[0].timestamp == datetime(2026, 1, 2, 12, 0)


# ---------------------------------------------------------------- emotion + heuristic extractor

def test_lexicon_tagging(cfg):
    e = cfg["emotion"]
    v, a = emotion.tag_text("Argued with the landlord. Angry and tired.", e["default_arousal"],
                            e["intensifier_boost"], e["negation_window"])
    assert v < -0.4 and a > 0.7
    v2, _ = emotion.tag_text("I was not happy", e["default_arousal"], e["intensifier_boost"], e["negation_window"])
    assert v2 < 0
    v3, a3 = emotion.tag_text("Read a paper about maps", e["default_arousal"], e["intensifier_boost"],
                              e["negation_window"])
    assert v3 == 0 and a3 == e["default_arousal"]


def test_heuristic_extractor_entities(cfg):
    entry = LogEntry("Walked across Ponte Vecchio at dusk, a violinist playing.", datetime(2026, 10, 5, 19, 10), "x")
    (f,) = HeuristicExtractor(cfg).extract([entry])
    assert "Ponte Vecchio" in f.entities and "violinist" in f.entities
    assert f.entity_kinds["Ponte Vecchio"] == "place" and f.entity_kinds["violinist"] == "person"
    assert f.extractor == "heuristic" and f.timestamp == entry.timestamp


# ---------------------------------------------------------------- LLM extractor (fake client)

class FakeMessages:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.calls = payload, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(stop_reason=self.stop_reason,
                               content=[SimpleNamespace(type="text", text=json.dumps(self.payload))])


def fake_client(payload, stop_reason="end_turn"):
    m = FakeMessages(payload, stop_reason)
    return SimpleNamespace(messages=m, beta=SimpleNamespace(messages=m)), m


ENTRIES = [LogEntry("Walked across Ponte Vecchio at dusk", datetime(2026, 10, 5, 19, 10), "log.md"),
           LogEntry("Argued with the landlord", datetime(2026, 10, 5, 17, 30), "log.md")]


def test_llm_extractor_parses_and_clamps(cfg):
    payload = {"fragments": [
        {"entry_index": 0, "text": "Crossed Ponte Vecchio at dusk", "type": "place",
         "entities": [{"name": "Ponte Vecchio", "kind": "place"}], "valence": 0.6, "arousal": 0.4},
        {"entry_index": 1, "text": "Shouting match with landlord", "type": "episodic",
         "entities": [{"name": "landlord", "kind": "person"}], "valence": -3, "arousal": 1.7},
        {"entry_index": 9, "text": "hallucinated entry", "type": "episodic", "entities": [], "valence": 0, "arousal": 0},
    ]}
    client, msgs = fake_client(payload)
    frags = LLMExtractor(cfg, client=client).extract(ENTRIES)
    assert len(frags) == 2
    assert frags[1].valence == -1.0 and frags[1].arousal == 1.0
    assert frags[1].timestamp == ENTRIES[1].timestamp
    assert frags[0].entities == ["Ponte Vecchio"] and frags[0].entity_kinds == {"Ponte Vecchio": "place"}
    assert frags[0].extractor == f"llm:{cfg['model']}"
    call = msgs.calls[0]
    assert call["model"] == cfg["model"]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default"


def test_llm_extractor_refusal(cfg):
    client, _ = fake_client({"fragments": []}, stop_reason="refusal")
    with pytest.raises(ExtractionError):
        LLMExtractor(cfg, client=client).extract(ENTRIES)


# ---------------------------------------------------------------- store

def _frags():
    ts = datetime(2026, 10, 5, 19, 10)
    return [ExtractedFragment("Ponte Vecchio at dusk", "place", ["Ponte Vecchio"], 0.6, 0.4, ts, "log", "heuristic",
                              {"Ponte Vecchio": "place"}),
            ExtractedFragment("Landlord argument", "episodic", ["landlord"], -0.7, 0.9, ts, "log", "heuristic")]


def test_store_roundtrip_dedupe_and_replay(tmp_path):
    emb = HashingEmbedder(64, 0.3)
    frags = _frags()
    vecs = emb.embed([f.text for f in frags])
    with MemoryStore(tmp_path / "m.db") as s:
        assert s.add(frags, vecs, emb.name) == ["frag_0001", "frag_0002"]
        assert s.add(frags, vecs, emb.name) == []  # duplicates skipped
        f = s.get("frag_0001")
        assert f.entities == ["Ponte Vecchio"] and f.valence == pytest.approx(0.6)
        assert f.entity_kinds == {"Ponte Vecchio": "place"}
        assert np.allclose(f.embedding, vecs[0])
        s.record_replay(["frag_0002"], "REM", [{"mode": "seed"}], datetime(2026, 10, 5, 23))
        assert s.get("frag_0002").replay_count == 1
        with pytest.raises(EmbedderMismatch):
            s.add(frags, vecs, "some-other-embedder")
