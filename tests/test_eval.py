"""Phase 3: bizarreness rater, separability, visual prompts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from dream_engine.eval.bizarreness import CATEGORIES, BizarrenessRater, rate_night
from dream_engine.eval.separability import bizarreness_comparison, separability
from dream_engine.memory.embed import HashingEmbedder
from dream_engine.report import eval_markdown
from dream_engine.visual.prompts import LLMVisualizer


class FakeClient:
    def __init__(self, respond):
        self.respond, self.calls = respond, []
        self.messages = self.beta = self
        self.beta.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        text = json.dumps(self.respond(kw))
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def _rating(kw):
    report = kw["messages"][0]["content"]
    s = 0.9 if "flying" in report else 0.0
    return {c: {"score": s if c != "uncertainty" else s * 1.5, "evidence": ""} for c in CATEGORIES}


def test_rater_is_blind_and_scores_mean(cfg):
    client = FakeClient(_rating)
    r = BizarrenessRater(cfg, client=client).rate("I am flying over the station.")
    call = client.calls[0]
    assert call["messages"][0]["content"] == "Dream report:\nI am flying over the station."  # nothing but the text
    assert r["categories"]["uncertainty"]["score"] == 1.0          # clamped from 1.35
    assert r["score"] == pytest.approx((0.9 + 0.9 + 1.0) / 3, abs=1e-3)


def test_rater_model_override(cfg):
    cfg["eval"]["rater_model"] = "claude-opus-5-5"
    client = FakeClient(_rating)
    BizarrenessRater(cfg, client=client).rate("x")
    assert client.calls[0]["model"] == "claude-opus-5-5"


REM_TEXT = "I am flying over a bridge that is also a courtyard while my grandmother becomes a violin, glowing fish"
NREM_TEXT = "I am thinking about the train timetable and whether to email my supervisor about the deadline"


def _nights(n: int) -> list[dict]:
    rng = np.random.default_rng(0)
    nights = []
    for i in range(n):
        eps = []
        for cycle in range(1, 6):
            for stage, text in (("N2", NREM_TEXT), ("N3", "A maze. Grey page. Quiet."), ("REM", REM_TEXT)):
                words = text.split()
                rng.shuffle(words)
                eps.append({"stage": stage, "profile": "REM_late" if stage == "REM" else stage, "cycle": cycle,
                            "narrative": " ".join(words), "word_count": len(words), "bizarreness_score": None})
        nights.append({"night_id": "2026-10-06", "seed": i, "episodes": eps})
    return nights


def test_rate_night_and_compare(cfg):
    nights = _nights(3)
    rater = BizarrenessRater(cfg, client=FakeClient(_rating))
    assert rate_night(nights[0], rater, 4) == 15
    assert rate_night(nights[0], rater, 4) == 0          # already rated
    assert rate_night(nights[0], rater, 4, rerate=True) == 15
    for n in nights[1:]:
        rate_night(n, rater, 4)
    b = bizarreness_comparison(nights)
    assert b["rem_mean"] > b["nrem_mean"] and b["nights_rem_higher"] == b["nights"] == 3
    assert b["mann_whitney_p_rem_greater"] < 0.001
    assert set(b["by_profile"]) == {"N2", "N3", "REM_late"}


def test_separability_grouped_by_night(cfg):
    nights = _nights(10)
    s = separability(nights, HashingEmbedder(256, 0.3), folds=5, seed=0)
    assert s["grouped_by_night"] and s["n_rem"] == 50 and s["n_nrem"] == 100
    assert s["accuracy"] >= 0.95
    assert s["baseline_majority_accuracy"] == pytest.approx(2 / 3, abs=1e-3)


def test_eval_markdown(cfg):
    nights = _nights(5)
    rater = BizarrenessRater(cfg, client=FakeClient(_rating))
    for n in nights:
        rate_night(n, rater, 4)
    summary = {"nights": 5, "rater": rater.name, "bizarreness": bizarreness_comparison(nights),
               "separability": separability(nights, HashingEmbedder(256, 0.3), 5, 0),
               "targets": {"bizarreness_pass": True, "separability_target": 0.75, "separability_pass": True}}
    md = eval_markdown(summary)
    assert "REM > NREM in **5 / 5** nights" in md and "PASS" in md


def test_visual_prompts(cfg):
    client = FakeClient(lambda kw: {"prompts": ["a", "b", "c", " "]})
    out = LLMVisualizer(cfg, client=client).prompts("I am flying.", 2)
    assert out == ["a", "b"]
    assert cfg["visual"]["style"] in client.calls[0]["messages"][0]["content"]
    assert LLMVisualizer(cfg, client=client).prompts("x", 0) == [] and len(client.calls) == 1
