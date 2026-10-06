from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from dream_engine.config import load_config
from dream_engine.memory.store import Fragment

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "data" / "day_logs"


@pytest.fixture
def cfg():
    return copy.deepcopy(load_config())


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def frag(fid: str, ts: datetime, valence=0.0, arousal=0.0, emb=None, replay_count=0, text=None) -> Fragment:
    return Fragment(id=fid, text=text or fid, type="episodic", entities=[], timestamp=ts, valence=valence,
                    arousal=arousal, embedding=unit(emb if emb is not None else [1.0, 0.0]),
                    replay_count=replay_count)
