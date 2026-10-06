from __future__ import annotations

import json

from click.testing import CliRunner

from conftest import LOGS
from dream_engine.cli import main


def test_ingest_then_replay(tmp_path):
    db = str(tmp_path / "memory.db")
    runner = CliRunner()
    for log in ["2026-08-20.md", "2026-10-04.md", "2026-10-05.md"]:
        r = runner.invoke(main, ["--db", db, "ingest", str(LOGS / log), "--extractor", "heuristic",
                                 "--embedder", "hashing"])
        assert r.exit_code == 0, r.output
    r = runner.invoke(main, ["--db", db, "replay", "--stage", "REM", "--date", "2026-10-06", "-n", "4",
                             "--json", "--commit"])
    assert r.exit_code == 0, r.output
    out = json.loads(r.output)
    assert len(out["picks"]) == 4
    assert out["picks"][0]["mode"] in {"seed", "dream_lag"}
    assert all("explanation" in p for p in out["picks"])

    r = runner.invoke(main, ["--db", db, "memories", "--json"])
    assert sum(f["replay_count"] for f in json.loads(r.output)) == 4
