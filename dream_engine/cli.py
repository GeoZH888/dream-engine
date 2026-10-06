"""`dream` command line."""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import click

from dream_engine.config import load_config
from dream_engine.ingest.extract import make_extractor
from dream_engine.ingest.parse_log import parse_log
from dream_engine.memory.embed import make_embedder
from dream_engine.memory.replay import ReplaySampler, chain_distance_stats
from dream_engine.memory.store import MemoryStore
from dream_engine.sleep.scheduler import sleep_onset

STAGES = ["N1", "N2", "N3", "REM", "REM_early", "REM_late"]


@click.group()
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to config.yaml (default: project config).")
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default=None, help="Override paths.db.")
@click.pass_context
def main(ctx: click.Context, config_path: str | None, db_path: str | None) -> None:
    """Dream Engine: generate dreams by simulating the brain's dream mechanisms."""
    # joblib cannot count physical cores on some Windows setups and prints a traceback
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(config_path)
    ctx.obj = {"cfg": cfg, "db": db_path or cfg["paths"]["db"]}


@main.command()
@click.argument("log_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--extractor", type=click.Choice(["llm", "heuristic"]), default=None)
@click.option("--embedder", type=click.Choice(["sentence-transformers", "hashing"]), default=None)
@click.option("--date", "on_date", type=click.DateTime(["%Y-%m-%d"]), default=None,
              help="Date of the log if not in its heading or file name.")
@click.pass_obj
def ingest(obj: dict, log_path: str, extractor: str | None, embedder: str | None, on_date) -> None:
    """Ingest a day log (day residue) into the memory store."""
    cfg = obj["cfg"]
    entries = parse_log(log_path, cfg["ingest"]["default_entry_time"], on_date.date() if on_date else None)
    click.echo(f"Parsed {len(entries)} entries from {log_path}")

    ext, warning = make_extractor(cfg, extractor)
    if warning:
        click.secho(f"warning: {warning}", fg="yellow", err=True)
    frags = ext.extract(entries)
    click.echo(f"Extracted {len(frags)} fragments with the {ext.name} extractor")

    emb = make_embedder(cfg, embedder)
    vectors = emb.embed([f.text for f in frags])
    with MemoryStore(obj["db"]) as store:
        added = store.add(frags, vectors, emb.name)
        click.echo(f"Stored {len(added)} new fragments ({len(frags) - len(added)} duplicates skipped); "
                   f"{store.count()} total in {obj['db']} [{emb.name}]")
        for fid in added:
            f = store.get(fid)
            click.echo(f"  {f.id}  {f.timestamp:%m-%d %H:%M}  {f.type:<8} v={f.valence:+.2f} a={f.arousal:.2f}  {f.text}")


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@click.pass_obj
def memories(obj: dict, as_json: bool) -> None:
    """List stored memory fragments."""
    with MemoryStore(obj["db"]) as store:
        frags = store.all()
    if as_json:
        click.echo(json.dumps([f.to_dict() for f in frags], indent=2, ensure_ascii=False))
        return
    for f in frags:
        ents = ", ".join(f.entities)
        click.echo(f"{f.id}  {f.timestamp:%Y-%m-%d %H:%M}  {f.type:<8} v={f.valence:+.2f} a={f.arousal:.2f} "
                   f"r={f.replay_count}  {f.text}" + (f"  [{ents}]" if ents else ""))


@main.command()
@click.option("--stage", type=click.Choice(STAGES, case_sensitive=False), default="REM")
@click.option("-n", "--fragments", "n", type=int, default=None,
              help="Fragments to replay (default: upper bound of the stage profile).")
@click.option("--date", "night", type=click.DateTime(["%Y-%m-%d"]), default=None,
              help="Night id (wake date); replay happens at sleep onset the evening before. Default: today.")
@click.option("--seed", type=int, default=None, help="RNG seed (default: config seed).")
@click.option("--commit", is_flag=True, help="Increment replay_count and log the replay event.")
@click.option("--stats", "stats_chains", type=int, default=0,
              help="Instead of one sample, run N chains and report association-distance statistics.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def replay(obj: dict, stage: str, n: int | None, night, seed: int | None, commit: bool,
           stats_chains: int, as_json: bool) -> None:
    """Sample a replay of memory fragments for one sleep stage, with traces."""
    cfg = obj["cfg"]
    now = sleep_onset(cfg, night.date() if night else date.today())
    if n is None:
        profile = cfg["stages"].get(stage) or cfg["stages"]["REM_late" if stage.upper().startswith("REM") else "N2"]
        n = profile["fragments"][1]
    with MemoryStore(obj["db"]) as store:
        frags = store.all()
        if not frags:
            raise click.ClickException("Memory store is empty; run `dream ingest` first.")
        sampler = ReplaySampler(frags, cfg["replay"], seed=cfg["seed"] if seed is None else seed)

        if stats_chains:
            stats = chain_distance_stats(sampler, stage, n, now, stats_chains)
            click.echo(json.dumps(stats, indent=2))
            return

        result = sampler.sample(stage, n, now)
        traces = result.traces()
        if commit:
            store.record_replay([f.id for f in result.fragments], stage, traces, now)

    if as_json:
        click.echo(json.dumps({"stage": stage, "now": now.isoformat(), "coefficients": result.coefficients,
                               "picks": [{**t, "text": p.fragment.text} for t, p in zip(traces, result.picks)]},
                              indent=2, ensure_ascii=False))
        return
    c = result.coefficients
    click.echo(f"{stage} replay at {now:%Y-%m-%d %H:%M}  (α={c['alpha']:.2f} β={c['beta']:.2f} "
               f"γ={c['gamma']:.2f} δ={c['delta']:.2f} ε={c['epsilon']:.2f})")
    for t, p in zip(traces, result.picks):
        f = p.fragment
        click.echo(f"\n[{t['step']}] {t['mode']:<11} {f.id}  ({f.timestamp:%Y-%m-%d})  {f.text}")
        click.echo(f"      {t['explanation']}")
        line = f"      p={t['probability']:.3f} among {t['candidates']} candidates"
        if "distance_from_prev" in t:
            lo, hi = t["distance_band"]
            line += (f"; distance from previous {t['distance_from_prev']:.3f} "
                     f"(rank {t['distance_rank']}, band {lo:.3f}–{hi:.3f})")
        click.echo(line)


def _load_fragments(db: str):
    with MemoryStore(db) as store:
        frags = store.all()
    if not frags:
        raise click.ClickException("Memory store is empty; run `dream ingest` first.")
    return frags


def _engines(cfg: dict, dry_run: bool):
    from dream_engine.llm import has_credentials
    from dream_engine.synth.narrator import DryRunNarrator, LLMNarrator
    from dream_engine.visual.prompts import DryRunVisualizer, LLMVisualizer

    if dry_run:
        return DryRunNarrator(cfg), DryRunVisualizer(cfg)
    if not has_credentials():
        raise click.ClickException("No Anthropic credentials (ANTHROPIC_API_KEY); use --dry-run to inspect seeds.")
    narrator = LLMNarrator(cfg)
    return narrator, LLMVisualizer(cfg, client=narrator.client)


def _report_errors(night: dict) -> None:
    failed = [e for e in night["episodes"] if e.get("error")]
    for e in failed:
        click.secho(f"warning: cycle {e['cycle']} {e['profile']} failed: {e['error']}", fg="yellow", err=True)


@main.command()
@click.option("--date", "night", type=click.DateTime(["%Y-%m-%d"]), default=None,
              help="Night id = wake date (default: today).")
@click.option("--seed", type=int, default=None, help="RNG seed (default: config seed).")
@click.option("--lucid", is_flag=True, help="Allow partial lucidity in late REM.")
@click.option("--dry-run", is_flag=True, help="Plan the night and write seeds without calling the LLM.")
@click.option("--commit", is_flag=True, help="Persist this night's replays to replay_count in the store.")
@click.option("--hypnogram", type=click.Path(exists=True), default=None,
              help="Drive the night with a real hypnogram: our CSV, a Sleep-EDF Hypnogram EDF, or a PSG EDF.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default=None)
@click.pass_obj
def night(obj: dict, night, seed: int | None, lucid: bool, dry_run: bool, commit: bool, hypnogram, out_dir) -> None:
    """Simulate a full night of dreams and write outputs/nights/<date>.json and .md."""
    from dream_engine.night import plan_night, run_night
    from dream_engine.report import write_night

    cfg = obj["cfg"]
    night_date = night.date() if night else date.today()
    seed = cfg["seed"] if seed is None else seed
    segments, source = None, None
    if hypnogram:
        from dream_engine.eeg.hypnogram import read_hypnogram, segments_from_labels

        labels, source = read_hypnogram(hypnogram, cfg)
        segments = segments_from_labels(labels, cfg)
    plan = plan_night(cfg, _load_fragments(obj["db"]), night_date, seed, lucid=lucid, segments=segments,
                      hypnogram_source=source)
    narrator, visualizer = _engines(cfg, dry_run)
    click.echo(f"Night {plan.night_id}: {len(plan.episodes)} episodes over "
               f"{max(s.cycle for s in plan.segments)} cycles (hypnogram: {plan.hypnogram_source}; "
               f"narrator: {narrator.name})")
    result = run_night(cfg, plan, narrator, visualizer)
    _report_errors(result)
    stem = plan.night_id + (f".{Path(hypnogram).stem}" if hypnogram else "") + (".dry" if dry_run else "")
    jp, mp = write_night(result, out_dir or cfg["paths"]["outputs"], stem=stem)
    if commit:
        with MemoryStore(obj["db"]) as store:
            for ep in plan.episodes:
                ids = [f.id for f in ep.seed.fragments if f.role == "replay"]
                store.record_replay(ids, ep.seed.profile, ep.replay_traces, ep.clock)
    for e in result["episodes"]:
        click.echo(f"  c{e['cycle']} {e['profile']:<9} {e['clock_time']}  {e['word_count']:>3} words  "
                   f"ops={','.join(e['operators_applied']) or '-'}")
    click.echo(f"Wrote {jp} and {mp}")


@main.command()
@click.option("--stage", type=click.Choice(["N2", "N3", "REM"], case_sensitive=False), required=True)
@click.option("--cycle", type=int, required=True)
@click.option("--date", "night", type=click.DateTime(["%Y-%m-%d"]), default=None)
@click.option("--seed", type=int, default=None)
@click.option("--lucid", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.pass_obj
def episode(obj: dict, stage: str, cycle: int, night, seed: int | None, lucid: bool, dry_run: bool) -> None:
    """Generate one episode: the same one `dream night` would produce at that stage and cycle."""
    from dream_engine.night import plan_night, run_night
    from dream_engine.report import night_markdown

    cfg = obj["cfg"]
    seed = cfg["seed"] if seed is None else seed
    plan = plan_night(cfg, _load_fragments(obj["db"]), night.date() if night else date.today(), seed, lucid=lucid)
    eps = [ep for ep in plan.episodes if ep.seed.stage == stage.upper() and ep.seed.cycle == cycle]
    if not eps:
        have = ", ".join(f"{ep.seed.stage}@{ep.seed.cycle}" for ep in plan.episodes)
        raise click.ClickException(f"No {stage} episode in cycle {cycle}. This night has: {have}")
    narrator, visualizer = _engines(cfg, dry_run)
    result = run_night(cfg, plan, narrator, visualizer, episodes=eps)
    _report_errors(result)
    md = night_markdown(result)
    click.echo(md[md.index("## 1."):])


def _evaluate(cfg: dict, nights: list[dict], rerate: bool, embedder_backend: str | None) -> dict:
    from dream_engine.eval.bizarreness import BizarrenessRater, rate_night
    from dream_engine.eval.separability import bizarreness_comparison, separability
    from dream_engine.llm import has_credentials

    if not has_credentials():
        raise click.ClickException("No Anthropic credentials (ANTHROPIC_API_KEY) for the bizarreness rater.")
    rater = BizarrenessRater(cfg)
    for n in nights:
        k = rate_night(n, rater, cfg["llm"]["concurrency"], rerate=rerate)
        click.echo(f"  rated {k} reports in night {n['night_id']} (seed {n['seed']})")
    biz = bizarreness_comparison(nights)
    sep = separability(nights, make_embedder(cfg, embedder_backend), cfg["eval"]["cv_folds"], cfg["seed"])
    target = cfg["eval"]["separability_target"]
    return {
        "nights": len(nights), "rater": rater.name, "bizarreness": biz, "separability": sep,
        "targets": {
            "bizarreness_pass": biz["nights"] > 0 and biz["nights_rem_higher"] == biz["nights"],
            "separability_target": target,
            "separability_pass": sep.get("accuracy", 0) >= target,
        },
    }


def _write_eval(summary: dict, out_dir: Path) -> None:
    from dream_engine.report import eval_markdown

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "eval.md").write_text(eval_markdown(summary), encoding="utf-8")
    click.echo(eval_markdown(summary))
    click.echo(f"Wrote {out_dir / 'eval.json'} and {out_dir / 'eval.md'}")


@main.command(name="eval")
@click.argument("night_files", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--rerate", is_flag=True, help="Re-rate reports that already have a bizarreness score.")
@click.option("--embedder", type=click.Choice(["sentence-transformers", "hashing"]), default=None)
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default=None,
              help="Where to write eval.json/.md (default: next to the first night file).")
@click.pass_obj
def eval_cmd(obj: dict, night_files: tuple[str, ...], rerate: bool, embedder: str | None, out_dir) -> None:
    """Rate bizarreness (written back into each night) and measure stage separability."""
    from dream_engine.report import write_night

    cfg = obj["cfg"]
    paths = [Path(p) for p in night_files]
    nights = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    summary = _evaluate(cfg, nights, rerate, embedder)
    for p, n in zip(paths, nights):
        write_night(n, p.parent, stem=p.stem)
    _write_eval(summary, Path(out_dir) if out_dir else paths[0].parent)


@main.command()
@click.option("--nights", "n_nights", type=int, default=None, help="Number of nights (default: eval.nights).")
@click.option("--date", "night", type=click.DateTime(["%Y-%m-%d"]), default=None, help="Night id for all nights.")
@click.option("--seed", type=int, default=None, help="First seed; nights use seed, seed+1, ...")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default=None)
@click.pass_obj
def study(obj: dict, n_nights: int | None, night, seed: int | None, out_dir) -> None:
    """Generate N nights (different seeds), then rate and evaluate them all (phase 3 acceptance)."""
    from dream_engine.night import plan_night, run_night
    from dream_engine.report import write_night

    cfg = obj["cfg"]
    n_nights = n_nights or cfg["eval"]["nights"]
    seed = cfg["seed"] if seed is None else seed
    night_date = night.date() if night else date.today()
    out = Path(out_dir or cfg["paths"]["study"])
    frags = _load_fragments(obj["db"])
    narrator, visualizer = _engines(cfg, dry_run=False)

    nights = []
    for i in range(n_nights):
        s = seed + i
        stem = f"{night_date.isoformat()}_s{s}"
        path = out / f"{stem}.json"
        if path.exists():  # resumable: keep nights already generated
            nights.append(json.loads(path.read_text(encoding="utf-8")))
            click.echo(f"[{i + 1}/{n_nights}] seed {s}: reusing {path}")
            continue
        result = run_night(cfg, plan_night(cfg, frags, night_date, s), narrator, visualizer)
        _report_errors(result)
        write_night(result, out, stem=stem)
        nights.append(result)
        click.echo(f"[{i + 1}/{n_nights}] seed {s}: {len(result['episodes'])} episodes")

    summary = _evaluate(cfg, nights, rerate=False, embedder_backend=None)
    for n in nights:
        write_night(n, out, stem=f"{n['night_id']}_s{n['seed']}")
    _write_eval(summary, out)


@main.command()
@click.argument("night_files", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default="outputs/viewer/index.html",
              show_default=True)
@click.option("--fragment", is_flag=True, help="Omit the <html>/<head>/<body> wrapper (for hosts that add their own).")
@click.pass_obj
def view(obj: dict, night_files: tuple[str, ...], out_path: str, fragment: bool) -> None:
    """Bundle night JSON files into one self-contained HTML viewer (phase 5)."""
    from dream_engine.viewer.build import write_viewer

    out = write_viewer([Path(p) for p in night_files], Path(out_path), standalone=not fragment)
    click.echo(f"Wrote {out} ({out.stat().st_size // 1024} KB, {len(night_files)} nights)")


@main.command()
@click.argument("night_files", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default="outputs/viewer/space.html",
              show_default=True)
@click.pass_obj
def space(obj: dict, night_files: tuple[str, ...], out_path: str) -> None:
    """3D memory space: memories placed by meaning, each dream drawn as a path through them."""
    from dream_engine.viewer.space import write_space

    cfg = obj["cfg"]
    with MemoryStore(obj["db"]) as store:
        frags, embedder = store.all(), store.embedder_name
    if not frags:
        raise click.ClickException("Memory store is empty; run `dream ingest` first.")
    out = write_space(frags, [Path(p) for p in night_files], Path(out_path), cfg["seed"], embedder or "?")
    click.echo(f"Wrote {out} ({out.stat().st_size // 1024} KB, {len(frags)} memories, {len(night_files)} nights)")


@main.group()
def eeg() -> None:
    """Phase 4: real sleep staging on Sleep-EDF."""


@eeg.command()
@click.pass_obj
def fetch(obj: dict) -> None:
    """Download the configured Sleep-EDF subset from PhysioNet (via MNE)."""
    from mne.datasets.sleep_physionet.age import fetch_data

    e = obj["cfg"]["eeg"]
    files = fetch_data(subjects=list(range(e["subjects"])), recording=e["recordings"], path=e["data_dir"],
                       on_missing="warn")
    click.echo(f"{len(files)} recordings in {e['data_dir']}")


@eeg.command()
@click.option("--rebuild", is_flag=True, help="Recompute features instead of using the cache.")
@click.pass_obj
def train(obj: dict, rebuild: bool) -> None:
    """Cross-validate the stager (grouped by subject), save held-out hypnograms, fit the final model."""
    import joblib

    from dream_engine.eeg.load import STAGES, find_recordings
    from dream_engine.eeg.stager import build_dataset, cross_validate, make_model, save_hypnogram_csv, \
        staging_markdown

    cfg = obj["cfg"]
    e = cfg["eeg"]
    data_dir = Path(e["data_dir"])
    recs = [r for r in find_recordings(data_dir) if r.hypnogram]
    if not recs:
        raise click.ClickException(f"No Sleep-EDF recordings in {data_dir}; run `dream eeg fetch`.")
    cache = data_dir / "features.npz"
    if rebuild and cache.exists():
        cache.unlink()
    click.echo(f"Features for {len(recs)} recordings")
    ds = build_dataset(recs, cfg, cache=cache, log=click.echo)
    click.echo(f"{len(ds.y)} epochs × {ds.X.shape[1]} features; cross-validating")
    res = cross_validate(ds, cfg, log=click.echo)
    oof = res.pop("_oof")
    for r in sorted(set(ds.recording.tolist())):
        m = ds.recording == r
        save_hypnogram_csv(data_dir / "heldout" / f"{r}.csv", [STAGES[i] for i in oof[m]],
                           [STAGES[i] for i in ds.y[m]], e["epoch_seconds"])
    joblib.dump(make_model(cfg).fit(ds.X, ds.y), data_dir / "stager.joblib")
    out = Path("outputs/eeg")
    out.mkdir(parents=True, exist_ok=True)
    (out / "staging.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = staging_markdown(res, e["accuracy_target"])
    (out / "staging.md").write_text(md, encoding="utf-8")
    click.echo(md)
    click.echo(f"Held-out hypnograms: {data_dir / 'heldout'}; model: {data_dir / 'stager.joblib'}")


@eeg.command()
@click.argument("psg", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None)
@click.pass_obj
def stage(obj: dict, psg: str, out_path: str | None) -> None:
    """Stage a PSG recording with the trained model and write a hypnogram CSV."""
    from dream_engine.eeg.hypnogram import read_hypnogram, segments_from_labels
    from dream_engine.eeg.stager import save_hypnogram_csv
    from dream_engine.sleep.scheduler import hypnogram_text

    cfg = obj["cfg"]
    labels, source = read_hypnogram(psg, cfg)
    out = Path(out_path or Path(psg).with_suffix(".hypnogram.csv"))
    save_hypnogram_csv(out, labels, epoch_s=cfg["eeg"]["epoch_seconds"])
    click.echo(hypnogram_text(segments_from_labels(labels, cfg)))
    click.echo(f"{source}: {len(labels)} epochs → {out}")


if __name__ == "__main__":
    main()
