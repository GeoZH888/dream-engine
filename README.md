# Dream Engine

Generates dreams by **simulating the mechanisms the brain uses to make them**:
day residue → hippocampal replay → sleep cycles → PGO bursts and bizarreness
operators → a narrator with the prefrontal critic switched off → visual prompts.
See [DREAM_ENGINE_SPEC.md](DREAM_ENGINE_SPEC.md) for the scientific model.

**Status: Phases 1–5** (ingest, memory, replay · scheduler, noise, narrator · eval, visual prompts · EEG staging · web viewer).

**[Explore the nights →](https://dream-engine-nights.netlify.app)** 22 generated nights (one driven by
real EEG staging, plus the 20-night study): click through the hypnogram, read each dream, and
see why every element appeared. The [3D memory space](https://dream-engine-nights.netlify.app/space.html)
shows where each dream travels through the day's memories.

## Setup

```bash
pip install -e ".[embeddings,dev]"
```

Set `ANTHROPIC_API_KEY` in your environment (never in code). All tunable numbers
live in [config.yaml](config.yaml).

## Usage

```bash
dream ingest data/day_logs/2026-10-05.md           # Claude extraction + MiniLM embeddings
dream memories                                      # fragments with valence/arousal/entities
dream replay --stage REM_late --date 2026-10-06     # one traced replay sample
dream replay --stage REM --date 2026-10-06 --stats 300   # association-distance statistics

dream night --date 2026-10-06 --seed 42             # full night → outputs/nights/2026-10-06.{json,md}
dream night --date 2026-10-06 --dry-run             # plan + seeds only, no LLM calls
dream night --date 2026-10-06 --lucid               # allow partial lucidity in late REM
dream episode --stage REM --cycle 4 --date 2026-10-06    # the same episode `night` would make

dream eval outputs/nights/2026-10-06.json           # blind bizarreness rating + separability
dream study --nights 20 --date 2026-10-06           # 20 seeds → rate → eval (phase 3 acceptance)

dream space outputs/nights/2026-10-06*.json outputs/study/2026-10-06_s*.json  # → outputs/viewer/space.html (3D)
dream view outputs/nights/2026-10-06*.json outputs/study/2026-10-06_s*.json   # → outputs/viewer/index.html (--fragment for hosts that add their own <html>)

dream eeg fetch                                     # Sleep-EDF subset (config eeg.subjects / recordings)
dream eeg train                                     # features → LightGBM, CV grouped by subject
dream eeg stage data/eeg/physionet-sleep-data/SC4001E0-PSG.edf   # PSG → hypnogram CSV
dream night --date 2026-10-06 --hypnogram data/eeg/heldout/4001.csv   # dreams follow a real night
```

`--hypnogram` accepts our CSV, a Sleep-EDF `*Hypnogram.edf` (expert labels) or a
`*PSG.edf` (staged by the trained model). `dream eeg train` writes out-of-fold
hypnograms to `data/eeg/heldout/`: each one was staged by a model that never saw
that subject, so it is an honest stand-in for a new person's night.

Offline / no API key: `--extractor heuristic` and `--embedder hashing` for ingest,
`--dry-run` for nights. A database is bound to one embedder.

Day logs: markdown/text (`- 19:10 Walked across Ponte Vecchio…`, date from a
`# YYYY-MM-DD` heading or the file name) or JSON (`[{"time": "19:10", "text": "…"}]`).

## Pipeline

| Module | Mechanism | What it does |
|---|---|---|
| `ingest/` | Day residue | Day log → fragments with type, typed entities (person/place/object), valence, arousal |
| `memory/replay.py` | Hippocampal replay | `w = α·recency + β·arousal + γ·\|valence\| + δ·novelty − ε·replays`; stage-modulated; medium-distance REM chains; dream-lag |
| `sleep/scheduler.py` | NREM/REM cycles | ~8 h, 5 cycles; REM 10 → 40 min, N3 fades; one N2, N3 and REM episode per cycle |
| `noise/pgo.py` | PGO waves | Poisson bursts over REM minutes → scene transition or random-fragment intrusion |
| `noise/augment.py` | Overfitted brain | Gaussian noise on a seed embedding → nearest stored memory ("noisy recall") |
| `noise/bizarre.py` | Hobson's bizarreness | 7 pure operators (identity_merge, place_fusion, scene_jump, physics_violation, object_transformation, time_distortion, uncertainty) + structural prior |
| `synth/` | dlPFC off · cortical synthesis | Critic-off system prompt, stage style and length, optional lucidity |
| `visual/prompts.py` | Visual association cortex | 1–3 image prompts per REM episode; `ImageBackend` protocol for later |
| `eval/` | — | Blind LLM bizarreness rater (discontinuity, incongruity, uncertainty); NREM-vs-REM classifier |
| `viewer/` | — | One self-contained HTML page: clickable hypnogram with PGO ticks, episode list, report, rated bizarreness with evidence, and the full trace per episode |
| `viewer/space.*` | — | 3D memory space: fragments placed by meaning (metric MDS on cosine distance), each dream drawn as a path; play a night episode by episode |
| `eeg/` | Polysomnography | Sleep-EDF → band-pass, 30-s epochs → δ/θ/α/σ/β power, Hjorth, spindle density, EOG features (±2-epoch context) → LightGBM → W/N1/N2/N3/REM; hypnogram → cycles (one per REM period) → night |

Every stochastic choice is seeded: the same `(date, seed)` gives the same seeds;
only narration varies. The night report (`.md`) shows a hypnogram, then for each
episode the narrative and **why each element appeared**: the replay weight formula
and probability, association distance, noisy-recall source, PGO burst minute, and
each distortion's directive.

### Notes on the model

- **Temperature.** `claude-sonnet-5-5` rejects non-default `temperature` (HTTP 400).
  With `llm.send_temperature: false` the stage temperatures (0.4–1.0) are recorded in
  the trace and expressed as an "associative looseness" instruction in the prompt.
  Set it to `true` for a model that accepts sampling parameters.
- **Refusal fallback.** `llm.fallbacks: default` re-runs a declined request on a
  fallback model server-side; `null` disables it.

## Tests

```bash
pytest
```

- Phase 1: weight formula, stage modulation, dream-lag rate, medium-distance REM chains.
- Phase 2: 5 cycles with REM growing and N3 fading, Poisson PGO, operator targeting and
  purity, augmentation, prompt contents, deterministic planning, stage profiles respected.
- Phase 3: rater blindness and scoring, per-night comparison, grouped-CV separability.

- Phase 4: spindle detection, band-power features, smoothing, cycles from REM periods,
  episodes on the longest REM/N3 segment, nights planned on a real hypnogram.

Live acceptance results: `outputs/study/eval.md` (phase 3), `outputs/eeg/staging.md` (phase 4).

## Ethics

Simulation only, no dream interpretation or diagnosis. Day logs are personal data;
they stay local except for the LLM calls. Sleep-EDF is public research data; any EEG or
dream-incubation work on people needs informed consent and ethics approval.
