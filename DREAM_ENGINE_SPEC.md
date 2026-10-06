# Dream Engine — Spec for Claude Code

> Goal: build a system that generates dreams by **simulating the mechanisms the
> brain uses to make them**, rather than just prompting an LLM with "write a dream".
> Each module maps to a known neuroscience mechanism, so the output is
> explainable and can be used in research.

---

## 0. Instructions to Claude Code

- Read this whole file before writing code. Build in the phases of §8, in order.
- After each phase, run the acceptance tests for that phase and stop for review.
- Python 3.11+. Keep modules small and testable. Never hard-code the API key.
- Put every tunable number in `config.yaml`, not in the code.

---

## 1. Scientific model → software mapping

| Brain mechanism | Evidence / theory | Software module |
|---|---|---|
| Day residue (recent experiences enter dreams) | Freud; Nielsen's dream-lag effect | `ingest/` — daily input log |
| Hippocampal replay of memory fragments | Wilson & McNaughton; consolidation theory | `memory/replay.py` — weighted sampling |
| Amygdala salience (emotion biases content) | Limbic hyperactivation in REM | `memory/emotion.py` — valence/arousal tags |
| Sleep cycles: NREM → REM, ~90 min, REM grows longer | Polysomnography | `sleep/scheduler.py` — night timeline |
| PGO waves (bursts of semi-random activation) | Activation-synthesis (Hobson & McCarley) | `noise/pgo.py` — stochastic injection |
| Noise to prevent overfitting | Overfitted Brain Hypothesis (Hoel, 2021) | `noise/augment.py` — distortion operators |
| Prefrontal deactivation (no self-reflection or logic check) | dlPFC hypoactivity in REM | `synth/prompts.py` — "critic-off" system prompt |
| Narrative construction | Cortical synthesis | `synth/narrator.py` — LLM generation |
| Visual association cortex | Visual imagery in REM | `visual/` — image prompt generation |
| Dream bizarreness | Hobson's categories: discontinuity, incongruity, uncertainty | `noise/bizarre.py` — operators + scorer |
| Lucidity (dlPFC partially re-activates) | Lucid dreaming research (Voss et al.) | `synth/lucid.py` — optional mode |

---

## 2. Architecture

```
 ┌────────────┐   ┌──────────────┐   ┌─────────────┐
 │ ingest/    │──▶│ memory graph │──▶│ replay      │
 │ day log    │   │ + emotion    │   │ sampler     │
 └────────────┘   └──────────────┘   └──────┬──────┘
                                            │ fragments
 ┌────────────┐   ┌──────────────┐   ┌──────▼──────┐
 │ sleep      │──▶│ stage params │──▶│ noise / PGO │
 │ scheduler  │   │ (NREM / REM) │   │ + bizarre   │
 └─────┬──────┘   └──────────────┘   └──────┬──────┘
       │ (optional)                         │ distorted seeds
 ┌─────▼──────┐                      ┌──────▼──────┐   ┌──────────┐
 │ EEG sleep  │                      │ narrator    │──▶│ visual   │
 │ staging    │                      │ (LLM)       │   │ prompts  │
 └────────────┘                      └──────┬──────┘   └──────────┘
                                            ▼
                                  dream_report.json + .md
```

---

## 3. Data model

### 3.1 Memory fragment
```json
{
  "id": "frag_0042",
  "text": "Walked across Ponte Vecchio at dusk, a violinist playing",
  "type": "episodic | semantic | person | place | object | emotion",
  "entities": ["Ponte Vecchio", "violinist", "dusk"],
  "timestamp": "2026-10-05T19:10:00",
  "valence": 0.6,          // -1 (negative) .. +1 (positive)
  "arousal": 0.4,          //  0 (calm) .. 1 (intense)
  "embedding": [/* vector */],
  "replay_count": 0
}
```

### 3.2 Dream episode
```json
{
  "night_id": "2026-10-06",
  "cycle": 3,
  "stage": "REM",
  "minute_start": 290,
  "seed_fragments": ["frag_0042", "frag_0007"],
  "operators_applied": ["identity_merge", "scene_jump"],
  "narrative": "...",
  "image_prompts": ["..."],
  "bizarreness_score": 0.71,
  "emotional_tone": {"valence": -0.2, "arousal": 0.8}
}
```

---

## 4. Modules

### 4.1 `ingest/` — day residue
- Input: a plain-text or markdown day log written by the user (or a JSON list).
- An LLM extracts fragments, entities, and valence/arousal from each entry.
- Store in SQLite (`memory.db`) with embeddings (sentence-transformers, local).

### 4.2 `memory/replay.py` — hippocampal replay sampler
Sampling weight for fragment *i*:

```
w_i = α · recency_i + β · arousal_i + γ · |valence_i| + δ · novelty_i − ε · replay_count_i
recency_i = exp(−Δt_i / τ)          # τ from config, e.g. 2 days
```
- **NREM** favors recency (sharper, more episodic replay).
- **REM** favors arousal and associative distance: after the first seed, sample
  neighbors at a *medium* embedding distance (not nearest; not random) to produce
  loose association chains.
- Include a small chance (config `old_memory_prob`, default 0.15) of pulling
  memories older than 30 days (the dream-lag effect).

### 4.3 `sleep/scheduler.py` — night timeline
Generate an ~8-hour night of ~5 cycles. REM duration grows across the night
(about 10 → 40 min). Output a list of `(minute, stage)` segments.
Each stage has a parameter profile:

| Param | N2 | N3 | REM early | REM late |
|---|---|---|---|---|
| fragments per episode | 1–2 | 1 | 3–4 | 4–6 |
| LLM temperature | 0.6 | 0.4 | 0.9 | 1.0 |
| bizarre operators | 0 | 0 | 1–2 | 2–4 |
| narrative length (words) | 40–80 | 20–40 | 150–250 | 250–400 |
| style | thought-like, mundane | fragmentary | vivid, visual | vivid, emotional, story-like |

### 4.4 `noise/` — PGO bursts and the overfitted-brain idea
- `pgo.py`: a Poisson process over REM minutes. Each burst triggers either a
  **scene transition** or the **injection of a random fragment**.
- `bizarre.py` operators (each a pure function on the seed structure):
  - `identity_merge` — two people become one ("my supervisor, who was also my grandmother")
  - `place_fusion` — two places merge (Florence station opening onto a Chinese courtyard)
  - `scene_jump` — abrupt discontinuity with no transition
  - `physics_violation` — flying, breathing underwater, gravity changes
  - `object_transformation` — an object changes category mid-scene
  - `time_distortion` — past and present overlap; the dreamer is a different age
  - `uncertainty` — vague identity ("someone I knew, maybe")
- `augment.py`: embedding-space noise. Perturb seed embeddings with Gaussian
  noise σ (from config) and retrieve the nearest fragments to the perturbed point.

### 4.5 `synth/` — narrator with the prefrontal critic switched off
System prompt rules (store in `synth/prompts.py`):
1. Write in first person, present tense, as a dream report told immediately on waking.
2. The dreamer **never questions** impossibilities while inside the dream.
3. Do not explain symbolism. Do not moralize. Do not resolve the plot.
4. Use only the provided seed fragments + operators. Do not invent unrelated themes.
5. Emotion is strong but its cause can be unclear.
6. N3: fragments only. REM: sensory detail, especially visual.

`lucid.py` (optional): partially turns the critic back on in late REM, so the
dreamer notices one anomaly and gains limited control.

Model: configurable in `config.yaml` (e.g. `model: claude-sonnet-5-5`), using the Anthropic Python SDK.

### 4.6 `visual/`
- For each REM episode, generate 1–3 image prompts: dream-like composition,
  soft edges, impossible perspectives. Prompts only in phase 1. An image-model
  backend is a pluggable adapter later.

### 4.7 `eval/` — measuring dream-likeness
- **Bizarreness score**: an LLM rater using Hobson's three categories, 0–1.
- **Stage separability**: a simple classifier on embeddings should separate
  NREM-generated from REM-generated reports (target ≥ 0.75 accuracy).
- **Optional human comparison**: compare against real reports from **DreamBank**
  (dreambank.net), e.g. Hall–Van de Castle content categories.

### 4.8 `eeg/` (optional, phase 4) — real sleep staging
- Use the Sleep-EDF dataset (PhysioNet) for development.
- Pipeline: band-pass filter → 30 s epochs → features (δ/θ/α/σ/β band power,
  spindle density, EOG activity for REM) → classifier (start with
  LightGBM, then a small CNN) → stages W/N1/N2/N3/REM.
- Replace the simulated scheduler with the real hypnogram, so the dreams follow
  an actual recorded night.
- Future extension: targeted dream incubation (audio cues during detected N1/REM,
  as in MIT Dormio). Research use only, with consent.

---

## 5. File structure

```
dream-engine/
├── config.yaml
├── README.md
├── dream_engine/
│   ├── ingest/          parse_log.py, extract.py
│   ├── memory/          store.py, replay.py, emotion.py
│   ├── sleep/           scheduler.py
│   ├── noise/           pgo.py, bizarre.py, augment.py
│   ├── synth/           prompts.py, narrator.py, lucid.py
│   ├── visual/          prompts.py
│   ├── eval/            bizarreness.py, separability.py
│   ├── eeg/             (phase 4)
│   └── cli.py
├── data/
│   ├── day_logs/
│   └── memory.db
├── outputs/nights/      2026-10-06.json, 2026-10-06.md
└── tests/
```

## 6. CLI

```bash
dream ingest data/day_logs/2026-10-05.md
dream night --date 2026-10-06 --seed 42           # full simulated night
dream episode --stage REM --cycle 4                # single episode
dream night --hypnogram data/eeg/night1.edf        # phase 4
dream eval outputs/nights/2026-10-06.json
```

## 7. Output (`outputs/nights/<date>.md`)
A readable night report: a hypnogram strip (text or PNG), then each episode
with stage, time, seed fragments, operators applied, narrative, and image prompts.
The trace of *why* each element appeared must always be present.
This explainability is the point of the project.

## 8. Build phases

| Phase | Scope | Acceptance |
|---|---|---|
| 1 | ingest + memory store + replay sampler | Sampling weights unit-tested; REM chains show medium-distance associations |
| 2 | scheduler + noise + narrator | `dream night` produces 5 cycles; NREM reports short and mundane, REM vivid and bizarre |
| 3 | eval + visual prompts | Bizarreness REM > NREM on 20 nights; separability ≥ 0.75 |
| 4 | EEG staging (Sleep-EDF) | Staging accuracy ≥ 0.75 vs. labels; night driven by real hypnogram |
| 5 | Optional: web viewer (night timeline + episodes) | — |

## 9. Config (`config.yaml` defaults)
```yaml
model: claude-sonnet-5-5
night_hours: 8
cycles: 5
replay: {alpha: 1.0, beta: 1.2, gamma: 0.8, delta: 0.5, epsilon: 0.3, tau_days: 2, old_memory_prob: 0.15}
noise:  {embedding_sigma: 0.15, pgo_rate_per_min: 0.2}
seed: 42
```

## 10. Ethics and limits
- This **simulates** dream-like content. It does not read or create real dreams.
- Day logs are personal data. Keep them local and never upload them except to the LLM call.
- No dream "interpretation" or psychological diagnosis in the output.
- EEG/incubation work on humans requires informed consent and ethics approval.

## 11. References
- Hobson & McCarley (1977). Activation-synthesis hypothesis. *Am J Psychiatry*.
- Hoel (2021). The overfitted brain: dreams evolved to assist generalization. *Patterns*.
- Revonsuo (2000). Threat simulation theory. *Behav Brain Sci*.
- Horikawa et al. (2013). Neural decoding of visual imagery during sleep. *Science*.
- Wilson & McNaughton (1994). Reactivation of hippocampal ensemble memories during sleep. *Science*.
- Ha & Schmidhuber (2018). World Models. / Hafner et al. DreamerV3.
- Haar Horowitz et al. (2020). Dormio: targeted dream incubation. *Conscious Cogn*.
- Kemp et al. Sleep-EDF database, PhysioNet.
