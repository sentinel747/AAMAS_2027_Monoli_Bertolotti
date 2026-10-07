# Coordination and Resource Management in Societies of LLM-Based Agents

Source code for the AAMAS 2027 submission *Coordination and Resource
Management in Societies of LLM-Based Agents*. A linguistic governor and
district administrators write the behavioural law of a mass-conserving Mars
colony of 300 non-linguistic colonists. Every comparison is paired by seed
against the ungoverned colony and read against the spread between identical
runs.

The records of every run the paper reports (`runs/`, about 570 MB compressed)
are distributed as a separate archive, too large for a Git repository. Unzip it
in the repository root: the scripts expect `runs/` next to `src/` and
`scripts/`.

## Layout

```
src/                 simulator (kernel, world, colonists, governors, administrators, semantic System-1 layer)
scripts/             experiment runner, campaign queues, analysis and figure scripts
scripts/paper/       figure scripts used only by the paper
configs/             provider registry (llm_providers.json) and run configurations
tests/               pytest suite (determinism, governors, policy language, replay)
rust/                optional PyO3 accelerator, bit-identical to the Python path
data/sagome/         country outline used for scale in figures
runs/                run records (separate data archive), see runs/CAMPAIGNS.md
```

## Installation

Tested with Python 3.13 on Windows 11; nothing is platform-specific.

```bash
python -m venv .venv
.venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests/governors
```

Three tests in that folder run a full simulation and therefore need the Mars
Climate Database (below); without it they fail with
`MCD call_mcd runtime is required`, and the other 418 pass.

The Rust accelerator is optional and produces bit-identical results:
`python scripts/build_rust.py` (requires a Rust toolchain and maturin).

### Mars Climate Database

Climate fields come from the Mars Climate Database v6.1 (Millour et al.), which
has its own licence and is **not** redistributed here. To re-run simulations
bit-exactly, request MCD 6.1 from its distributors
(<https://www-mars.lmd.jussieu.fr/>) and place it so that
`data/mcd_runtime/MCD_6.1/data/` exists (`scripts/prepare_mcd_data.py` helps
with the layout). Recomputing the paper's numbers from the included records does
**not** need the MCD.

## 1. Recompute the paper's numbers from the records (no model, no MCD)

Every number in the results section is computed from files in `runs/`. Run the
scripts from the repository root:

| paper element | command |
|---|---|
| Prompt variants, single levels, blind arm, per-campaign repeat spread (Qwen3.8-27B) | `python scripts/esiti_campagna.py runs/paper_qwen` |
| Same for the second model (gpt-oss-20b); verdicts are withheld because its repeated arm is incomplete | `python scripts/esiti_campagna.py runs/paper_oss` |
| Decision-history anchoring | `python scripts/analisi_ancoraggio.py runs/paper_qwen/variante_* runs/paper_qwen/cieco_due_livelli` |
| Stress test on the four other worlds | `python scripts/controllo_copertura_mondi_report.py` |
| Format, not model (Fig. 3), written to `figures/`; its top row combines the four executions of the reference arm (`runs/controllo_copertura_20260926/variante_D_vera*` and `runs/paper_qwen/variante_D*`) | `python scripts/paper/figura_formato_2x2.py` |

Comments and output of the analysis scripts are in Italian; numbers, file
names and arguments are not. Each script documents its inputs in its header.

## 2. Re-run the experiments

Every run is launched by `scripts/run_governor_experiment.py`. The exact command
line of each run of the two main campaigns is in `runs/piano_paper_qwen.txt` and
`runs/piano_paper_oss.txt` (tab-separated: arm label, output folder, seed,
arguments). For example, the ungoverned baseline on seed 3:

```bash
python scripts/run_governor_experiment.py --arms none --strato-ambientale \
  --correzioni-motore --pavimento-cantieri 1.1 --steps 1000 --agents 300 \
  --cadence 25 --dotazione 0.6 --log-interval 250 --snapshot-interval 25 \
  --wait 600 --map-profile balanced --copertura-elettrica-vera \
  --seeds 3 --out runs/rerun/ctrl_none
```

**Ungoverned and scripted runs are deterministic**: the same seed and
configuration give a bit-identical final state, which you can check against
`final_metrics.json` and the state digest of the included run.

**Language-model runs are not deterministic**, which is why the paper repeats
the reference arm and reads every effect against the spread between identical
executions. They need an OpenAI-compatible endpoint:

- set `base_url` of the `gpu_farm4` (vLLM, `Qwen/Qwen3.8-27B-FP8`) and
  `gpu_farm` (Ollama, `gpt-oss:20b`) entries in `configs/llm_providers.json` to
  your own server (the published file contains a placeholder), and export the
  key named in `api_key_env`;
- enable model calls explicitly: `export MARSABM_ALLOW_LLM_CALLS=1`;
- check latency first: `python scripts/check_provider.py`; the governor cadence
  must exceed it, or no policy ever takes effect.

The System-1 "choosing" arms with Jev call a commercial API (TypeSafe), read
from the `TYPESAFE_AI` environment variable; Qwen in that format is read from
its own first-token log-probabilities and is free.

A governor's decisions can be replayed from `governor_decisions.jsonl` with no
provider (`governors.replay_from` in the run configuration, `src/governors/record.py`).

## What is in each run folder, and what is not

Kept, for every run: `config.yaml`, `run_metadata.json`, `final_metrics.json`,
`state_timeseries.csv` (one row per step), `governor_decisions.jsonl` and
`administrator_decisions.jsonl` (picture sent, full model answer, parsed policy,
verdict), `governor_policy_hits.json`, `dead_agents.jsonl`, `llm_usage.jsonl`,
`semantic_decisions.jsonl` (System-1 runs), `research_summary.json`, and the
**final** world snapshot under `world_snapshots/`. Campaign folders keep
`results.json` / `results.jsonl`, the per-campaign log and queue files.

Left out, to keep the archive publishable (about 190 GB in full): per-action
traces (`validated_actions`, `rejected_actions`, `events`, `replay_events`,
`agent_thoughts`, `agent_states`), the static world file, intermediate
snapshots, and runs discarded by the admission gate (lost governor rounds or
provider errors). All of these are regenerated by re-running the configuration.

Not every campaign is complete: `runs/paper_oss` holds 49 of its 65 planned
runs; the paper uses only gpt-oss arms with all five seeds.

## Licences

Code: MIT (`LICENSE`). Run records: CC BY 4.0 (`DATA_LICENSE.md`).
The Mars Climate Database is not part of this repository and keeps its own licence.
