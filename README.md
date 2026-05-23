# GuardedRepair: Harm-Aware Post-hoc Repair for LLM Mathematical Reasoning

This repository contains the code and reported artifacts for the paper:

**Guarded Repair for Harm-Aware Post-hoc Verification of LLM Mathematical Reasoning**

The project studies **post-hoc repair** of cached LLM mathematical reasoning traces. The central goal is not only to improve final accuracy, but also to avoid replacing answers that were already correct. The repository therefore reports repair outcomes with fixed/broken accounting: fixed errors, broken-correct cases, accepted-repair precision, candidate flow, and direct-regeneration baselines.

## What is included

```text
.
├── src/                         # Core checking, diagnostics, parsing, evaluation, and model clients
├── data/                        # Prepared GSM8K, ASDiv, SVAMP, and MultiArith JSONL files used by experiments
├── artifacts/
│   ├── final_runs/              # Reported final run outputs and cached traces
│   ├── candidate_flow/          # Candidate-flow summary JSON files
│   └── semantic_graph_audit/    # Manual audit files for surface semantic-risk graph diagnostics
├── *.py                         # Main experiment and analysis scripts
├── requirements.txt             # Minimal Python dependencies
├── .env.example                 # Template for API key setup
└── README.md
```

The repository intentionally excludes local scratch outputs, historical snapshots, Python caches, macOS metadata, backup files, and secrets.

## Installation

Tested with Python 3.12 on macOS.

```bash
git clone <your-repo-url>
cd guarded-repair
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For API-based DeepSeek repair runs, create a local `.env` file:

```bash
cp .env.example .env
# edit .env and set DEEPSEEK_API_KEY
```

For local Qwen repair-model checks, install and run Ollama separately, then pull the relevant models, for example:

```bash
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b
```

## Main reported artifacts

The paper's reported runs are stored under `artifacts/final_runs/`.

Key runs:

```text
artifacts/final_runs/gsm8k_full_bestof3_relaxed_support
artifacts/final_runs/asdiv_random1000_qwen15b_bestof3_weak_relaxed_support
artifacts/final_runs/gsm8k_baseline_strong_solve_all
artifacts/final_runs/gsm8k_baseline_strong_solve_triggered
artifacts/final_runs/gsm8k_baseline_direct_bestof3_gated
artifacts/final_runs/gsm8k_ablation_guarded_n1
artifacts/final_runs/gsm8k_ablation_guarded_n2
artifacts/final_runs/gsm8k_ablation_n3_no_graph_guard
artifacts/final_runs/gsm8k_ablation_n3_no_equation_support
artifacts/final_runs/gsm8k_ablation_n3_relaxed_missing_constraint
artifacts/final_runs/svamp_full_qwen15b_guarded_n3
artifacts/final_runs/multiarith_full_qwen15b_guarded_n3
artifacts/final_runs/asdiv_seed42_qwen15b_repair_qwen25_7b_n3
artifacts/final_runs/asdiv_seed42_qwen15b_repair_qwen25_14b_n3
```

Each run directory may contain:

```text
final_report.json              # Summary metrics
raw_results.jsonl              # Per-example diagnostic and repair outputs
initial_reasoning_cache.jsonl  # Cached initial traces, when relevant
analysis_summary.json          # Downstream analysis summary, when generated
error_set.jsonl                # Remaining errors, when generated
fixed_cases.jsonl              # Fixed cases, when generated
broken_cases.jsonl             # Broken-correct cases, when generated
```

## Reproducing analysis from saved artifacts

You can recompute summary analyses from a saved run without regenerating model outputs.

Example: analyze the main GSM8K run from its saved raw results:

```bash
mkdir -p outputs/from_cache_guarded
cp artifacts/final_runs/gsm8k_full_bestof3_relaxed_support/raw_results.jsonl outputs/from_cache_guarded/raw_results.jsonl
GUARDED_OUTPUT_DIR=outputs/from_cache_guarded python analyze_guarded_results.py
```

Candidate-flow summaries can be generated from raw results:

```bash
RAW_RESULTS_PATH=artifacts/final_runs/gsm8k_full_bestof3_relaxed_support/raw_results.jsonl \
OUTPUT_PATH=outputs/candidate_flow_gsm8k.json \
python analyze_candidate_flow.py
```

Directly inspecting final reports:

```bash
cat artifacts/final_runs/gsm8k_full_bestof3_relaxed_support/final_report.json
cat artifacts/final_runs/asdiv_random1000_qwen15b_bestof3_weak_relaxed_support/final_report.json
```

## Running new experiments

The scripts use `config.py` and environment variables. The most common entry points are:

```bash
python generate_initial_cache.py
python generate_initial_cache_small_model.py
python run_pipeline_guarded_from_cache.py
python analyze_guarded_results.py
python analyze_candidate_flow.py
python run_strong_direct_baseline.py
```

Example guarded repair run from a cached initial-trace file:

```bash
GUARDED_OUTPUT_DIR=outputs/from_cache_guarded \
ENABLE_RELAXED_SUPPORT_ACCEPT=true \
LLM_REPAIR_NUM_CANDIDATES=3 \
python run_pipeline_guarded_from_cache.py
```

Example weak-reasoner run using a small local initial model:

```bash
SMALL_INITIAL_MODEL=qwen2.5:1.5b \
python generate_initial_cache_small_model.py

GUARDED_OUTPUT_DIR=outputs/asdiv_qwen15b_guarded_n3 \
ENABLE_WEAK_REASONER_RELAXED_ACCEPT=true \
LLM_REPAIR_NUM_CANDIDATES=3 \
python run_pipeline_guarded_from_cache.py
```

## Datasets

Prepared JSONL files are in `data/`. They include:

- `gsm8k_test_full.jsonl`
- `asdiv_numeric_random_1000_seed42.jsonl`
- `asdiv_numeric_random_1000_seed0.jsonl`
- `asdiv_numeric_random_1000_seed1.jsonl`
- `asdiv_numeric_random_1000_seed2.jsonl`
- `asdiv_numeric_full.jsonl`
- `svamp_full.jsonl`
- `multiarith_full.jsonl`

ASDiv numeric subsets are uniformly sampled from a filtered numeric pool before repair runs. The filtering and sampling scripts are included:

```bash
python prepare_asdiv.py
python prepare_asdiv_numeric_subset.py
```

## Important notes

- The repository does **not** include API keys. Use `.env.example` to create your own local `.env`.
- The reported API-based runs may not be exactly reproducible from scratch because provider-side model snapshots can change. Cached traces and raw result logs are included to reproduce the repair-stage analysis.
- Full ASDiv exploratory runs that were not used in the paper are intentionally excluded from the release package.
- `outputs/` is ignored by Git and should be used for local reruns.

## Citation

If you use this repository, please cite the paper:

```bibtex
@misc{xia2026guardedrepair,
  title={Guarded Repair for Harm-Aware Post-hoc Verification of LLM Mathematical Reasoning},
  author={Xia, Haizhou},
  year={2026},
  eprint={TBD},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
