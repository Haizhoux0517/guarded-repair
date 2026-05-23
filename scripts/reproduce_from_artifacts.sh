#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/from_cache_guarded
cp artifacts/final_runs/gsm8k_full_bestof3_relaxed_support/raw_results.jsonl outputs/from_cache_guarded/raw_results.jsonl
GUARDED_OUTPUT_DIR=outputs/from_cache_guarded python analyze_guarded_results.py

RAW_RESULTS_PATH=artifacts/final_runs/gsm8k_full_bestof3_relaxed_support/raw_results.jsonl \
OUTPUT_PATH=outputs/candidate_flow_gsm8k.json \
python analyze_candidate_flow.py

cat artifacts/final_runs/gsm8k_full_bestof3_relaxed_support/final_report.json
cat artifacts/final_runs/asdiv_random1000_qwen15b_bestof3_weak_relaxed_support/final_report.json
