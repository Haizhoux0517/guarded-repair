"""
Strong-repair-model direct solving baselines for GSM8K.

Place this file at the project root, next to run_pipeline_guarded_from_cache.py.
It reuses the existing dataset loader, evaluator, parser, diagnostics, and guarded
acceptance functions from your current pipeline.

Modes:
  1) solve_all
     Strong repair model solves every GSM8K test problem from the problem only.
     No initial trace, no hint, no gates. Replaces all initial outputs.

  2) solve_triggered
     Strong repair model solves only the samples triggered by the main guarded run.
     Problem only. No initial trace, no hint, no gates. Untriggered samples keep the
     original initial trace.

  3) direct_bestof3_gated
     Strong repair model generates N problem-only candidates for the triggered samples.
     It uses the same graph guard + LLM acceptance policy as the main guarded repair.
     Difference from main system: no initial trace is shown to the model, and no
     deterministic diagnostic hint is shown to the model.

Example:
  DIRECT_BASELINE_MODE=solve_all \
  DIRECT_BASELINE_OUTPUT_DIR=outputs/direct_baselines/solve_all \
  python run_strong_direct_baseline.py

  DIRECT_BASELINE_MODE=solve_triggered \
  TRIGGER_SOURCE=final_runs/gsm8k_guarded_n3/raw_results.jsonl \
  DIRECT_BASELINE_OUTPUT_DIR=outputs/direct_baselines/solve_triggered \
  python run_strong_direct_baseline.py

  DIRECT_BASELINE_MODE=direct_bestof3_gated \
  TRIGGER_SOURCE=final_runs/gsm8k_guarded_n3/raw_results.jsonl \
  LLM_REPAIR_NUM_CANDIDATES=3 \
  ENABLE_RELAXED_SUPPORT_ACCEPT=true \
  DIRECT_BASELINE_OUTPUT_DIR=outputs/direct_baselines/direct_bestof3_gated \
  python run_strong_direct_baseline.py
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Reuse your existing pipeline implementation.
# This import intentionally uses the project-root file name.
import run_pipeline_guarded_from_cache as gp

from config import LOCAL_DATASET_PATH, PRO_MODEL
from src.deepseek_client import call_deepseek
from src.evaluator import evaluate_result, summarize


INITIAL_CACHE_PATH = os.getenv(
    "INITIAL_CACHE_PATH",
    "outputs/cache/initial_reasoning_cache.jsonl",
)
OUTPUT_DIR = os.getenv(
    "DIRECT_BASELINE_OUTPUT_DIR",
    "outputs/direct_baselines/strong_direct",
)
RAW_RESULTS_PATH = os.path.join(OUTPUT_DIR, "raw_results.jsonl")
FINAL_REPORT_PATH = os.path.join(OUTPUT_DIR, "final_report.json")
ERROR_SET_PATH = os.path.join(OUTPUT_DIR, "error_set.jsonl")

MODE = os.getenv("DIRECT_BASELINE_MODE", "solve_all").strip()
TRIGGER_SOURCE = os.getenv(
    "TRIGGER_SOURCE",
    "outputs/from_cache_guarded/raw_results.jsonl",
)
N = max(1, int(os.getenv("LLM_REPAIR_NUM_CANDIDATES", "1")))
MAX_TOKENS = int(os.getenv("DIRECT_SOLVE_MAX_TOKENS", "768"))
FORMAT_RETRY_MAX_TOKENS = int(os.getenv("DIRECT_SOLVE_FORMAT_RETRY_MAX_TOKENS", "512"))
ENABLE_FORMAT_RETRY = gp.env_bool("ENABLE_FORMAT_RETRY", True)


VALID_MODES = {
    "solve_all",
    "solve_triggered",
    "direct_bestof3_gated",
}


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def triggered_ids_from_source(path: str) -> set[str]:
    rows = load_jsonl(path)
    triggered: set[str] = set()
    for row in rows:
        ev = row.get("evaluation") or {}
        was_repaired = bool(ev.get("was_repaired", row.get("was_repaired", False)))
        if was_repaired:
            triggered.add(str(row.get("id")))
    if not triggered:
        raise ValueError(
            f"No triggered ids found in TRIGGER_SOURCE={path}. "
            "Make sure this is the main guarded raw_results.jsonl."
        )
    return triggered


def build_direct_solve_prompt(problem: str, attempt_idx: int = 0) -> str:
    if attempt_idx == 0:
        style = "Solve carefully using concise arithmetic steps."
    elif attempt_idx == 1:
        style = "Recompute independently and pay attention to every stated quantity."
    else:
        style = "Solve from scratch in the simplest valid way; avoid unnecessary explanation."

    return f"""
You are a strict math solver.

Return ONLY valid JSON.
Do NOT use markdown.
Do NOT write any explanation outside JSON.

The JSON schema must be exactly:
{{
  "steps": ["short arithmetic equation or short factual statement"],
  "final_answer": "number"
}}

Rules:
- Use at most 4 steps.
- Prefer number-only arithmetic equations.
- Do not put units inside equations.
- final_answer must contain only the answer number.
- Attempt strategy: {style}

Problem:
{problem}
""".strip()


def build_direct_format_retry_prompt(problem: str, bad_output: str) -> str:
    return f"""
Your previous output was not valid for the required format.

Rewrite the solution as ONLY valid JSON.
Do not include markdown.
Do not include explanation outside JSON.

Required JSON schema:
{{
  "steps": ["short arithmetic equation or short factual statement"],
  "final_answer": "number"
}}

Rules:
- Use at most 4 steps.
- Use the original problem as the source of truth.
- Prefer clean arithmetic equations.
- final_answer must contain only the answer number.

Problem:
{problem}

Invalid previous output:
{bad_output}
""".strip()


def call_direct_json(prompt: str, max_tokens: int) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict JSON-only math solver. "
                "Return only a valid JSON object. No markdown. No prose outside JSON."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return call_deepseek(
        model=PRO_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=1,
    )


def direct_solve_candidate(problem: str, attempt_idx: int = 0) -> Tuple[str, Dict[str, Any]]:
    first_prompt = build_direct_solve_prompt(problem, attempt_idx=attempt_idx)
    first_raw = call_direct_json(first_prompt, max_tokens=MAX_TOKENS)
    first_candidate = gp.json_repair_to_reasoning(first_raw)
    first_quality = gp.candidate_output_quality(first_candidate)
    metadata: Dict[str, Any] = {
        "attempt_idx": attempt_idx,
        "format_retry_used": False,
        "first_raw_response": first_raw,
        "first_candidate": first_candidate,
        "first_quality": first_quality,
        "retry_raw_response": "",
        "retry_candidate": "",
        "retry_quality": None,
    }
    if first_quality.get("valid", False) or not ENABLE_FORMAT_RETRY:
        return first_candidate, metadata

    retry_prompt = build_direct_format_retry_prompt(problem, first_raw)
    retry_raw = call_direct_json(retry_prompt, max_tokens=FORMAT_RETRY_MAX_TOKENS)
    retry_candidate = gp.json_repair_to_reasoning(retry_raw)
    retry_quality = gp.candidate_output_quality(retry_candidate)
    metadata.update(
        {
            "format_retry_used": True,
            "retry_raw_response": retry_raw,
            "retry_candidate": retry_candidate,
            "retry_quality": retry_quality,
        }
    )
    if retry_quality.get("valid", False):
        return retry_candidate, metadata
    return first_candidate, metadata


def make_base_result(sample: Dict[str, Any], cache_item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    problem = sample.get("problem", sample.get("question", ""))
    gold_answer = str(sample.get("answer", sample.get("gold_answer", ""))).strip()
    initial_reasoning = ""
    if cache_item:
        initial_reasoning = (
            cache_item.get("reasoning", "")
            or cache_item.get("initial_reasoning", "")
            or ""
        )
    return {
        "id": sample.get("id"),
        "problem": problem,
        "gold_answer": gold_answer,
        "initial_reasoning": initial_reasoning,
        "initial_checker_results": None,
        "initial_constraint_results": None,
        "initial_meta_diagnosis": None,
        "initial_semantic_graph": None,
        "deterministic_hint": "",
        "repair_candidate": "",
        "repair_failed": False,
        "repair_source": "",
        "llm_repair_metadata": {},
        "repair_attempts": [],
        "num_repair_attempts": 0,
        "candidate_checker_results": None,
        "candidate_constraint_results": None,
        "candidate_meta_diagnosis": None,
        "candidate_semantic_graph": None,
        "acceptance_decision": None,
        "graph_guard_decision": None,
        "final_reasoning": initial_reasoning,
        "was_repaired": False,
        "repair_accepted": False,
    }


def process_direct_ungated(
    sample: Dict[str, Any],
    cache_item: Optional[Dict[str, Any]],
    should_replace: bool,
) -> Dict[str, Any]:
    result = make_base_result(sample, cache_item)
    problem = result["problem"]
    if should_replace:
        result["was_repaired"] = True
        try:
            candidate, metadata = direct_solve_candidate(problem, attempt_idx=0)
            result["repair_candidate"] = candidate
            result["final_reasoning"] = candidate
            result["repair_accepted"] = True
            result["repair_source"] = "strong_direct_solve"
            result["llm_repair_metadata"] = metadata
            result["num_repair_attempts"] = 1
        except Exception as e:
            result["repair_failed"] = True
            result["repair_source"] = "strong_direct_solve_failed"
            result["llm_repair_metadata"] = {"error": str(e)}
    result["evaluation"] = evaluate_result(result)
    return result


def process_direct_bestofn_gated(
    sample: Dict[str, Any],
    cache_item: Optional[Dict[str, Any]],
    should_try: bool,
) -> Dict[str, Any]:
    result = make_base_result(sample, cache_item)
    problem = result["problem"]
    initial_reasoning = result["initial_reasoning"]

    # Always compute initial diagnostics for gated mode, because the existing
    # acceptance policy compares candidate diagnostics against the initial trace.
    initial_diag = gp.run_diagnostics(problem, initial_reasoning)
    result["initial_checker_results"] = initial_diag["checker_results"]
    result["initial_constraint_results"] = initial_diag["constraint_results"]
    result["initial_meta_diagnosis"] = initial_diag["meta_diagnosis"]
    result["initial_semantic_graph"] = initial_diag["semantic_graph"]

    if not should_try:
        result["evaluation"] = evaluate_result(result)
        return result

    result["was_repaired"] = True
    attempts: List[Dict[str, Any]] = []
    selected = None
    selected_meta: Dict[str, Any] = {}

    for attempt_idx in range(N):
        try:
            cand, meta = direct_solve_candidate(problem, attempt_idx=attempt_idx)
            repair_result = gp.try_candidate_repair(
                problem=problem,
                initial_reasoning=initial_reasoning,
                initial_meta=initial_diag["meta_diagnosis"],
                initial_graph=initial_diag["semantic_graph"],
                candidate=cand,
                source=f"llm_direct_solve_attempt_{attempt_idx}",
            )
            attempts.append(
                {
                    "attempt_idx": attempt_idx,
                    "candidate": repair_result.get("candidate", ""),
                    "accepted": bool(repair_result.get("accepted", False)),
                    "source": repair_result.get("source", ""),
                    "metadata": meta,
                    "acceptance_decision": repair_result.get("acceptance_decision"),
                    "graph_guard_decision": repair_result.get("graph_guard_decision"),
                    "candidate_meta_diagnosis": (
                        (repair_result.get("candidate_diagnostics") or {}).get("meta_diagnosis")
                    ),
                    "candidate_semantic_graph": (
                        (repair_result.get("candidate_diagnostics") or {}).get("semantic_graph")
                    ),
                }
            )
            if selected is None or gp.candidate_result_rank(repair_result) > gp.candidate_result_rank(selected):
                selected = repair_result
                selected_meta = meta
            if repair_result.get("accepted", False):
                selected = repair_result
                selected_meta = meta
                break
        except Exception as e:
            attempts.append(
                {
                    "attempt_idx": attempt_idx,
                    "candidate": "",
                    "accepted": False,
                    "source": f"llm_direct_solve_attempt_{attempt_idx}_failed",
                    "metadata": {"error": str(e)},
                    "acceptance_decision": None,
                    "graph_guard_decision": None,
                    "candidate_meta_diagnosis": None,
                    "candidate_semantic_graph": None,
                }
            )

    result["repair_attempts"] = attempts
    result["num_repair_attempts"] = len(attempts)
    result["llm_repair_metadata"] = {
        **selected_meta,
        "num_candidates_requested": N,
        "num_candidates_generated": len(attempts),
        "direct_solve_baseline": True,
    }

    if selected is not None:
        diag = selected.get("candidate_diagnostics") or {}
        result["repair_candidate"] = selected.get("candidate", "")
        result["repair_source"] = selected.get("source", "")
        result["candidate_checker_results"] = diag.get("checker_results")
        result["candidate_constraint_results"] = diag.get("constraint_results")
        result["candidate_meta_diagnosis"] = diag.get("meta_diagnosis")
        result["candidate_semantic_graph"] = diag.get("semantic_graph")
        result["acceptance_decision"] = selected.get("acceptance_decision")
        result["graph_guard_decision"] = selected.get("graph_guard_decision")
        result["repair_accepted"] = bool(selected.get("accepted", False))
        if result["repair_accepted"]:
            result["final_reasoning"] = result["repair_candidate"]

    result["evaluation"] = evaluate_result(result)
    return result


def main() -> None:
    if MODE not in VALID_MODES:
        raise ValueError(f"DIRECT_BASELINE_MODE must be one of {sorted(VALID_MODES)}, got {MODE!r}")

    ensure_output_dir()
    dataset = gp.load_local_jsonl_dataset(LOCAL_DATASET_PATH)
    cache = gp.load_initial_cache(INITIAL_CACHE_PATH)
    triggered_ids: set[str] = set()
    if MODE in {"solve_triggered", "direct_bestof3_gated"}:
        triggered_ids = triggered_ids_from_source(TRIGGER_SOURCE)
        print(f"Loaded {len(triggered_ids)} triggered ids from {TRIGGER_SOURCE}")

    results: List[Dict[str, Any]] = []
    with open(RAW_RESULTS_PATH, "w", encoding="utf-8") as raw_f:
        for sample in tqdm(dataset, desc=f"Running direct baseline: {MODE}"):
            sample_id = str(sample.get("id"))
            cache_item = cache.get(sample_id)
            if MODE == "solve_all":
                result = process_direct_ungated(sample, cache_item, should_replace=True)
            elif MODE == "solve_triggered":
                result = process_direct_ungated(
                    sample,
                    cache_item,
                    should_replace=(sample_id in triggered_ids),
                )
            else:
                result = process_direct_bestofn_gated(
                    sample,
                    cache_item,
                    should_try=(sample_id in triggered_ids),
                )
            results.append(result)
            raw_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            raw_f.flush()

    report = summarize(results)
    total_attempts = sum(int(r.get("num_repair_attempts", 0) or 0) for r in results)
    format_retry_used = sum(
        1 for r in results
        if (r.get("llm_repair_metadata") or {}).get("format_retry_used", False)
    )
    report.update(
        {
            "baseline_mode": MODE,
            "model": PRO_MODEL,
            "local_dataset_path": LOCAL_DATASET_PATH,
            "cache_path": INITIAL_CACHE_PATH,
            "trigger_source": TRIGGER_SOURCE if triggered_ids else "",
            "triggered_only": MODE in {"solve_triggered", "direct_bestof3_gated"},
            "uses_initial_trace_in_prompt": False,
            "uses_diagnostic_hint_in_prompt": False,
            "uses_gates": MODE == "direct_bestof3_gated",
            "num_candidates": N if MODE == "direct_bestof3_gated" else 1,
            "total_llm_repair_attempts": total_attempts,
            "format_retry_used": format_retry_used,
            "max_tokens": MAX_TOKENS,
            "format_retry_max_tokens": FORMAT_RETRY_MAX_TOKENS,
        }
    )
    save_json(FINAL_REPORT_PATH, report)
    with open(ERROR_SET_PATH, "w", encoding="utf-8") as f:
        for row in results:
            if not row.get("evaluation", {}).get("final_correct", False):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Direct baseline finished.")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved raw results to: {RAW_RESULTS_PATH}")
    print(f"Saved final report to: {FINAL_REPORT_PATH}")
    print(f"Saved error set to: {ERROR_SET_PATH}")


if __name__ == "__main__":
    main()
