"""
Candidate-flow analysis for existing guarded/direct raw_results.jsonl.

Usage:
  RAW_RESULTS_PATH=final_runs/gsm8k_guarded_n3/raw_results.jsonl \
  OUTPUT_PATH=outputs/candidate_flow_gsm8k.json \
  python analyze_candidate_flow.py
"""

import json
import os
from typing import Any, Dict, List

from src.evaluator import is_correct
from src.reasoning_parser import extract_final_answer

RAW_RESULTS_PATH = os.getenv("RAW_RESULTS_PATH", "outputs/from_cache_guarded/raw_results.jsonl")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "outputs/candidate_flow.json")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_eval(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("evaluation") or {}


def initial_correct(row: Dict[str, Any]) -> bool:
    ev = get_eval(row)
    if "initial_correct" in ev:
        return bool(ev["initial_correct"])
    return is_correct(extract_final_answer(row.get("initial_reasoning", "")), row.get("gold_answer", ""))


def final_correct(row: Dict[str, Any]) -> bool:
    ev = get_eval(row)
    if "final_correct" in ev:
        return bool(ev["final_correct"])
    return is_correct(extract_final_answer(row.get("final_reasoning", "")), row.get("gold_answer", ""))


def was_repaired(row: Dict[str, Any]) -> bool:
    ev = get_eval(row)
    return bool(ev.get("was_repaired", row.get("was_repaired", False)))


def repair_accepted(row: Dict[str, Any]) -> bool:
    ev = get_eval(row)
    return bool(ev.get("repair_accepted", row.get("repair_accepted", False)))


def candidate_answers(row: Dict[str, Any]) -> List[str]:
    answers: List[str] = []
    attempts = row.get("repair_attempts") or []
    if attempts:
        for att in attempts:
            cand = att.get("candidate", "") or ""
            ans = extract_final_answer(cand)
            if ans:
                answers.append(ans)
    else:
        cand = row.get("repair_candidate", "") or ""
        ans = extract_final_answer(cand)
        if ans:
            answers.append(ans)
    return answers


def candidate_correct_exists(row: Dict[str, Any]) -> bool:
    gold = row.get("gold_answer", "")
    return any(is_correct(ans, gold) for ans in candidate_answers(row))


def main() -> None:
    rows = load_jsonl(RAW_RESULTS_PATH)
    initial_wrong = [r for r in rows if not initial_correct(r)]
    initial_correct_rows = [r for r in rows if initial_correct(r)]
    triggered_wrong = [r for r in initial_wrong if was_repaired(r)]
    correct_candidate_wrong = [r for r in initial_wrong if candidate_correct_exists(r)]
    fixed = [r for r in rows if (not initial_correct(r)) and final_correct(r)]
    broken = [r for r in rows if initial_correct(r) and (not final_correct(r))]
    rejected_correct_wrong = [
        r for r in correct_candidate_wrong
        if not repair_accepted(r) or not final_correct(r)
    ]
    no_correct_candidate_wrong = [
        r for r in initial_wrong
        if not candidate_correct_exists(r)
    ]
    final_wrong = [r for r in rows if not final_correct(r)]

    summary = {
        "raw_results_path": RAW_RESULTS_PATH,
        "total": len(rows),
        "initial_correct": len(initial_correct_rows),
        "initial_wrong": len(initial_wrong),
        "triggered_wrong": len(triggered_wrong),
        "correct_candidate_exists_among_initial_wrong": len(correct_candidate_wrong),
        "accepted_correct_repair": len(fixed),
        "rejected_correct_repair": len(rejected_correct_wrong),
        "no_correct_candidate_among_initial_wrong": len(no_correct_candidate_wrong),
        "final_wrong": len(final_wrong),
        "broken_correct": len(broken),
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
