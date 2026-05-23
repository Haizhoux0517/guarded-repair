################################
#
# LOSER，失败的 LLM-semantic-validator ablation
#
################################

import json
from tqdm import tqdm

from config import (
    PRO_MODEL,
    META_STEP_THRESHOLD,
    META_GLOBAL_THRESHOLD,
    REPAIR_MIN_IMPROVEMENT,
    ENABLE_ORACLE_ERROR_ANALYSIS,
)

from src.deepseek_client import call_deepseek
from src.prompts import REPAIR_PROMPT
from src.symbolic_checker import check_reasoning
from src.constraint_checker import check_constraint_coverage
from src.meta_diagnoser import compute_meta_diagnosis
from src.semantic_validator import validate_semantics
from src.evaluator import evaluate_result, summarize, build_error_set, is_correct
from src.reasoning_parser import extract_final_answer
from src.utils import save_jsonl, save_json


INITIAL_CACHE_PATH = "outputs/cache/initial_reasoning_cache.jsonl"

OUTPUT_DIR = "outputs/from_cache_semantic"
RAW_RESULTS_PATH = f"{OUTPUT_DIR}/raw_results.jsonl"
FINAL_REPORT_PATH = f"{OUTPUT_DIR}/final_report.json"
ERROR_SET_PATH = f"{OUTPUT_DIR}/error_set.jsonl"
ERROR_ANALYSIS_PATH = f"{OUTPUT_DIR}/error_analysis.jsonl"


SEMANTIC_TRIGGER_THRESHOLD = 0.75
SEMANTIC_ACCEPT_THRESHOLD = 0.70


def load_jsonl(path):
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    return items


def run_repair(problem, reasoning, diagnosis):
    repair_messages = [
        {
            "role": "user",
            "content": REPAIR_PROMPT.format(
                problem=problem,
                reasoning=reasoning,
                diagnosis=diagnosis,
            ),
        }
    ]

    try:
        repaired = call_deepseek(
            model=PRO_MODEL,
            messages=repair_messages,
            temperature=0.0,
        )

        if repaired and repaired.strip():
            return repaired.strip()

        return ""

    except RuntimeError as e:
        print(f"[Repair failed] {e}")
        return ""


def diagnose_reasoning(problem, reasoning):
    checker_results = check_reasoning(reasoning)

    constraint_results = check_constraint_coverage(
        problem=problem,
        reasoning=reasoning,
    )

    meta_diagnosis = compute_meta_diagnosis(
        checker_results=checker_results,
        constraint_results=constraint_results,
        step_threshold=META_STEP_THRESHOLD,
        global_threshold=META_GLOBAL_THRESHOLD,
    )

    semantic_diagnosis = validate_semantics(
        problem=problem,
        reasoning=reasoning,
    )

    return (
        checker_results,
        constraint_results,
        meta_diagnosis,
        semantic_diagnosis,
    )


def should_generate_repair(meta_diagnosis, semantic_diagnosis):
    if meta_diagnosis.get("is_consistent") is False:
        return True

    if semantic_diagnosis.get("needs_repair") is True:
        return True

    if semantic_diagnosis.get("semantic_score", 1.0) < SEMANTIC_TRIGGER_THRESHOLD:
        return True

    return False


def semantic_acceptance_decision(
    initial_meta,
    candidate_meta,
    initial_semantic,
    candidate_semantic,
    min_improvement,
):
    """
    Acceptance gate that combines:
    1. meta-consistency improvement
    2. semantic validity
    3. no major semantic regression
    """

    initial_meta_score = initial_meta.get("global_consistency_score", 0.0)
    candidate_meta_score = candidate_meta.get("global_consistency_score", 0.0)

    initial_sem_score = initial_semantic.get("semantic_score", 1.0)
    candidate_sem_score = candidate_semantic.get("semantic_score", 1.0)

    meta_improvement = candidate_meta_score - initial_meta_score
    semantic_improvement = candidate_sem_score - initial_sem_score

    candidate_semantically_valid = candidate_semantic.get("is_semantically_valid", True)
    candidate_needs_repair = candidate_semantic.get("needs_repair", False)

    if not candidate_semantically_valid:
        return {
            "accept": False,
            "initial_meta_score": initial_meta_score,
            "candidate_meta_score": candidate_meta_score,
            "initial_semantic_score": initial_sem_score,
            "candidate_semantic_score": candidate_sem_score,
            "meta_improvement": meta_improvement,
            "semantic_improvement": semantic_improvement,
            "reason": "Rejected because candidate is semantically invalid.",
        }

    if candidate_needs_repair:
        return {
            "accept": False,
            "initial_meta_score": initial_meta_score,
            "candidate_meta_score": candidate_meta_score,
            "initial_semantic_score": initial_sem_score,
            "candidate_semantic_score": candidate_sem_score,
            "meta_improvement": meta_improvement,
            "semantic_improvement": semantic_improvement,
            "reason": "Rejected because semantic validator says candidate still needs repair.",
        }

    if candidate_sem_score < SEMANTIC_ACCEPT_THRESHOLD:
        return {
            "accept": False,
            "initial_meta_score": initial_meta_score,
            "candidate_meta_score": candidate_meta_score,
            "initial_semantic_score": initial_sem_score,
            "candidate_semantic_score": candidate_sem_score,
            "meta_improvement": meta_improvement,
            "semantic_improvement": semantic_improvement,
            "reason": "Rejected because candidate semantic score is too low.",
        }

    if meta_improvement >= min_improvement or semantic_improvement > 0:
        return {
            "accept": True,
            "initial_meta_score": initial_meta_score,
            "candidate_meta_score": candidate_meta_score,
            "initial_semantic_score": initial_sem_score,
            "candidate_semantic_score": candidate_sem_score,
            "meta_improvement": meta_improvement,
            "semantic_improvement": semantic_improvement,
            "reason": "Accepted because candidate improves meta-consistency or semantic validity.",
        }

    return {
        "accept": False,
        "initial_meta_score": initial_meta_score,
        "candidate_meta_score": candidate_meta_score,
        "initial_semantic_score": initial_sem_score,
        "candidate_semantic_score": candidate_sem_score,
        "meta_improvement": meta_improvement,
        "semantic_improvement": semantic_improvement,
        "reason": "Rejected because candidate does not improve reliability.",
    }


def build_generation_failure_result(item):
    gold_answer = item["gold_answer"]

    result = {
        "id": item["id"],
        "problem": item["problem"],
        "gold_answer": gold_answer,

        "initial_reasoning": "",
        "initial_checker_results": None,
        "initial_constraint_results": None,
        "initial_meta_diagnosis": {
            "is_consistent": False,
            "global_consistency_score": 0.0,
            "average_step_score": 0.0,
            "symbolic_coverage": 0.0,
            "constraint_coverage_score": 0.0,
            "missing_constraints": [],
            "error_step": None,
            "error_type": "generation_failure",
            "step_scores": [],
            "explanation": "Initial reasoning generation failed in cached baseline.",
        },
        "initial_semantic_diagnosis": {
            "is_semantically_valid": False,
            "semantic_score": 0.0,
            "error_type": "generation_failure",
            "needs_repair": True,
            "suspected_issue": "empty_reasoning",
            "explanation": "Initial reasoning generation failed.",
        },

        "repair_candidate": "",
        "repair_failed": True,
        "candidate_checker_results": None,
        "candidate_constraint_results": None,
        "candidate_meta_diagnosis": None,
        "candidate_semantic_diagnosis": None,
        "acceptance_decision": None,

        "final_reasoning": "",
        "was_repaired": False,
        "repair_accepted": False,
    }

    result["evaluation"] = {
        "initial_answer": None,
        "final_answer": None,
        "gold_answer": gold_answer,

        "initial_correct": False,
        "final_correct": False,

        "was_repaired": False,
        "repair_accepted": False,

        "fixed_error": False,
        "broke_correct": False,
        "failed_to_fix": True,
    }

    return result


def run_experiment():
    cache_items = load_jsonl(INITIAL_CACHE_PATH)

    all_results = []
    oracle_error_analysis = []

    for item in tqdm(cache_items, desc="Running semantic pipeline from cache"):
        idx = item["id"]
        problem = item["problem"]
        gold_answer = item["gold_answer"]
        initial_reasoning = item.get("initial_reasoning", "")

        if item.get("generation_failed") or not initial_reasoning.strip():
            result = build_generation_failure_result(item)
            all_results.append(result)
            continue

        (
            initial_checker_results,
            initial_constraint_results,
            initial_meta_diagnosis,
            initial_semantic_diagnosis,
        ) = diagnose_reasoning(problem, initial_reasoning)

        final_reasoning = initial_reasoning

        repair_candidate = ""
        candidate_checker_results = None
        candidate_constraint_results = None
        candidate_meta_diagnosis = None
        candidate_semantic_diagnosis = None
        acceptance_decision = None

        was_repaired = False
        repair_accepted = False
        repair_failed = False

        if should_generate_repair(initial_meta_diagnosis, initial_semantic_diagnosis):
            repair_diagnosis = {
                "meta_diagnosis": initial_meta_diagnosis,
                "semantic_diagnosis": initial_semantic_diagnosis,
            }

            repair_candidate = run_repair(
                problem=problem,
                reasoning=initial_reasoning,
                diagnosis=repair_diagnosis,
            )

            if repair_candidate.strip():
                was_repaired = True

                (
                    candidate_checker_results,
                    candidate_constraint_results,
                    candidate_meta_diagnosis,
                    candidate_semantic_diagnosis,
                ) = diagnose_reasoning(problem, repair_candidate)

                acceptance_decision = semantic_acceptance_decision(
                    initial_meta=initial_meta_diagnosis,
                    candidate_meta=candidate_meta_diagnosis,
                    initial_semantic=initial_semantic_diagnosis,
                    candidate_semantic=candidate_semantic_diagnosis,
                    min_improvement=REPAIR_MIN_IMPROVEMENT,
                )

                if acceptance_decision["accept"]:
                    final_reasoning = repair_candidate
                    repair_accepted = True

            else:
                repair_failed = True

        result = {
            "id": idx,
            "problem": problem,
            "gold_answer": gold_answer,

            "initial_reasoning": initial_reasoning,
            "initial_checker_results": initial_checker_results,
            "initial_constraint_results": initial_constraint_results,
            "initial_meta_diagnosis": initial_meta_diagnosis,
            "initial_semantic_diagnosis": initial_semantic_diagnosis,

            "repair_candidate": repair_candidate,
            "repair_failed": repair_failed,
            "candidate_checker_results": candidate_checker_results,
            "candidate_constraint_results": candidate_constraint_results,
            "candidate_meta_diagnosis": candidate_meta_diagnosis,
            "candidate_semantic_diagnosis": candidate_semantic_diagnosis,
            "acceptance_decision": acceptance_decision,

            "final_reasoning": final_reasoning,
            "was_repaired": was_repaired,
            "repair_accepted": repair_accepted,
        }

        result["evaluation"] = evaluate_result(result)
        all_results.append(result)

        if ENABLE_ORACLE_ERROR_ANALYSIS:
            initial_answer = extract_final_answer(initial_reasoning)
            initial_correct = is_correct(initial_answer, gold_answer)

            if not initial_correct:
                oracle_repair_diagnosis = {
                    "meta_diagnosis": initial_meta_diagnosis,
                    "semantic_diagnosis": initial_semantic_diagnosis,
                }

                oracle_repair_candidate = run_repair(
                    problem=problem,
                    reasoning=initial_reasoning,
                    diagnosis=oracle_repair_diagnosis,
                )

                oracle_candidate_checker_results = None
                oracle_candidate_constraint_results = None
                oracle_candidate_meta_diagnosis = None
                oracle_candidate_semantic_diagnosis = None
                oracle_acceptance = None
                oracle_final_reasoning = initial_reasoning

                if oracle_repair_candidate.strip():
                    (
                        oracle_candidate_checker_results,
                        oracle_candidate_constraint_results,
                        oracle_candidate_meta_diagnosis,
                        oracle_candidate_semantic_diagnosis,
                    ) = diagnose_reasoning(problem, oracle_repair_candidate)

                    oracle_acceptance = semantic_acceptance_decision(
                        initial_meta=initial_meta_diagnosis,
                        candidate_meta=oracle_candidate_meta_diagnosis,
                        initial_semantic=initial_semantic_diagnosis,
                        candidate_semantic=oracle_candidate_semantic_diagnosis,
                        min_improvement=REPAIR_MIN_IMPROVEMENT,
                    )

                    if oracle_acceptance["accept"]:
                        oracle_final_reasoning = oracle_repair_candidate

                oracle_item = {
                    "id": idx,
                    "problem": problem,
                    "gold_answer": gold_answer,
                    "initial_answer": initial_answer,
                    "initial_reasoning": initial_reasoning,

                    "initial_checker_results": initial_checker_results,
                    "initial_constraint_results": initial_constraint_results,
                    "initial_meta_diagnosis": initial_meta_diagnosis,
                    "initial_semantic_diagnosis": initial_semantic_diagnosis,

                    "oracle_repair_candidate": oracle_repair_candidate,
                    "oracle_candidate_checker_results": oracle_candidate_checker_results,
                    "oracle_candidate_constraint_results": oracle_candidate_constraint_results,
                    "oracle_candidate_meta_diagnosis": oracle_candidate_meta_diagnosis,
                    "oracle_candidate_semantic_diagnosis": oracle_candidate_semantic_diagnosis,
                    "oracle_acceptance": oracle_acceptance,
                    "oracle_final_reasoning": oracle_final_reasoning,
                    "oracle_repair_failed": not bool(oracle_repair_candidate.strip()),
                }

                oracle_item["oracle_evaluation"] = {
                    "candidate_answer": extract_final_answer(oracle_repair_candidate),
                    "candidate_correct": is_correct(
                        extract_final_answer(oracle_repair_candidate),
                        gold_answer,
                    ),
                    "final_answer_after_gate": extract_final_answer(oracle_final_reasoning),
                    "final_correct_after_gate": is_correct(
                        extract_final_answer(oracle_final_reasoning),
                        gold_answer,
                    ),
                }

                oracle_error_analysis.append(oracle_item)

    report = summarize(all_results)
    error_set = build_error_set(all_results)

    report["num_repair_failures"] = sum(
        1 for r in all_results if r.get("repair_failed")
    )

    report["cache_path"] = INITIAL_CACHE_PATH
    report["from_cache"] = True
    report["semantic_validator"] = True
    report["semantic_trigger_threshold"] = SEMANTIC_TRIGGER_THRESHOLD
    report["semantic_accept_threshold"] = SEMANTIC_ACCEPT_THRESHOLD

    if ENABLE_ORACLE_ERROR_ANALYSIS:
        oracle_total = len(oracle_error_analysis)

        oracle_candidate_fixed = sum(
            item["oracle_evaluation"]["candidate_correct"]
            for item in oracle_error_analysis
        )

        oracle_gate_fixed = sum(
            item["oracle_evaluation"]["final_correct_after_gate"]
            for item in oracle_error_analysis
        )

        oracle_repair_failures = sum(
            item.get("oracle_repair_failed", False)
            for item in oracle_error_analysis
        )

        report["oracle_error_analysis_total"] = oracle_total
        report["oracle_candidate_fixed"] = oracle_candidate_fixed
        report["oracle_candidate_repair_rate"] = (
            oracle_candidate_fixed / oracle_total if oracle_total else 0
        )
        report["oracle_gate_fixed"] = oracle_gate_fixed
        report["oracle_gate_repair_rate"] = (
            oracle_gate_fixed / oracle_total if oracle_total else 0
        )
        report["oracle_repair_failures"] = oracle_repair_failures

    save_jsonl(RAW_RESULTS_PATH, all_results)
    save_json(FINAL_REPORT_PATH, report)
    save_jsonl(ERROR_SET_PATH, error_set)

    if ENABLE_ORACLE_ERROR_ANALYSIS:
        save_jsonl(ERROR_ANALYSIS_PATH, oracle_error_analysis)

    print("Semantic pipeline from cache finished.")
    print(report)
    print(f"Saved raw results to: {RAW_RESULTS_PATH}")
    print(f"Saved final report to: {FINAL_REPORT_PATH}")
    print(f"Saved error set to: {ERROR_SET_PATH}")

    if ENABLE_ORACLE_ERROR_ANALYSIS:
        print(f"Saved oracle error analysis to: {ERROR_ANALYSIS_PATH}")


if __name__ == "__main__":
    run_experiment()