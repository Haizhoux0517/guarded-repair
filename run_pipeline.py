from tqdm import tqdm

from config import (
    FLASH_MODEL,
    PRO_MODEL,
    NUM_SAMPLES,
    DATASET_SEED,
    DATASET_SPLIT,
    LOCAL_DATASET_PATH,
    RAW_RESULTS_PATH,
    FINAL_REPORT_PATH,
    ERROR_SET_PATH,
    ERROR_ANALYSIS_PATH,
    ENABLE_ORACLE_ERROR_ANALYSIS,
    META_STEP_THRESHOLD,
    META_GLOBAL_THRESHOLD,
    REPAIR_MIN_IMPROVEMENT,
)

from src.dataset_loader import load_gsm8k
from src.deepseek_client import call_deepseek
from src.prompts import REASONER_PROMPT, REPAIR_PROMPT
from src.symbolic_checker import check_reasoning
from src.constraint_checker import check_constraint_coverage
from src.meta_diagnoser import compute_meta_diagnosis
from src.acceptance_gate import evaluate_candidate_reasoning, should_accept_repair
from src.evaluator import evaluate_result, summarize, build_error_set, is_correct
from src.reasoning_parser import extract_final_answer
from src.utils import save_jsonl, save_json


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


def should_generate_repair(meta_diagnosis):
    return meta_diagnosis["is_consistent"] is False


def diagnose_reasoning(problem, reasoning):
    """
    Unified diagnosis function.

    It combines:
    1. symbolic arithmetic checking
    2. constraint coverage checking
    3. meta-consistency scoring
    """

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

    return checker_results, constraint_results, meta_diagnosis


def run_experiment():
    samples = load_gsm8k(
        num_samples=NUM_SAMPLES,
        seed=DATASET_SEED,
        split=DATASET_SPLIT,
        local_path=LOCAL_DATASET_PATH,
        rebuild=True,
    )

    all_results = []
    oracle_error_analysis = []

    for idx, sample in enumerate(tqdm(samples, desc="Running experiment")):
        problem = sample["question"]
        gold_answer = sample["answer"]

        try:
            reasoner_messages = [
                {
                    "role": "user",
                    "content": REASONER_PROMPT.format(problem=problem),
                }
            ]

            initial_reasoning = call_deepseek(
                model=FLASH_MODEL,
                messages=reasoner_messages,
                temperature=0.0,
            )

        except RuntimeError as e:

            print(f"[Initial reasoning failed] sample_id={idx}, error={e}")

            result = {

                "id": idx,

                "problem": problem,

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

                    "explanation": "Initial reasoning generation failed.",

                },

                "repair_candidate": "",

                "repair_failed": True,

                "candidate_checker_results": None,

                "candidate_constraint_results": None,

                "candidate_meta_diagnosis": None,

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

            all_results.append(result)

            continue

        (
            initial_checker_results,
            initial_constraint_results,
            initial_meta_diagnosis,
        ) = diagnose_reasoning(problem, initial_reasoning)

        final_reasoning = initial_reasoning

        repair_candidate = ""
        candidate_checker_results = None
        candidate_constraint_results = None
        candidate_meta_diagnosis = None
        acceptance_decision = None

        was_repaired = False
        repair_accepted = False
        repair_failed = False

        if should_generate_repair(initial_meta_diagnosis):
            repair_candidate = run_repair(
                problem=problem,
                reasoning=initial_reasoning,
                diagnosis=initial_meta_diagnosis,
            )

            if repair_candidate.strip():
                was_repaired = True

                (
                    candidate_checker_results,
                    candidate_constraint_results,
                    candidate_meta_diagnosis,
                ) = diagnose_reasoning(problem, repair_candidate)

                acceptance_decision = should_accept_repair(
                    initial_meta=initial_meta_diagnosis,
                    candidate_meta=candidate_meta_diagnosis,
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

            "repair_candidate": repair_candidate,
            "repair_failed": repair_failed,
            "candidate_checker_results": candidate_checker_results,
            "candidate_constraint_results": candidate_constraint_results,
            "candidate_meta_diagnosis": candidate_meta_diagnosis,
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
                oracle_repair_candidate = run_repair(
                    problem=problem,
                    reasoning=initial_reasoning,
                    diagnosis=initial_meta_diagnosis,
                )

                oracle_candidate_checker_results = None
                oracle_candidate_constraint_results = None
                oracle_candidate_meta_diagnosis = None
                oracle_acceptance = None
                oracle_final_reasoning = initial_reasoning

                if oracle_repair_candidate.strip():
                    (
                        oracle_candidate_checker_results,
                        oracle_candidate_constraint_results,
                        oracle_candidate_meta_diagnosis,
                    ) = diagnose_reasoning(problem, oracle_repair_candidate)

                    oracle_acceptance = should_accept_repair(
                        initial_meta=initial_meta_diagnosis,
                        candidate_meta=oracle_candidate_meta_diagnosis,
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

                    "oracle_repair_candidate": oracle_repair_candidate,
                    "oracle_candidate_checker_results": oracle_candidate_checker_results,
                    "oracle_candidate_constraint_results": oracle_candidate_constraint_results,
                    "oracle_candidate_meta_diagnosis": oracle_candidate_meta_diagnosis,
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

    num_repair_failures = sum(1 for r in all_results if r.get("repair_failed"))
    report["num_repair_failures"] = num_repair_failures

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

    print("Experiment finished.")
    print(report)
    print(f"Saved raw results to: {RAW_RESULTS_PATH}")
    print(f"Saved final report to: {FINAL_REPORT_PATH}")
    print(f"Saved error set to: {ERROR_SET_PATH}")

    if ENABLE_ORACLE_ERROR_ANALYSIS:
        print(f"Saved oracle error analysis to: {ERROR_ANALYSIS_PATH}")


if __name__ == "__main__":
    run_experiment()