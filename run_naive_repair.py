from tqdm import tqdm

from config import (
    FLASH_MODEL,
    PRO_MODEL,
    NUM_SAMPLES,
    DATASET_SEED,
    DATASET_SPLIT,
    LOCAL_DATASET_PATH,
    ENABLE_ORACLE_ERROR_ANALYSIS,
    META_STEP_THRESHOLD,
    META_GLOBAL_THRESHOLD,
)

from src.dataset_loader import load_gsm8k
from src.deepseek_client import call_deepseek
from src.prompts import REASONER_PROMPT, REPAIR_PROMPT
from src.symbolic_checker import check_reasoning
from src.constraint_checker import check_constraint_coverage
from src.meta_diagnoser import compute_meta_diagnosis
from src.evaluator import evaluate_result, summarize, build_error_set, is_correct
from src.reasoning_parser import extract_final_answer
from src.utils import save_jsonl, save_json


OUTPUT_DIR = "outputs/ablation_naive_repair"
RAW_RESULTS_PATH = f"{OUTPUT_DIR}/raw_results.jsonl"
FINAL_REPORT_PATH = f"{OUTPUT_DIR}/final_report.json"
ERROR_SET_PATH = f"{OUTPUT_DIR}/error_set.jsonl"
ERROR_ANALYSIS_PATH = f"{OUTPUT_DIR}/error_analysis.jsonl"


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

    return checker_results, constraint_results, meta_diagnosis


def should_generate_repair(meta_diagnosis):
    return meta_diagnosis["is_consistent"] is False


def run_experiment():
    samples = load_gsm8k(
        num_samples=NUM_SAMPLES,
        seed=DATASET_SEED,
        split=DATASET_SPLIT,
        local_path=LOCAL_DATASET_PATH,
        rebuild=False,
    )

    all_results = []
    oracle_error_analysis = []

    for idx, sample in enumerate(tqdm(samples, desc="Running naive repair ablation")):
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
        repair_failed = False
        was_repaired = False

        # Naive repair:
        # if meta-diagnosis says unreliable, generate repair and directly accept it.
        if should_generate_repair(initial_meta_diagnosis):
            repair_candidate = run_repair(
                problem=problem,
                reasoning=initial_reasoning,
                diagnosis=initial_meta_diagnosis,
            )

            if repair_candidate.strip():
                final_reasoning = repair_candidate
                was_repaired = True
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

            "final_reasoning": final_reasoning,
            "was_repaired": was_repaired,
            "repair_accepted": was_repaired,
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

                oracle_final_reasoning = (
                    oracle_repair_candidate
                    if oracle_repair_candidate.strip()
                    else initial_reasoning
                )

                oracle_item = {
                    "id": idx,
                    "problem": problem,
                    "gold_answer": gold_answer,
                    "initial_answer": initial_answer,
                    "initial_reasoning": initial_reasoning,
                    "initial_meta_diagnosis": initial_meta_diagnosis,
                    "oracle_repair_candidate": oracle_repair_candidate,
                    "oracle_final_reasoning": oracle_final_reasoning,
                    "oracle_repair_failed": not bool(oracle_repair_candidate.strip()),
                }

                oracle_item["oracle_evaluation"] = {
                    "candidate_answer": extract_final_answer(oracle_repair_candidate),
                    "candidate_correct": is_correct(
                        extract_final_answer(oracle_repair_candidate),
                        gold_answer,
                    ),
                    "final_answer_after_naive_repair": extract_final_answer(
                        oracle_final_reasoning
                    ),
                    "final_correct_after_naive_repair": is_correct(
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

    if ENABLE_ORACLE_ERROR_ANALYSIS:
        oracle_total = len(oracle_error_analysis)

        oracle_candidate_fixed = sum(
            item["oracle_evaluation"]["candidate_correct"]
            for item in oracle_error_analysis
        )

        oracle_naive_fixed = sum(
            item["oracle_evaluation"]["final_correct_after_naive_repair"]
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
        report["oracle_naive_fixed"] = oracle_naive_fixed
        report["oracle_naive_repair_rate"] = (
            oracle_naive_fixed / oracle_total if oracle_total else 0
        )
        report["oracle_repair_failures"] = oracle_repair_failures

    save_jsonl(RAW_RESULTS_PATH, all_results)
    save_json(FINAL_REPORT_PATH, report)
    save_jsonl(ERROR_SET_PATH, error_set)

    if ENABLE_ORACLE_ERROR_ANALYSIS:
        save_jsonl(ERROR_ANALYSIS_PATH, oracle_error_analysis)

    print("Naive repair ablation finished.")
    print(report)
    print(f"Saved raw results to: {RAW_RESULTS_PATH}")
    print(f"Saved final report to: {FINAL_REPORT_PATH}")
    print(f"Saved error set to: {ERROR_SET_PATH}")

    if ENABLE_ORACLE_ERROR_ANALYSIS:
        print(f"Saved oracle error analysis to: {ERROR_ANALYSIS_PATH}")


if __name__ == "__main__":
    run_experiment()