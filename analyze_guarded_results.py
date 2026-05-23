import json
import os
import re
from collections import Counter
from typing import Any, Dict, List


GUARDED_OUTPUT_DIR = os.getenv("GUARDED_OUTPUT_DIR", "outputs/from_cache_guarded")

RAW_RESULTS_PATH = os.getenv(
    "RAW_RESULTS_PATH",
    os.path.join(GUARDED_OUTPUT_DIR, "raw_results.jsonl"),
)

FINAL_REPORT_PATH = os.getenv(
    "FINAL_REPORT_PATH",
    os.path.join(GUARDED_OUTPUT_DIR, "final_report.json"),
)

ANALYSIS_OUTPUT_DIR = os.getenv(
    "ANALYSIS_OUTPUT_DIR",
    os.path.join(GUARDED_OUTPUT_DIR, "analysis"),
)

SUMMARY_PATH = os.path.join(ANALYSIS_OUTPUT_DIR, "analysis_summary.json")

FIXED_CASES_PATH = os.path.join(ANALYSIS_OUTPUT_DIR, "fixed_cases.jsonl")
BROKEN_CASES_PATH = os.path.join(ANALYSIS_OUTPUT_DIR, "broken_cases.jsonl")
REMAINING_ERRORS_PATH = os.path.join(ANALYSIS_OUTPUT_DIR, "remaining_errors.jsonl")

CORRECT_BUT_REJECTED_PATH = os.path.join(
    ANALYSIS_OUTPUT_DIR,
    "correct_but_rejected.jsonl",
)

INITIALLY_WRONG_CORRECT_BUT_REJECTED_PATH = os.path.join(
    ANALYSIS_OUTPUT_DIR,
    "initially_wrong_candidate_correct_but_rejected.jsonl",
)

INITIALLY_CORRECT_CORRECT_BUT_REJECTED_PATH = os.path.join(
    ANALYSIS_OUTPUT_DIR,
    "initially_correct_candidate_correct_but_rejected.jsonl",
)

INITIALLY_WRONG_GRAPH_GUARD_REJECTED_PATH = os.path.join(
    ANALYSIS_OUTPUT_DIR,
    "initially_wrong_graph_guard_rejected.jsonl",
)

INITIALLY_WRONG_FORMAT_FAILED_PATH = os.path.join(
    ANALYSIS_OUTPUT_DIR,
    "initially_wrong_format_failed.jsonl",
)

INITIALLY_WRONG_CANDIDATE_WRONG_PATH = os.path.join(
    ANALYSIS_OUTPUT_DIR,
    "initially_wrong_candidate_wrong.jsonl",
)

INITIALLY_WRONG_NO_CANDIDATE_PATH = os.path.join(
    ANALYSIS_OUTPUT_DIR,
    "initially_wrong_no_candidate.jsonl",
)

REJECTED_BY_REASON_PATH = os.path.join(
    ANALYSIS_OUTPUT_DIR,
    "rejected_by_reason.jsonl",
)

LLM_FORMAT_FAILURES_PATH = os.path.join(
    ANALYSIS_OUTPUT_DIR,
    "llm_format_failures.jsonl",
)


def ensure_output_dir() -> None:
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSONL at {path}, line {line_idx}: {e}"
                ) from e

    return rows


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer(ans: Any) -> str:
    text = str(ans or "").strip()
    text = text.replace(",", "")

    try:
        value = float(text)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except Exception:
        return text


def answers_equal(a: Any, b: Any) -> bool:
    return normalize_answer(a) == normalize_answer(b)


def answers_differ(a: Any, b: Any) -> bool:
    return not answers_equal(a, b)


def get_eval(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("evaluation", {}) or {}


def get_gold_answer(row: Dict[str, Any]) -> str:
    return str(row.get("gold_answer", "")).strip()


def get_initial_answer(row: Dict[str, Any]) -> str:
    return str(get_eval(row).get("initial_answer", "") or "").strip()


def get_final_answer(row: Dict[str, Any]) -> str:
    return str(get_eval(row).get("final_answer", "") or "").strip()


def get_acceptance_reason(row: Dict[str, Any]) -> str:
    decision = row.get("acceptance_decision") or {}
    return str(decision.get("reason", "") or "")


def get_graph_reason(row: Dict[str, Any]) -> str:
    decision = row.get("graph_guard_decision") or {}
    return str(decision.get("reason", "") or "")


def extract_final_answer_from_text(text: str) -> str:
    if not text:
        return ""

    matches = re.findall(
        r"Final Answer\s*:\s*([^\n\r]+)",
        text,
        flags=re.IGNORECASE,
    )

    if not matches:
        return ""

    answer = matches[-1].strip()
    answer = re.sub(r"[.$。]+$", "", answer)

    numeric_match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer)
    if numeric_match:
        return numeric_match.group(0)

    return answer


def get_candidate_answer(row: Dict[str, Any]) -> str:
    candidate = row.get("repair_candidate", "") or ""
    return extract_final_answer_from_text(candidate)


def candidate_exists(row: Dict[str, Any]) -> bool:
    return bool((row.get("repair_candidate", "") or "").strip())


def candidate_is_correct(row: Dict[str, Any]) -> bool:
    candidate_answer = get_candidate_answer(row)
    gold_answer = get_gold_answer(row)

    return bool(candidate_answer) and answers_equal(candidate_answer, gold_answer)


def candidate_is_wrong(row: Dict[str, Any]) -> bool:
    candidate_answer = get_candidate_answer(row)
    gold_answer = get_gold_answer(row)

    return bool(candidate_answer) and not answers_equal(candidate_answer, gold_answer)


def initial_is_correct(row: Dict[str, Any]) -> bool:
    eval_data = get_eval(row)

    if "initial_correct" in eval_data:
        return bool(eval_data.get("initial_correct"))

    return answers_equal(get_initial_answer(row), get_gold_answer(row))


def final_is_correct(row: Dict[str, Any]) -> bool:
    eval_data = get_eval(row)

    if "final_correct" in eval_data:
        return bool(eval_data.get("final_correct"))

    return answers_equal(get_final_answer(row), get_gold_answer(row))


def repair_was_accepted(row: Dict[str, Any]) -> bool:
    eval_data = get_eval(row)

    if "repair_accepted" in eval_data:
        return bool(eval_data.get("repair_accepted"))

    return bool(row.get("repair_accepted", False))


def repair_was_attempted(row: Dict[str, Any]) -> bool:
    eval_data = get_eval(row)

    if "was_repaired" in eval_data:
        return bool(eval_data.get("was_repaired"))

    return bool(row.get("was_repaired", False))


def classify_rejection(row: Dict[str, Any]) -> str:
    acceptance_decision = row.get("acceptance_decision") or {}
    graph_decision = row.get("graph_guard_decision") or {}

    acceptance_reason = str(acceptance_decision.get("reason", "") or "").lower()

    if row.get("repair_failed"):
        return "llm_call_failed"

    if not candidate_exists(row):
        return "no_candidate"

    if "empty, too short, missing final answer" in acceptance_reason:
        return "format_or_missing_final_answer"

    if "meta-discussion" in acceptance_reason:
        return "llm_rambling_or_meta_discussion"

    if "final answer did not change" in acceptance_reason:
        return "no_answer_change"

    if graph_decision.get("accept") is False:
        return "graph_guard_rejection"

    if "does not satisfy high-risk" in acceptance_reason:
        return "strict_acceptance_rejection"

    if "does not sufficiently improve reliability" in acceptance_reason:
        return "meta_gate_no_improvement"

    if "arithmetic" in acceptance_reason:
        return "arithmetic_or_checker_rejection"

    return "other_rejection"


def compact_case(row: Dict[str, Any]) -> Dict[str, Any]:
    eval_data = get_eval(row)

    initial_graph = row.get("initial_semantic_graph") or {}
    candidate_graph = row.get("candidate_semantic_graph") or {}
    initial_meta = row.get("initial_meta_diagnosis") or {}
    candidate_meta = row.get("candidate_meta_diagnosis") or {}

    return {
        "id": row.get("id"),
        "problem": row.get("problem"),
        "gold_answer": row.get("gold_answer"),

        "initial_answer": get_initial_answer(row),
        "candidate_answer": get_candidate_answer(row),
        "final_answer": get_final_answer(row),

        "initial_correct": initial_is_correct(row),
        "candidate_correct": candidate_is_correct(row),
        "final_correct": final_is_correct(row),

        "was_repaired": repair_was_attempted(row),
        "repair_accepted": repair_was_accepted(row),

        "fixed_error": eval_data.get("fixed_error"),
        "broke_correct": eval_data.get("broke_correct"),
        "failed_to_fix": eval_data.get("failed_to_fix"),

        "repair_source": row.get("repair_source"),

        "initial_meta_error_type": initial_meta.get("error_type"),
        "candidate_meta_error_type": candidate_meta.get("error_type"),
        "initial_meta_score": initial_meta.get("global_consistency_score"),
        "candidate_meta_score": candidate_meta.get("global_consistency_score"),

        "initial_graph_error_type": initial_graph.get("error_type"),
        "candidate_graph_error_type": candidate_graph.get("error_type"),
        "initial_graph_score": initial_graph.get("semantic_graph_score"),
        "candidate_graph_score": candidate_graph.get("semantic_graph_score"),

        "initial_semantic_pattern_issues": initial_graph.get("semantic_pattern_issues", []),
        "candidate_semantic_pattern_issues": candidate_graph.get("semantic_pattern_issues", []),

        "acceptance_reason": get_acceptance_reason(row),
        "graph_guard_reason": get_graph_reason(row),
        "rejection_class": classify_rejection(row),

        "deterministic_hint": row.get("deterministic_hint", ""),
        "repair_candidate": row.get("repair_candidate", ""),
        "initial_reasoning": row.get("initial_reasoning", ""),
        "final_reasoning": row.get("final_reasoning", ""),
    }


def main() -> None:
    ensure_output_dir()

    rows = load_jsonl(RAW_RESULTS_PATH)
    final_report = load_json(FINAL_REPORT_PATH)

    fixed_cases: List[Dict[str, Any]] = []
    broken_cases: List[Dict[str, Any]] = []
    remaining_errors: List[Dict[str, Any]] = []

    correct_but_rejected: List[Dict[str, Any]] = []
    initially_wrong_correct_but_rejected: List[Dict[str, Any]] = []
    initially_correct_correct_but_rejected: List[Dict[str, Any]] = []

    initially_wrong_graph_guard_rejected: List[Dict[str, Any]] = []
    initially_wrong_format_failed: List[Dict[str, Any]] = []
    initially_wrong_candidate_wrong: List[Dict[str, Any]] = []
    initially_wrong_no_candidate: List[Dict[str, Any]] = []

    rejected_by_reason: List[Dict[str, Any]] = []
    llm_format_failures: List[Dict[str, Any]] = []

    repair_source_counter = Counter()
    initial_graph_error_counter = Counter()
    candidate_graph_error_counter = Counter()
    initial_meta_error_counter = Counter()
    candidate_meta_error_counter = Counter()
    rejection_counter = Counter()

    accepted_count = 0
    rejected_count = 0

    accepted_noop_repairs = 0
    accepted_answer_changing_repairs = 0

    candidate_correct_count = 0
    candidate_correct_but_rejected_count = 0

    initially_wrong_count = 0
    initially_correct_count = 0

    initially_wrong_candidate_correct_count = 0
    initially_wrong_candidate_wrong_count = 0
    initially_wrong_candidate_missing_count = 0

    initially_correct_candidate_correct_count = 0
    initially_correct_candidate_wrong_count = 0

    for row in rows:
        compact = compact_case(row)

        init_correct = initial_is_correct(row)
        fin_correct = final_is_correct(row)
        cand_exists = candidate_exists(row)
        cand_correct = candidate_is_correct(row)
        cand_wrong = candidate_is_wrong(row)

        initial_answer = get_initial_answer(row)
        candidate_answer = get_candidate_answer(row)

        if init_correct:
            initially_correct_count += 1
        else:
            initially_wrong_count += 1

        repair_source = row.get("repair_source") or "none"
        repair_source_counter[repair_source] += 1

        initial_graph = row.get("initial_semantic_graph") or {}
        candidate_graph = row.get("candidate_semantic_graph") or {}
        initial_meta = row.get("initial_meta_diagnosis") or {}
        candidate_meta = row.get("candidate_meta_diagnosis") or {}

        initial_graph_error_counter[initial_graph.get("error_type", "none")] += 1
        candidate_graph_error_counter[candidate_graph.get("error_type", "none")] += 1
        initial_meta_error_counter[initial_meta.get("error_type", "none")] += 1
        candidate_meta_error_counter[candidate_meta.get("error_type", "none")] += 1

        eval_data = get_eval(row)

        if eval_data.get("fixed_error"):
            fixed_cases.append(compact)

        if eval_data.get("broke_correct"):
            broken_cases.append(compact)

        if not fin_correct:
            remaining_errors.append(compact)

        if repair_was_attempted(row):
            if repair_was_accepted(row):
                accepted_count += 1

                if initial_answer and candidate_answer and not answers_differ(initial_answer, candidate_answer):
                    accepted_noop_repairs += 1
                else:
                    accepted_answer_changing_repairs += 1

            else:
                rejected_count += 1
                rejection_class = classify_rejection(row)
                rejection_counter[rejection_class] += 1
                rejected_by_reason.append(compact)

                if rejection_class in {
                    "format_or_missing_final_answer",
                    "llm_rambling_or_meta_discussion",
                }:
                    llm_format_failures.append(compact)

        if cand_correct:
            candidate_correct_count += 1

            if not repair_was_accepted(row):
                candidate_correct_but_rejected_count += 1
                correct_but_rejected.append(compact)

                if init_correct:
                    initially_correct_correct_but_rejected.append(compact)
                else:
                    initially_wrong_correct_but_rejected.append(compact)

        if not init_correct:
            if not cand_exists:
                initially_wrong_candidate_missing_count += 1
                initially_wrong_no_candidate.append(compact)
            elif cand_correct:
                initially_wrong_candidate_correct_count += 1
            elif cand_wrong:
                initially_wrong_candidate_wrong_count += 1
                initially_wrong_candidate_wrong.append(compact)
            else:
                initially_wrong_candidate_missing_count += 1
                initially_wrong_no_candidate.append(compact)

            rejection_class = classify_rejection(row)

            if not repair_was_accepted(row):
                if rejection_class == "graph_guard_rejection":
                    initially_wrong_graph_guard_rejected.append(compact)

                if rejection_class in {
                    "format_or_missing_final_answer",
                    "llm_rambling_or_meta_discussion",
                }:
                    initially_wrong_format_failed.append(compact)

        if init_correct:
            if cand_correct:
                initially_correct_candidate_correct_count += 1
            elif cand_wrong:
                initially_correct_candidate_wrong_count += 1

    analysis = {
        "total": len(rows),

        "accepted_repairs": accepted_count,
        "rejected_repairs": rejected_count,

        "fixed_cases": len(fixed_cases),
        "broken_cases": len(broken_cases),
        "remaining_errors": len(remaining_errors),

        "candidate_correct_count": candidate_correct_count,
        "candidate_correct_but_rejected_count": candidate_correct_but_rejected_count,

        "initially_wrong_count": initially_wrong_count,
        "initially_correct_count": initially_correct_count,

        "initially_wrong_candidate_correct_count": initially_wrong_candidate_correct_count,
        "initially_wrong_candidate_correct_but_rejected": len(
            initially_wrong_correct_but_rejected
        ),
        "initially_wrong_candidate_wrong_count": initially_wrong_candidate_wrong_count,
        "initially_wrong_candidate_missing_or_unparseable": initially_wrong_candidate_missing_count,

        "initially_correct_candidate_correct_count": initially_correct_candidate_correct_count,
        "initially_correct_candidate_correct_but_rejected": len(
            initially_correct_correct_but_rejected
        ),
        "initially_correct_candidate_wrong_count": initially_correct_candidate_wrong_count,

        "initially_wrong_graph_guard_rejected": len(
            initially_wrong_graph_guard_rejected
        ),
        "initially_wrong_format_failed": len(initially_wrong_format_failed),
        "initially_wrong_candidate_wrong": len(initially_wrong_candidate_wrong),
        "initially_wrong_no_candidate": len(initially_wrong_no_candidate),

        "accepted_noop_repairs": accepted_noop_repairs,
        "accepted_answer_changing_repairs": accepted_answer_changing_repairs,

        "llm_format_failures": len(llm_format_failures),
    }

    summary = {
        "input_files": {
            "raw_results_path": RAW_RESULTS_PATH,
            "final_report_path": FINAL_REPORT_PATH,
        },
        "final_report": final_report,
        "analysis": analysis,
        "counters": {
            "repair_source_counter": dict(repair_source_counter),
            "initial_graph_error_counter": dict(initial_graph_error_counter),
            "candidate_graph_error_counter": dict(candidate_graph_error_counter),
            "initial_meta_error_counter": dict(initial_meta_error_counter),
            "candidate_meta_error_counter": dict(candidate_meta_error_counter),
            "rejection_counter": dict(rejection_counter),
        },
        "output_files": {
            "summary": SUMMARY_PATH,
            "fixed_cases": FIXED_CASES_PATH,
            "broken_cases": BROKEN_CASES_PATH,
            "remaining_errors": REMAINING_ERRORS_PATH,
            "correct_but_rejected": CORRECT_BUT_REJECTED_PATH,
            "initially_wrong_candidate_correct_but_rejected": INITIALLY_WRONG_CORRECT_BUT_REJECTED_PATH,
            "initially_correct_candidate_correct_but_rejected": INITIALLY_CORRECT_CORRECT_BUT_REJECTED_PATH,
            "initially_wrong_graph_guard_rejected": INITIALLY_WRONG_GRAPH_GUARD_REJECTED_PATH,
            "initially_wrong_format_failed": INITIALLY_WRONG_FORMAT_FAILED_PATH,
            "initially_wrong_candidate_wrong": INITIALLY_WRONG_CANDIDATE_WRONG_PATH,
            "initially_wrong_no_candidate": INITIALLY_WRONG_NO_CANDIDATE_PATH,
            "rejected_by_reason": REJECTED_BY_REASON_PATH,
            "llm_format_failures": LLM_FORMAT_FAILURES_PATH,
        },
    }

    save_json(SUMMARY_PATH, summary)

    save_jsonl(FIXED_CASES_PATH, fixed_cases)
    save_jsonl(BROKEN_CASES_PATH, broken_cases)
    save_jsonl(REMAINING_ERRORS_PATH, remaining_errors)

    save_jsonl(CORRECT_BUT_REJECTED_PATH, correct_but_rejected)
    save_jsonl(
        INITIALLY_WRONG_CORRECT_BUT_REJECTED_PATH,
        initially_wrong_correct_but_rejected,
    )
    save_jsonl(
        INITIALLY_CORRECT_CORRECT_BUT_REJECTED_PATH,
        initially_correct_correct_but_rejected,
    )

    save_jsonl(
        INITIALLY_WRONG_GRAPH_GUARD_REJECTED_PATH,
        initially_wrong_graph_guard_rejected,
    )
    save_jsonl(
        INITIALLY_WRONG_FORMAT_FAILED_PATH,
        initially_wrong_format_failed,
    )
    save_jsonl(
        INITIALLY_WRONG_CANDIDATE_WRONG_PATH,
        initially_wrong_candidate_wrong,
    )
    save_jsonl(
        INITIALLY_WRONG_NO_CANDIDATE_PATH,
        initially_wrong_no_candidate,
    )

    save_jsonl(REJECTED_BY_REASON_PATH, rejected_by_reason)
    save_jsonl(LLM_FORMAT_FAILURES_PATH, llm_format_failures)

    print("Analysis finished.")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))

    print("\nTop rejection reasons:")
    for key, value in rejection_counter.most_common():
        print(f"  {key}: {value}")

    print("\nImportant output files:")
    print(f"  Summary: {SUMMARY_PATH}")
    print(f"  Fixed cases: {FIXED_CASES_PATH}")
    print(f"  Broken cases: {BROKEN_CASES_PATH}")
    print(f"  Remaining errors: {REMAINING_ERRORS_PATH}")
    print(
        "  Initially wrong but candidate correct rejected: "
        f"{INITIALLY_WRONG_CORRECT_BUT_REJECTED_PATH}"
    )
    print(
        "  Initially wrong graph guard rejected: "
        f"{INITIALLY_WRONG_GRAPH_GUARD_REJECTED_PATH}"
    )
    print(
        "  Initially wrong format failed: "
        f"{INITIALLY_WRONG_FORMAT_FAILED_PATH}"
    )


if __name__ == "__main__":
    main()
