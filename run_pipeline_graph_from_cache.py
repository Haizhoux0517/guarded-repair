import json
import os
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from config import REPAIR_MODEL
from src.dataset_loader import load_dataset
from src.deepseek_client import call_deepseek
from src.prompts import REPAIR_PROMPT
from src.symbolic_checker import check_reasoning
from src.constraint_checker import check_constraints
from src.meta_diagnoser import diagnose_reasoning
from src.acceptance_gate import should_accept_repair
from src.evaluator import evaluate_result, summarize_results
from src.reasoning_parser import extract_final_answer
from src.semantic_graph_checker import check_semantic_graph

from deterministic_semantic_repair import deterministic_semantic_repair


INITIAL_CACHE_PATH = "outputs/cache/initial_reasoning_cache.jsonl"

OUTPUT_DIR = "outputs/from_cache_guarded"
RAW_RESULTS_PATH = os.path.join(OUTPUT_DIR, "raw_results.jsonl")
FINAL_REPORT_PATH = os.path.join(OUTPUT_DIR, "final_report.json")
ERROR_SET_PATH = os.path.join(OUTPUT_DIR, "error_set.jsonl")

ENABLE_GRAPH_GUARD = True
ENABLE_DETERMINISTIC_REPAIR = True
ENABLE_LLM_REPAIR = True

GRAPH_TRIGGER_THRESHOLD = 0.80
GRAPH_ACCEPT_MIN_SCORE = 0.75
GRAPH_MIN_SCORE_DROP = 0.05

MIN_REPAIR_LENGTH = 40


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_initial_cache(path: str) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Initial cache not found: {path}\n"
            "Please run: python generate_initial_cache.py"
        )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            sample_id = str(item.get("id"))
            cache[sample_id] = item

    return cache


def safe_final_answer(reasoning: str) -> str:
    try:
        ans = extract_final_answer(reasoning or "")
        if ans is None:
            return ""
        return str(ans).strip()
    except Exception:
        return ""


def normalize_answer_for_compare(ans: str) -> str:
    ans = str(ans or "").strip()

    try:
        value = float(ans)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except Exception:
        return ans


def answers_differ(a: str, b: str) -> bool:
    return normalize_answer_for_compare(a) != normalize_answer_for_compare(b)


def run_diagnostics(problem: str, reasoning: str) -> Dict[str, Any]:
    if not reasoning or not reasoning.strip():
        return {
            "checker_results": None,
            "constraint_results": None,
            "meta_diagnosis": {
                "is_consistent": False,
                "global_consistency_score": 0.0,
                "average_step_score": 0.0,
                "symbolic_coverage": 0.0,
                "constraint_coverage_score": 0.0,
                "missing_constraints": [],
                "error_step": None,
                "error_type": "generation_failure",
                "step_scores": [],
                "explanation": "Reasoning is empty.",
            },
            "semantic_graph": check_semantic_graph(problem, reasoning),
        }

    checker_results = check_reasoning(reasoning)
    constraint_results = check_constraints(problem, reasoning)
    meta_diagnosis = diagnose_reasoning(
        checker_results=checker_results,
        constraint_results=constraint_results,
    )
    semantic_graph = check_semantic_graph(problem, reasoning)

    return {
        "checker_results": checker_results,
        "constraint_results": constraint_results,
        "meta_diagnosis": meta_diagnosis,
        "semantic_graph": semantic_graph,
    }


def should_attempt_repair(
    initial_meta: Dict[str, Any],
    initial_graph: Dict[str, Any],
    initial_reasoning: str,
) -> bool:
    if not initial_reasoning or not initial_reasoning.strip():
        return True

    if not initial_meta.get("is_consistent", False):
        return True

    if float(initial_meta.get("global_consistency_score", 0.0)) < 0.75:
        return True

    if initial_graph.get("needs_repair", False):
        return True

    if float(initial_graph.get("semantic_graph_score", 1.0)) < GRAPH_TRIGGER_THRESHOLD:
        return True

    return False


def is_valid_repair_text(candidate: str) -> bool:
    if not candidate or not candidate.strip():
        return False

    candidate = candidate.strip()

    if len(candidate) < MIN_REPAIR_LENGTH:
        return False

    if "Final Answer" not in candidate:
        return False

    if not safe_final_answer(candidate):
        return False

    return True


def call_llm_repair(
    problem: str,
    initial_reasoning: str,
    meta_diagnosis: Dict[str, Any],
    semantic_graph: Dict[str, Any],
) -> str:
    diagnosis_payload = {
        "meta_diagnosis": meta_diagnosis,
        "semantic_graph": semantic_graph,
        "instruction": (
            "Repair the reasoning only if the previous solution misinterprets "
            "the problem. Recompute from the original problem. Pay attention to "
            "per-item/per-day/per-person meanings, comparative phrases, remaining/left, "
            "equally, additional, total, and answer format."
        ),
    }

    prompt = REPAIR_PROMPT.format(
        problem=problem,
        reasoning=initial_reasoning,
        diagnosis=json.dumps(diagnosis_payload, ensure_ascii=False, indent=2),
    )

    return call_deepseek(
        model=REPAIR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
        max_retries=3,
    )


def run_deterministic_repair(
    problem: str,
    initial_reasoning: str,
    meta_diagnosis: Dict[str, Any],
    semantic_graph: Dict[str, Any],
) -> str:
    if not ENABLE_DETERMINISTIC_REPAIR:
        return ""

    try:
        candidate = deterministic_semantic_repair(
            problem=problem,
            reasoning=initial_reasoning,
            meta_diagnosis=meta_diagnosis,
            semantic_graph=semantic_graph,
        )
        return candidate or ""

    except TypeError:
        try:
            candidate = deterministic_semantic_repair(
                problem=problem,
                initial_reasoning=initial_reasoning,
                graph_result=semantic_graph,
            )
            return candidate or ""
        except TypeError:
            try:
                candidate = deterministic_semantic_repair(problem, initial_reasoning)
                return candidate or ""
            except Exception:
                return ""
        except Exception:
            return ""

    except Exception:
        return ""


def graph_guard_accept(
    initial_graph: Dict[str, Any],
    candidate_graph: Dict[str, Any],
) -> Dict[str, Any]:
    initial_score = float(initial_graph.get("semantic_graph_score", 0.0))
    candidate_score = float(candidate_graph.get("semantic_graph_score", 0.0))

    if candidate_graph.get("needs_repair", False):
        return {
            "accept": False,
            "reason": "Rejected because candidate still triggers semantic graph repair.",
            "initial_graph_score": initial_score,
            "candidate_graph_score": candidate_score,
        }

    if candidate_score < GRAPH_ACCEPT_MIN_SCORE:
        return {
            "accept": False,
            "reason": "Rejected because candidate semantic graph score is below threshold.",
            "initial_graph_score": initial_score,
            "candidate_graph_score": candidate_score,
        }

    if candidate_score + GRAPH_MIN_SCORE_DROP < initial_score:
        return {
            "accept": False,
            "reason": "Rejected because candidate semantic graph score drops too much.",
            "initial_graph_score": initial_score,
            "candidate_graph_score": candidate_score,
        }

    return {
        "accept": True,
        "reason": "Accepted by semantic graph guard.",
        "initial_graph_score": initial_score,
        "candidate_graph_score": candidate_score,
    }


def deterministic_accept_repair(
    initial_reasoning: str,
    candidate: str,
    initial_meta: Dict[str, Any],
    candidate_meta: Dict[str, Any],
    initial_graph: Dict[str, Any],
    candidate_graph: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deterministic repair uses a different acceptance policy from LLM repair.

    Reason:
    - LLM repair can hallucinate, so the original meta gate is useful.
    - Deterministic repair is rule-produced, so if it creates a valid answer-changing
      candidate and the semantic graph guard accepts it, we should not reject it only
      because the old meta score prefers the original arithmetic chain.
    """

    initial_answer = safe_final_answer(initial_reasoning)
    candidate_answer = safe_final_answer(candidate)

    initial_score = float(initial_meta.get("global_consistency_score", 0.0))
    candidate_score = float(candidate_meta.get("global_consistency_score", 0.0))

    if not candidate_answer:
        return {
            "accept": False,
            "reason": "Rejected deterministic repair because candidate has no parseable final answer.",
            "initial_score": initial_score,
            "candidate_score": candidate_score,
            "initial_answer": initial_answer,
            "candidate_answer": candidate_answer,
        }

    if not answers_differ(initial_answer, candidate_answer):
        return {
            "accept": False,
            "reason": "Rejected deterministic repair because final answer did not change.",
            "initial_score": initial_score,
            "candidate_score": candidate_score,
            "initial_answer": initial_answer,
            "candidate_answer": candidate_answer,
        }

    if candidate_graph.get("needs_repair", False):
        return {
            "accept": False,
            "reason": "Rejected deterministic repair because semantic graph still flags candidate.",
            "initial_score": initial_score,
            "candidate_score": candidate_score,
            "initial_answer": initial_answer,
            "candidate_answer": candidate_answer,
        }

    if float(candidate_graph.get("semantic_graph_score", 0.0)) < GRAPH_ACCEPT_MIN_SCORE:
        return {
            "accept": False,
            "reason": "Rejected deterministic repair because semantic graph score is too low.",
            "initial_score": initial_score,
            "candidate_score": candidate_score,
            "initial_answer": initial_answer,
            "candidate_answer": candidate_answer,
        }

    if not candidate_meta.get("is_consistent", False):
        return {
            "accept": False,
            "reason": "Rejected deterministic repair because candidate meta diagnosis is inconsistent.",
            "initial_score": initial_score,
            "candidate_score": candidate_score,
            "initial_answer": initial_answer,
            "candidate_answer": candidate_answer,
        }

    return {
        "accept": True,
        "reason": "Accepted deterministic repair because it produced a valid answer-changing candidate and passed semantic graph/meta consistency checks.",
        "initial_score": initial_score,
        "candidate_score": candidate_score,
        "initial_answer": initial_answer,
        "candidate_answer": candidate_answer,
    }


def try_candidate_repair(
    problem: str,
    initial_reasoning: str,
    initial_meta: Dict[str, Any],
    initial_graph: Dict[str, Any],
    candidate: str,
    source: str,
) -> Dict[str, Any]:
    if not is_valid_repair_text(candidate):
        return {
            "accepted": False,
            "source": source,
            "candidate": candidate or "",
            "candidate_diagnostics": None,
            "acceptance_decision": {
                "accept": False,
                "reason": "Rejected because candidate repair text is empty, too short, missing Final Answer, or has no parseable answer.",
            },
            "graph_guard_decision": None,
        }

    candidate_diagnostics = run_diagnostics(problem, candidate)

    candidate_meta = candidate_diagnostics["meta_diagnosis"]
    candidate_graph = candidate_diagnostics["semantic_graph"]

    if source == "deterministic":
        acceptance_decision = deterministic_accept_repair(
            initial_reasoning=initial_reasoning,
            candidate=candidate,
            initial_meta=initial_meta,
            candidate_meta=candidate_meta,
            initial_graph=initial_graph,
            candidate_graph=candidate_graph,
        )
    else:
        acceptance_decision = should_accept_repair(
            initial_diagnosis=initial_meta,
            candidate_diagnosis=candidate_meta,
        )

    if ENABLE_GRAPH_GUARD:
        graph_guard_decision = graph_guard_accept(
            initial_graph=initial_graph,
            candidate_graph=candidate_graph,
        )
    else:
        graph_guard_decision = {
            "accept": True,
            "reason": "Graph guard disabled.",
        }

    accepted = bool(
        acceptance_decision.get("accept", False)
        and graph_guard_decision.get("accept", False)
    )

    return {
        "accepted": accepted,
        "source": source,
        "candidate": candidate,
        "candidate_diagnostics": candidate_diagnostics,
        "acceptance_decision": acceptance_decision,
        "graph_guard_decision": graph_guard_decision,
    }


def apply_repair_result(
    repair_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "repair_candidate": repair_result.get("candidate", ""),
        "candidate_diagnostics": repair_result.get("candidate_diagnostics"),
        "acceptance_decision": repair_result.get("acceptance_decision"),
        "graph_guard_decision": repair_result.get("graph_guard_decision"),
        "repair_source": repair_result.get("source", ""),
        "repair_accepted": bool(repair_result.get("accepted", False)),
    }


def process_sample(sample: Dict[str, Any], cache_item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    sample_id = sample.get("id")
    problem = sample.get("problem", "")
    gold_answer = str(sample.get("answer", sample.get("gold_answer", ""))).strip()

    if cache_item is None:
        initial_reasoning = ""
        initial_diagnostics = run_diagnostics(problem, initial_reasoning)
        initial_diagnostics["meta_diagnosis"]["explanation"] = "Initial reasoning missing from cache."
    else:
        initial_reasoning = (
            cache_item.get("reasoning", "")
            or cache_item.get("initial_reasoning", "")
            or ""
        )
        initial_diagnostics = run_diagnostics(problem, initial_reasoning)

    initial_checker_results = initial_diagnostics["checker_results"]
    initial_constraint_results = initial_diagnostics["constraint_results"]
    initial_meta = initial_diagnostics["meta_diagnosis"]
    initial_graph = initial_diagnostics["semantic_graph"]

    repair_candidate = ""
    repair_failed = False
    repair_source = ""
    candidate_diagnostics = None
    acceptance_decision = None
    graph_guard_decision = None

    final_reasoning = initial_reasoning
    was_repaired = False
    repair_accepted = False

    if should_attempt_repair(initial_meta, initial_graph, initial_reasoning):
        was_repaired = True

        best_rejected_result: Optional[Dict[str, Any]] = None

        # 1. Deterministic repair first.
        deterministic_candidate = run_deterministic_repair(
            problem=problem,
            initial_reasoning=initial_reasoning,
            meta_diagnosis=initial_meta,
            semantic_graph=initial_graph,
        )

        deterministic_result = try_candidate_repair(
            problem=problem,
            initial_reasoning=initial_reasoning,
            initial_meta=initial_meta,
            initial_graph=initial_graph,
            candidate=deterministic_candidate,
            source="deterministic",
        )

        if deterministic_result["candidate"]:
            best_rejected_result = deterministic_result

        if deterministic_result["accepted"]:
            applied = apply_repair_result(deterministic_result)

            repair_candidate = applied["repair_candidate"]
            candidate_diagnostics = applied["candidate_diagnostics"]
            acceptance_decision = applied["acceptance_decision"]
            graph_guard_decision = applied["graph_guard_decision"]
            repair_source = applied["repair_source"]
            repair_accepted = applied["repair_accepted"]

            final_reasoning = repair_candidate

        else:
            # 2. LLM repair fallback.
            if ENABLE_LLM_REPAIR:
                try:
                    llm_candidate = call_llm_repair(
                        problem=problem,
                        initial_reasoning=initial_reasoning,
                        meta_diagnosis=initial_meta,
                        semantic_graph=initial_graph,
                    )

                    llm_result = try_candidate_repair(
                        problem=problem,
                        initial_reasoning=initial_reasoning,
                        initial_meta=initial_meta,
                        initial_graph=initial_graph,
                        candidate=llm_candidate,
                        source="llm",
                    )

                    if llm_result["candidate"]:
                        best_rejected_result = llm_result

                    if llm_result["accepted"]:
                        applied = apply_repair_result(llm_result)

                        repair_candidate = applied["repair_candidate"]
                        candidate_diagnostics = applied["candidate_diagnostics"]
                        acceptance_decision = applied["acceptance_decision"]
                        graph_guard_decision = applied["graph_guard_decision"]
                        repair_source = applied["repair_source"]
                        repair_accepted = applied["repair_accepted"]

                        final_reasoning = repair_candidate

                    else:
                        if best_rejected_result is not None:
                            applied = apply_repair_result(best_rejected_result)

                            repair_candidate = applied["repair_candidate"]
                            candidate_diagnostics = applied["candidate_diagnostics"]
                            acceptance_decision = applied["acceptance_decision"]
                            graph_guard_decision = applied["graph_guard_decision"]
                            repair_source = applied["repair_source"]

                except Exception as e:
                    repair_failed = True
                    print(f"[Repair failed] {e}")

                    # Important:
                    # If LLM fails, keep deterministic candidate information
                    # instead of losing it.
                    if best_rejected_result is not None:
                        applied = apply_repair_result(best_rejected_result)

                        repair_candidate = applied["repair_candidate"]
                        candidate_diagnostics = applied["candidate_diagnostics"]
                        acceptance_decision = applied["acceptance_decision"]
                        graph_guard_decision = applied["graph_guard_decision"]
                        repair_source = applied["repair_source"]

            else:
                if best_rejected_result is not None:
                    applied = apply_repair_result(best_rejected_result)

                    repair_candidate = applied["repair_candidate"]
                    candidate_diagnostics = applied["candidate_diagnostics"]
                    acceptance_decision = applied["acceptance_decision"]
                    graph_guard_decision = applied["graph_guard_decision"]
                    repair_source = applied["repair_source"]

    result = {
        "id": sample_id,
        "problem": problem,
        "gold_answer": gold_answer,

        "initial_reasoning": initial_reasoning,
        "initial_checker_results": initial_checker_results,
        "initial_constraint_results": initial_constraint_results,
        "initial_meta_diagnosis": initial_meta,
        "initial_semantic_graph": initial_graph,

        "repair_candidate": repair_candidate,
        "repair_failed": repair_failed,
        "repair_source": repair_source,

        "candidate_checker_results": (
            candidate_diagnostics["checker_results"] if candidate_diagnostics else None
        ),
        "candidate_constraint_results": (
            candidate_diagnostics["constraint_results"] if candidate_diagnostics else None
        ),
        "candidate_meta_diagnosis": (
            candidate_diagnostics["meta_diagnosis"] if candidate_diagnostics else None
        ),
        "candidate_semantic_graph": (
            candidate_diagnostics["semantic_graph"] if candidate_diagnostics else None
        ),

        "acceptance_decision": acceptance_decision,
        "graph_guard_decision": graph_guard_decision,

        "final_reasoning": final_reasoning,
        "was_repaired": was_repaired,
        "repair_accepted": repair_accepted,
    }

    result["evaluation"] = evaluate_result(result)
    return result


def main() -> None:
    ensure_output_dir()

    dataset = load_dataset()
    cache = load_initial_cache(INITIAL_CACHE_PATH)

    results: List[Dict[str, Any]] = []

    with open(RAW_RESULTS_PATH, "w", encoding="utf-8") as raw_f:
        for sample in tqdm(dataset, desc="Running guarded pipeline from cache"):
            sample_id = str(sample.get("id"))
            cache_item = cache.get(sample_id)

            result = process_sample(sample, cache_item)
            results.append(result)

            raw_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            raw_f.flush()

    report = summarize_results(results)
    report.update(
        {
            "cache_path": INITIAL_CACHE_PATH,
            "from_cache": True,
            "guarded_pipeline": True,
            "graph_guard_enabled": ENABLE_GRAPH_GUARD,
            "deterministic_repair_enabled": ENABLE_DETERMINISTIC_REPAIR,
            "llm_repair_enabled": ENABLE_LLM_REPAIR,
            "graph_trigger_threshold": GRAPH_TRIGGER_THRESHOLD,
            "graph_accept_min_score": GRAPH_ACCEPT_MIN_SCORE,
            "deterministic_accept_policy": "answer_change_plus_graph_guard_plus_meta_consistency",
        }
    )

    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    error_results = [
        r for r in results
        if not r.get("evaluation", {}).get("final_correct", False)
    ]

    with open(ERROR_SET_PATH, "w", encoding="utf-8") as f:
        for r in error_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Guarded pipeline from cache finished.")
    print(report)
    print(f"Saved raw results to: {RAW_RESULTS_PATH}")
    print(f"Saved final report to: {FINAL_REPORT_PATH}")
    print(f"Saved error set to: {ERROR_SET_PATH}")


if __name__ == "__main__":
    main()