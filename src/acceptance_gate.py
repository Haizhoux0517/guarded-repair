from typing import Dict, Any
from src.symbolic_checker import check_reasoning
from src.meta_diagnoser import compute_meta_diagnosis

def evaluate_candidate_reasoning(
    reasoning: str,
    step_threshold: float,
    global_threshold: float,
) -> Dict[str, Any]:
    checker_results = check_reasoning(reasoning)
    meta_diagnosis = compute_meta_diagnosis(
        checker_results=checker_results,
        step_threshold=step_threshold,
        global_threshold=global_threshold,
    )

    return {
        "checker_results": checker_results,
        "meta_diagnosis": meta_diagnosis,
    }

def should_accept_repair(
    initial_meta: Dict[str, Any],
    candidate_meta: Dict[str, Any],
    min_improvement: float = 0.08,
) -> Dict[str, Any]:
    """
    Repair Acceptance Gate.

    Accept repair only if:
    1. candidate has higher meta-consistency score
    2. improvement is large enough
    3. candidate is not explicitly inconsistent
    """
    initial_score = initial_meta.get("global_consistency_score", 0.0)
    candidate_score = candidate_meta.get("global_consistency_score", 0.0)
    improvement = candidate_score - initial_score
    candidate_consistent = candidate_meta.get("is_consistent", False)
    accept = (
        candidate_consistent
        and improvement >= min_improvement
    )

    return {
        "accept": accept,
        "initial_score": initial_score,
        "candidate_score": candidate_score,
        "score_improvement": improvement,
        "reason": (
            "Accepted because candidate repair improves meta-consistency."
            if accept
            else "Rejected because candidate repair does not sufficiently improve reliability."
        ),
    }