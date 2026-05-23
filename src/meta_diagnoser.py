from typing import Dict, Any


def compute_meta_diagnosis(
    checker_results: Dict[str, Any],
    constraint_results: Dict[str, Any] = None,
    arithmetic_weight: float = 0.35,
    support_weight: float = 0.25,
    coverage_weight: float = 0.20,
    constraint_weight: float = 0.20,
    step_threshold: float = 0.55,
    global_threshold: float = 0.65,
) -> Dict[str, Any]:
    """
    Meta-Consistency and Constraint Coverage Verification.

    This module checks:
    1. Arithmetic consistency
    2. Step-level support
    3. Symbolic coverage
    4. Problem constraint coverage

    The key upgrade:
    It can detect cases where the reasoning is internally consistent
    but ignores important problem constraints.
    """

    steps = checker_results.get("steps", [])
    summary = checker_results.get("summary", {})

    if constraint_results is None:
        constraint_results = {
            "constraint_coverage_score": 1.0,
            "missing_constraints": [],
        }

    constraint_score = constraint_results.get("constraint_coverage_score", 1.0)
    missing_constraints = constraint_results.get("missing_constraints", [])

    step_scores = []

    for step in steps:
        arithmetic_score = compute_arithmetic_score(step)
        support_score = compute_support_score(step)
        coverage_score = compute_coverage_score(step)

        consistency_score = (
            arithmetic_weight * arithmetic_score
            + support_weight * support_score
            + coverage_weight * coverage_score
            + constraint_weight * constraint_score
        )

        step_scores.append(
            {
                "step": step.get("step"),
                "arithmetic_score": arithmetic_score,
                "support_score": support_score,
                "coverage_score": coverage_score,
                "constraint_score": constraint_score,
                "consistency_score": consistency_score,
                "is_unreliable": consistency_score < step_threshold,
                "text": step.get("text", ""),
            }
        )

    if step_scores:
        min_step = min(step_scores, key=lambda x: x["consistency_score"])
        average_score = sum(s["consistency_score"] for s in step_scores) / len(step_scores)
    else:
        min_step = None
        average_score = 0.0

    symbolic_coverage = summary.get("symbolic_coverage", 0.0)
    has_explicit_failure = summary.get("has_explicit_failure", False)

    global_consistency_score = (
        0.55 * average_score
        + 0.25 * symbolic_coverage
        + 0.20 * constraint_score
    )

    has_missing_constraints = len(missing_constraints) > 0

    is_consistent = (
        global_consistency_score >= global_threshold
        and not has_explicit_failure
        and not has_critical_missing_constraint(missing_constraints)
    )

    error_step = None
    error_type = "none"

    if not is_consistent:
        if has_critical_missing_constraint(missing_constraints):
            error_step = None
            error_type = "missing_constraint"
        elif min_step is not None:
            error_step = min_step["step"]
            error_type = infer_error_type(min_step, summary)

    return {
        "is_consistent": is_consistent,
        "global_consistency_score": global_consistency_score,
        "average_step_score": average_score,
        "symbolic_coverage": symbolic_coverage,
        "constraint_coverage_score": constraint_score,
        "missing_constraints": missing_constraints,
        "error_step": error_step,
        "error_type": error_type,
        "step_scores": step_scores,
        "explanation": build_explanation(
            is_consistent=is_consistent,
            global_score=global_consistency_score,
            error_step=error_step,
            error_type=error_type,
            missing_constraints=missing_constraints,
        ),
    }


def compute_arithmetic_score(step: Dict[str, Any]) -> float:
    if step.get("passed") is False:
        return 0.0

    equations = step.get("equations", [])

    if equations:
        return 1.0

    return 0.7


def compute_support_score(step: Dict[str, Any]) -> float:
    equations = step.get("equations", [])
    numbers = step.get("numbers", [])

    if equations:
        return 1.0

    if len(numbers) >= 2:
        return 0.65

    if len(numbers) == 1:
        return 0.45

    return 0.25


def compute_coverage_score(step: Dict[str, Any]) -> float:
    if step.get("checkable"):
        return 1.0

    if step.get("weakly_checkable"):
        return 0.6

    return 0.25


def has_critical_missing_constraint(missing_constraints) -> bool:
    """
    A simple general rule:
    if any numerical constraint from the problem is unused,
    the reasoning may be missing necessary problem information.

    This is intentionally conservative and dataset-general.
    """

    return len(missing_constraints) > 0


def infer_error_type(
    min_step: Dict[str, Any],
    summary: Dict[str, Any],
) -> str:
    if min_step["arithmetic_score"] == 0.0:
        return "arithmetic_error"

    if summary.get("low_symbolic_coverage", False):
        return "low_symbolic_coverage"

    if min_step["support_score"] < 0.5:
        return "logical_jump"

    return "unknown"


def build_explanation(
    is_consistent: bool,
    global_score: float,
    error_step,
    error_type: str,
    missing_constraints,
) -> str:
    if is_consistent:
        return f"Reasoning is accepted with meta-consistency score {global_score:.3f}."

    if error_type == "missing_constraint":
        missing = [
            f"{item['number']} ({item['context']})"
            for item in missing_constraints
        ]

        return (
            f"Reasoning is rejected with meta-consistency score {global_score:.3f}. "
            f"Detected missing problem constraints: {missing}."
        )

    return (
        f"Reasoning is rejected with meta-consistency score {global_score:.3f}. "
        f"Most unreliable step: {error_step}. "
        f"Predicted error type: {error_type}."
    )