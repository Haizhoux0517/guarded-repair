import re
from typing import Dict, Any, List

from src.reasoning_parser import normalize_math_symbols, extract_numbers


def extract_problem_constraints(problem: str) -> Dict[str, Any]:
    """
    Extract lightweight numerical constraints from the problem.

    This module is dataset-general:
    it extracts numbers and their local textual context.
    """

    problem = normalize_math_symbols(problem or "")
    numbers = extract_numbers(problem)
    constraints = []
    for number in numbers:
        context = extract_number_context(problem, number)
        constraints.append(
            {
                "number": number,
                "context": context,
                "is_used": False,
            }
        )

    return {
        "problem_numbers": numbers,
        "constraints": constraints,
    }


def extract_reasoning_used_numbers(reasoning: str) -> List[str]:
    """
    Extract all numerical values used in reasoning.
    """
    reasoning = normalize_math_symbols(reasoning or "")
    return extract_numbers(reasoning)


def check_constraint_coverage(problem: str, reasoning: str) -> Dict[str, Any]:
    """
    Check whether numerical constraints in the problem are used in reasoning.

    Example:
    Problem: "Imma has 3 cats. She feeds her cats twice a day with 60 grams..."
    Reasoning uses only 2, 60, 720.
    Missing constraint: 3
    """

    problem_info = extract_problem_constraints(problem)
    reasoning_numbers = extract_reasoning_used_numbers(reasoning)

    reasoning_number_set = set(reasoning_numbers)

    constraints = []

    for constraint in problem_info["constraints"]:
        number = constraint["number"]
        is_used = number in reasoning_number_set

        constraints.append(
            {
                "number": number,
                "context": constraint["context"],
                "is_used": is_used,
            }
        )

    total_constraints = len(constraints)
    used_constraints = sum(1 for c in constraints if c["is_used"])
    missing_constraints = [c for c in constraints if not c["is_used"]]

    coverage_score = (
        used_constraints / total_constraints
        if total_constraints > 0
        else 1.0
    )

    return {
        "total_constraints": total_constraints,
        "used_constraints": used_constraints,
        "missing_constraints": missing_constraints,
        "constraint_coverage_score": coverage_score,
        "reasoning_numbers": reasoning_numbers,
    }


def extract_number_context(text: str, number: str, window: int = 8) -> str:
    """
    Extract a small context window around a number.
    This helps explain what condition may have been ignored.
    """

    tokens = re.findall(r"\w+|[^\w\s]", text)

    for i, token in enumerate(tokens):
        if token == number:
            start = max(0, i - window)
            end = min(len(tokens), i + window + 1)
            return " ".join(tokens[start:end])

    return ""