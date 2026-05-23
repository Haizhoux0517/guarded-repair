import re
import sympy as sp

from src.reasoning_parser import (
    parse_steps,
    extract_equations,
    extract_final_answer,
    extract_numbers as robust_extract_numbers,
    normalize_math_symbols,
)

def check_reasoning(reasoning: str):
    steps = parse_steps(reasoning)
    results = []
    for idx, step in enumerate(steps, start=1):
        equations = extract_equations(step)
        numbers = extract_numbers(step)
        step_result = {
            "step": idx,
            "text": step,
            "equations": equations,
            "numbers": numbers,
            "passed": True,
            "checkable": bool(equations),
            "weakly_checkable": (not equations and len(numbers) >= 2),
            "errors": [],
        }
        for equation in equations:
            ok, error_msg = check_equation(equation)
            if not ok:
                step_result["passed"] = False
                step_result["errors"].append(
                    {
                        "equation": equation,
                        "error": error_msg,
                    }
                )
        results.append(step_result)
    final_answer = extract_final_answer(reasoning)
    summary = {
        "num_steps": len(steps),
        "num_checkable_steps": sum(r["checkable"] for r in results),
        "num_weakly_checkable_steps": sum(r["weakly_checkable"] for r in results),
        "has_explicit_failure": any(r["passed"] is False for r in results),
        "final_answer": final_answer,
        "low_symbolic_coverage": False,
    }
    if len(steps) > 0:
        coverage = summary["num_checkable_steps"] / len(steps)
        summary["symbolic_coverage"] = coverage
        summary["low_symbolic_coverage"] = coverage < 0.3
    else:
        summary["symbolic_coverage"] = 0.0
        summary["low_symbolic_coverage"] = True
    return {
        "steps": results,
        "summary": summary,
    }

def check_equation(equation: str):
    """
    Check whether a numeric arithmetic equation is correct.

    This version sanitizes comma-formatted numbers and common formatting symbols
    before passing expressions to SymPy.

    Examples:
        564,237 + 495,718 = 1,059,955
        1410 + 6908 = 8318
        56 / 84 = 2/3
    """
    try:
        equation = sanitize_equation(equation)
        if "=" not in equation:
            return False, "Equation parse error: missing '='"
        left, right = equation.split("=", 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            return False, "Equation parse error: empty side"
        left_value = sp.sympify(left)
        right_value = sp.sympify(right)
        if sp.simplify(left_value - right_value) == 0:
            return True, None
        return False, f"Arithmetic mismatch: {left} != {right}"
    except Exception as e:
        return False, f"Equation parse error: {str(e)}"

def sanitize_equation(equation: str) -> str:
    """
    Normalize an equation before symbolic checking.

    Important:
    - remove thousands separators in numbers
    - normalize math symbols
    - remove currency symbols
    - keep fractions such as 2/3 intact
    """

    equation = normalize_math_symbols(equation or "")
    # Remove currency signs and common unit artifacts that may appear next to numbers.
    equation = equation.replace("$", "")
    # Remove commas used as thousands separators:
    # 1,059,955 -> 1059955
    equation = re.sub(r"(?<=\d),(?=\d{3}\b)", "", equation)
    # Normalize percent notation only when it appears as a standalone numeric percentage.
    # For equation checking, 40% should become 0.40 if such expression is extracted.
    equation = re.sub(
        r"(\d+(?:\.\d+)?)\s*%",
        lambda m: str(float(m.group(1)) / 100),
        equation,
    )
    # Remove trailing punctuation.
    equation = equation.strip().strip(".。;；,，")
    return equation

def extract_numbers(text: str):
    """
    Use the robust number extractor from reasoning_parser.

    This keeps comma-formatted numbers together:
        1,059,955 -> 1059955
    instead of splitting them into 1, 059, 955.
    """
    return robust_extract_numbers(text)

if __name__ == "__main__":
    demos = [
        "564,237 + 495,718 = 1,059,955",
        "1410 + 6908 = 8318",
        "56 / 84 = 2/3",
        "100 - 40 = 60",
        "0.6 * 40 = 24",
    ]
    for demo in demos:
        print(demo, "=>", check_equation(demo))