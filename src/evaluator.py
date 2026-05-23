from fractions import Fraction
import re
from src.reasoning_parser import extract_final_answer

def normalize_answer(ans):
    """
    Normalize answer for comparison.

    Supports:
      - comma numbers
      - decimals
      - fractions
      - ratios
      - time strings
      - yes/no answers
    """
    if ans is None:
        return None

    ans = str(ans).strip()
    ans = ans.strip("`")
    ans = ans.strip(".。;,，")
    ans = ans.replace("，", ",")
    # Remove unit parentheses, e.g., "9 (apples)" -> "9".
    ans = re.sub(r"\([^)]*\)", "", ans).strip()
    # Remove commas in numbers.
    ans = ans.replace(",", "")
    # Normalize spaces around fraction / colon.
    ans = re.sub(r"\s*/\s*", "/", ans)
    ans = re.sub(r"\s*:\s*", ":", ans)
    if ans.lower() in {"yes", "no"}:
        return ans.lower()
    # Normalize 12.0 -> 12.
    try:
        value = float(ans)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except Exception:
        return ans

def is_correct(predicted, gold):
    pred = normalize_answer(predicted)
    gold = normalize_answer(gold)
    if pred is None or pred == "":
        return False
    if gold is None or gold == "":
        return False
    # Exact normalized match first.
    if pred == gold:
        return True
    # Numeric equivalence: 12 and 12.0.
    try:
        return float(pred) == float(gold)
    except Exception:
        pass
    # Fraction equivalence: 2/3 and 0.6666 are not always safe due to rounding,
    # but 2/3 and 4/6 should match.
    pred_frac = to_fraction(pred)
    gold_frac = to_fraction(gold)
    if pred_frac is not None and gold_frac is not None:
        return pred_frac == gold_frac
    # Ratio equivalence: 2:3 and 4:6.
    pred_ratio = to_ratio(pred)
    gold_ratio = to_ratio(gold)
    if pred_ratio is not None and gold_ratio is not None:
        return pred_ratio == gold_ratio
    # Ratio vs fraction equivalence: 2:3 and 2/3.
    if pred_ratio is not None and gold_frac is not None:
        return ratio_to_fraction(pred_ratio) == gold_frac
    if gold_ratio is not None and pred_frac is not None:
        return ratio_to_fraction(gold_ratio) == pred_frac
    return False

def to_fraction(ans):
    ans = normalize_answer(ans)
    if ans is None:
        return None
    try:
        if re.fullmatch(r"[-+]?\d+/\d+", ans):
            return Fraction(ans)

        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", ans):
            return Fraction(ans)
    except Exception:
        return None
    return None

def to_ratio(ans):
    ans = normalize_answer(ans)
    if ans is None:
        return None
    if not re.fullmatch(r"[-+]?\d+:\d+", ans):
        return None
    try:
        left, right = ans.split(":")
        left = int(left)
        right = int(right)
        if right == 0:
            return None
        frac = Fraction(left, right)
        return (frac.numerator, frac.denominator)
    except Exception:
        return None

def ratio_to_fraction(ratio):
    if ratio is None:
        return None
    numerator, denominator = ratio
    return Fraction(numerator, denominator)

def evaluate_result(result):
    initial_answer = extract_final_answer(result["initial_reasoning"])
    final_answer = extract_final_answer(result.get("final_reasoning", ""))
    initial_correct = is_correct(initial_answer, result["gold_answer"])
    final_correct = is_correct(final_answer, result["gold_answer"])
    was_repaired = result.get("was_repaired", False)
    repair_accepted = result.get("repair_accepted", False)
    return {
        "initial_answer": initial_answer,
        "final_answer": final_answer,
        "gold_answer": result["gold_answer"],
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "was_repaired": was_repaired,
        "repair_accepted": repair_accepted,
        "fixed_error": (not initial_correct) and final_correct,
        "broke_correct": initial_correct and (not final_correct),
        "failed_to_fix": (not initial_correct) and (not final_correct),
    }

def summarize(all_results):
    total = len(all_results)
    initial_correct = sum(r["evaluation"]["initial_correct"] for r in all_results)
    final_correct = sum(r["evaluation"]["final_correct"] for r in all_results)
    repaired_cases = [
        r for r in all_results
        if r["evaluation"]["was_repaired"]
    ]
    accepted_cases = [
        r for r in all_results
        if r["evaluation"]["repair_accepted"]
    ]
    num_repaired = len(repaired_cases)
    num_accepted = len(accepted_cases)
    fixed_errors = sum(r["evaluation"]["fixed_error"] for r in all_results)
    broke_correct = sum(r["evaluation"]["broke_correct"] for r in all_results)
    failed_to_fix = sum(r["evaluation"]["failed_to_fix"] for r in all_results)
    initial_wrong = total - initial_correct
    return {
        "total": total,
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "initial_accuracy": initial_correct / total if total else 0,
        "final_accuracy": final_correct / total if total else 0,
        "absolute_improvement": (final_correct - initial_correct) / total if total else 0,
        "initial_wrong": initial_wrong,
        "num_repair_candidates": num_repaired,
        "num_repairs_accepted": num_accepted,
        "fixed_errors": fixed_errors,
        "broke_correct": broke_correct,
        "failed_to_fix": failed_to_fix,
        "repair_acceptance_rate": num_accepted / num_repaired if num_repaired else 0,
        "accepted_repair_precision": fixed_errors / num_accepted if num_accepted else 0,
        "error_repair_rate": fixed_errors / initial_wrong if initial_wrong else 0,
        "harm_rate": broke_correct / total if total else 0,
    }

def build_error_set(all_results):
    return [
        r for r in all_results
        if r["evaluation"]["initial_correct"] is False
    ]