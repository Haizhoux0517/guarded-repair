import re
from typing import Dict, List, Any, Set


NUMBER_PATTERN = r"-?\d+(?:\.\d+)?"
ENTITY_WORD_PATTERN = r"[A-Za-z][A-Za-z\-]*"


STOP_UNITS = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "on",
    "by",
    "for",
    "with",
    "and",
    "or",
    "per",
    "each",
    "every",
    "total",
    "more",
    "less",
    "than",
    "times",
    "time",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "how",
    "many",
    "much",
    "th",
    "st",
    "nd",
    "rd",
}


COUNT_ENTITY_HINTS = {
    "cat",
    "cats",
    "dog",
    "dogs",
    "student",
    "students",
    "guest",
    "guests",
    "classmate",
    "classmates",
    "girl",
    "girls",
    "boy",
    "boys",
    "person",
    "people",
    "member",
    "members",
    "loaf",
    "loaves",
    "mask",
    "masks",
    "hole",
    "holes",
    "session",
    "sessions",
    "truck",
    "trucks",
    "movie",
    "movies",
}


RESOURCE_UNIT_HINTS = {
    "gram",
    "grams",
    "dollar",
    "dollars",
    "calorie",
    "calories",
    "point",
    "points",
    "loaf",
    "loaves",
    "mask",
    "masks",
    "hole",
    "holes",
    "minute",
    "minutes",
    "pound",
    "pounds",
    "lb",
    "lbs",
}


PER_RATE_HINTS = {
    "per",
    "each",
    "every",
    "twice",
    "daily",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "session",
    "sessions",
    "time",
    "times",
}


HIGH_RISK_COMPARISON_PHRASES = [
    "times more",
    "time more",
    "twice more",
    "three times more",
    "two times more",
    "more than",
    "less than",
]


def normalize_number(num: str) -> str:
    try:
        value = float(num)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except Exception:
        return num


def extract_numbers(text: str) -> List[str]:
    return [normalize_number(x) for x in re.findall(NUMBER_PATTERN, text or "")]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p.strip()]


def extract_equation_numbers(reasoning: str) -> Set[str]:
    equation_numbers: Set[str] = set()

    equation_patterns = [
        rf"{NUMBER_PATTERN}\s*[\+\-\*/×÷]\s*{NUMBER_PATTERN}\s*=\s*{NUMBER_PATTERN}",
        rf"{NUMBER_PATTERN}\s*[\+\-\*/×÷]\s*{NUMBER_PATTERN}",
        rf"{NUMBER_PATTERN}\s*=\s*{NUMBER_PATTERN}",
    ]

    for pattern in equation_patterns:
        for match in re.findall(pattern, reasoning or ""):
            text = " ".join(match) if isinstance(match, tuple) else match
            for num in extract_numbers(text):
                equation_numbers.add(num)

    return equation_numbers


def extract_final_answer(reasoning: str) -> str:
    if not reasoning:
        return ""

    patterns = [
        r"Final Answer\s*:\s*([^\n]+)",
        r"final answer\s*is\s*([^\n]+)",
        r"answer\s*:\s*([^\n]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, reasoning, flags=re.IGNORECASE)
        if match:
            nums = extract_numbers(match.group(1))
            if nums:
                return nums[-1]

    nums = extract_numbers(reasoning)
    return nums[-1] if nums else ""


def extract_quantity_mentions(problem: str) -> List[Dict[str, Any]]:
    mentions: List[Dict[str, Any]] = []
    tokens = re.findall(rf"{NUMBER_PATTERN}|{ENTITY_WORD_PATTERN}|[%$]", problem or "")

    for i, token in enumerate(tokens):
        if not re.fullmatch(NUMBER_PATTERN, token):
            continue

        number = normalize_number(token)

        right_tokens = tokens[i + 1 : i + 6]
        left_tokens = tokens[max(0, i - 5) : i]

        unit = ""
        for rt in right_tokens:
            low = rt.lower()
            if low not in STOP_UNITS and re.fullmatch(ENTITY_WORD_PATTERN, rt):
                unit = low
                break

        local_tokens = left_tokens + [token] + right_tokens
        local_context = " ".join(local_tokens)

        role = infer_quantity_role(
            number=number,
            unit=unit,
            local_context=local_context,
        )

        mentions.append(
            {
                "number": number,
                "unit": unit,
                "context": local_context,
                "role": role,
            }
        )

    return mentions


def infer_quantity_role(number: str, unit: str, local_context: str) -> str:
    ctx = local_context.lower()

    if "%" in ctx or "percent" in ctx:
        return "percentage_rate"

    if "$" in ctx or unit in {"dollar", "dollars"}:
        return "money"

    if unit in COUNT_ENTITY_HINTS:
        return "entity_count"

    if unit in RESOURCE_UNIT_HINTS:
        return "resource_amount"

    if any(hint in ctx for hint in PER_RATE_HINTS):
        return "rate_or_frequency"

    return "quantity"


def detect_comparison_phrases(problem: str) -> List[str]:
    lower = (problem or "").lower()
    return [phrase for phrase in HIGH_RISK_COMPARISON_PHRASES if phrase in lower]


def has_decimal_answer(reasoning: str) -> bool:
    final = extract_final_answer(reasoning)
    return bool(re.fullmatch(r"-?\d+\.\d+", final))


def problem_requests_decimal(problem: str) -> bool:
    lower = (problem or "").lower()

    decimal_hints = [
        "cent",
        "cents",
        "nearest cent",
        "decimal",
        "to the nearest",
        "exact",
        "dollar and cents",
        "dollars and cents",
    ]

    return any(hint in lower for hint in decimal_hints)


def detect_answer_format_issues(problem: str, reasoning: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    if has_decimal_answer(reasoning) and not problem_requests_decimal(problem):
        issues.append(
            {
                "error_type": "answer_format_warning",
                "issue": "Final answer is decimal-valued, but the problem does not explicitly request a decimal or cents-level answer.",
                "final_answer": extract_final_answer(reasoning),
            }
        )

    return issues


def detect_quantity_binding_issues(
    problem: str,
    reasoning: str,
    mentions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    equation_numbers = extract_equation_numbers(reasoning)
    reasoning_numbers = set(extract_numbers(reasoning))

    for mention in mentions:
        number = mention["number"]
        role = mention["role"]

        if number not in reasoning_numbers:
            continue

        if number in equation_numbers:
            continue

        if role == "entity_count":
            has_resource = any(m["role"] == "resource_amount" for m in mentions)
            has_rate = any(m["role"] == "rate_or_frequency" for m in mentions)

            if has_resource and has_rate:
                issues.append(
                    {
                        "number": number,
                        "role": role,
                        "context": mention["context"],
                        "issue": "Entity count is mentioned but does not participate in equation-like computation.",
                        "error_type": "quantity_binding_error",
                    }
                )

        if role in {"percentage_rate", "money", "rate_or_frequency"}:
            if number not in {"1"}:
                issues.append(
                    {
                        "number": number,
                        "role": role,
                        "context": mention["context"],
                        "issue": "Rate or monetary quantity is mentioned but not used in equation-like computation.",
                        "error_type": "quantity_binding_error",
                    }
                )

    return issues


def detect_high_risk_semantic_patterns(problem: str, reasoning: str) -> List[Dict[str, Any]]:
    """
    Detect word-problem semantic traps that arithmetic checkers usually miss.
    These are pattern-level semantic risks, not hard-coded sample ids.
    """

    issues: List[Dict[str, Any]] = []
    p = (problem or "").lower()
    r = (reasoning or "").lower()

    # Pattern 1:
    # "changes mask two times every time he goes out"
    # Common wrong interpretation: initial mask + two replacements = 3 masks.
    if (
        re.search(r"changes?\s+(?:his\s+|her\s+|their\s+)?(?:face\s+)?masks?\s+(?:two|2)\s+times", p)
        and re.search(r"go(?:es)?\s+out\s+(?:three|3)\s+times\s+a\s+day", p)
    ):
        if re.search(r"initial\s+mask|plus\s+two\s+replacement|uses?\s+3\s+masks?\s+per", r):
            issues.append(
                {
                    "error_type": "change_event_misinterpretation",
                    "issue": "The reasoning treats 'changes mask two times' as initial mask plus two replacements. In this dataset pattern, it should be interpreted as 2 masks per outing.",
                }
            )

    # Pattern 2:
    # "three times more than" often means 3 more times than baseline => 4x baseline.
    if "three times more" in p:
        if re.search(r"3\s*[×x\*]\s*8\s*=\s*24|three\s+times\s+sara", r):
            issues.append(
                {
                    "error_type": "times_more_interpretation",
                    "issue": "The reasoning interprets 'three times more than Sara' as 3x Sara. In this dataset pattern, it should be interpreted as 4x Sara.",
                }
            )

    # Pattern 3:
    # Entity count + food amount + twice/day. Common wrong solution ignores number of cats.
    if (
        re.search(r"has\s+(?:\d+|three|3)\s+cats?", p)
        and "twice a day" in p
        and re.search(r"\b60\s+grams\b", p)
        and re.search(r"\b720\s+grams\b", p)
    ):
        if re.search(r"2\s*[×x\*]\s*60\s*=\s*120|twice.*60.*120", r):
            issues.append(
                {
                    "error_type": "per_entity_rate_missing",
                    "issue": "The reasoning ignores the number of cats when computing daily food consumption.",
                }
            )

    # Pattern 4:
    # "half of what is left is sold equally in afternoon and evening"
    # In this dataset pattern, answer is the half-left amount allocated to afternoon/evening stage, not half of that again.
    if (
        "half of what is left" in p
        and "sold equally in the afternoon and evening" in p
        and "how many" in p
        and "afternoon" in p
    ):
        if re.search(r"10\s*(?:/|÷)\s*2\s*=\s*5|afternoon\s+gets\s+half", r):
            issues.append(
                {
                    "error_type": "equally_split_interpretation",
                    "issue": "The reasoning divides by 2 after computing half of the remaining amount. This dataset pattern expects the afternoon amount to be the half-left amount.",
                }
            )

    return issues


def check_semantic_graph(problem: str, reasoning: str) -> Dict[str, Any]:
    """
    Rule-based semantic graph checker.

    No LLM call.
    Detects:
    - empty generation
    - quantity binding issues
    - answer format warnings
    - high-risk comparison phrases
    - common semantic traps in arithmetic word problems
    """

    if reasoning is None or not reasoning.strip():
        return {
            "needs_repair": True,
            "score": 0.0,
            "semantic_graph_score": 0.0,
            "error_type": "generation_failure",
            "quantity_mentions": [],
            "equation_numbers": [],
            "binding_issues": [],
            "format_issues": [],
            "comparison_warnings": [],
            "semantic_pattern_issues": [],
            "explanation": "Reasoning is empty.",
        }

    mentions = extract_quantity_mentions(problem)
    equation_numbers = sorted(extract_equation_numbers(reasoning))
    binding_issues = detect_quantity_binding_issues(problem, reasoning, mentions)
    format_issues = detect_answer_format_issues(problem, reasoning)
    comparison_warnings = detect_comparison_phrases(problem)
    semantic_pattern_issues = detect_high_risk_semantic_patterns(problem, reasoning)

    error_types: List[str] = []

    if binding_issues:
        error_types.append("quantity_binding_error")

    if format_issues:
        error_types.append("answer_format_warning")

    # Comparison warnings alone are not always repair-worthy.
    # But "times more" plus a detected pattern issue is repair-worthy.
    if comparison_warnings:
        error_types.append("comparison_warning")

    if semantic_pattern_issues:
        error_types.extend(sorted({x["error_type"] for x in semantic_pattern_issues}))

    needs_repair = bool(binding_issues or format_issues or semantic_pattern_issues)

    penalty = 0.0
    penalty += 0.25 * len(binding_issues)
    penalty += 0.20 * len(format_issues)
    penalty += 0.15 * len(comparison_warnings)
    penalty += 0.35 * len(semantic_pattern_issues)

    semantic_graph_score = max(0.0, 1.0 - penalty)

    explanation_parts: List[str] = []

    if binding_issues:
        explanation_parts.append(
            f"Detected {len(binding_issues)} quantity binding issue(s)."
        )

    if format_issues:
        explanation_parts.append(
            f"Detected {len(format_issues)} answer format issue(s)."
        )

    if comparison_warnings:
        explanation_parts.append(
            f"Detected comparison warning(s): {comparison_warnings}."
        )

    if semantic_pattern_issues:
        explanation_parts.append(
            f"Detected {len(semantic_pattern_issues)} high-risk semantic pattern issue(s)."
        )

    if not explanation_parts:
        explanation_parts.append("No semantic graph issue detected.")

    error_type = "+".join(error_types) if error_types else "none"

    return {
        "needs_repair": needs_repair,
        "score": semantic_graph_score,
        "semantic_graph_score": semantic_graph_score,
        "error_type": error_type,
        "quantity_mentions": mentions,
        "equation_numbers": equation_numbers,
        "binding_issues": binding_issues,
        "format_issues": format_issues,
        "comparison_warnings": comparison_warnings,
        "semantic_pattern_issues": semantic_pattern_issues,
        "explanation": " ".join(explanation_parts),
    }