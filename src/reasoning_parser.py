import re
from typing import List

def normalize_math_symbols(text: str) -> str:
    """
    Normalize common math symbols so the symbolic checker can parse them.
    """
    if text is None:
        return ""
    return (
        str(text)
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("：", ":")
    )

def parse_steps(reasoning: str) -> List[str]:
    """
    Extract reasoning steps from model output.

    Expected format:
    Step 1: ...
    Step 2: ...
    Final Answer: ...
    """
    reasoning = reasoning or ""
    pattern = r"(Step\s+\d+\s*:\s.*?)(?=Step\s+\d+\s*:|Final Answer\s*:|$)"
    matches = re.findall(pattern, reasoning, flags=re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches if m.strip()]

def extract_final_answer(reasoning: str) -> str:
    """
    Extract the final answer while preserving common non-integer answer forms.

    Supports:
      - comma numbers: 1,059,955 -> 1059955
      - decimals: 12.5
      - fractions: 17/7
      - time / ratio style: 12:50, 2:3
      - yes/no answers
    """
    if not reasoning:
        return ""
    reasoning = normalize_math_symbols(reasoning)
    patterns = [
        r"Final Answer\s*:\s*([^\n]+)",
        r"final answer\s*is\s*([^\n]+)",
        r"answer\s*:\s*([^\n]+)",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, reasoning, flags=re.IGNORECASE))
    if matches:
        last_answer_text = matches[-1]
        answer = extract_answer_value(last_answer_text)
        if answer:
            return answer
    # Fallback: use the last answer-like value in the whole reasoning.
    values = extract_answer_values(reasoning)
    return values[-1] if values else ""

def extract_answer_value(text: str) -> str:
    """
    Extract one answer value from a short answer span.
    """
    values = extract_answer_values(text)
    return values[-1] if values else ""


def extract_answer_values(text: str) -> List[str]:
    """
    Extract answer-like values while preserving special formats.
    """
    text = normalize_math_symbols(text or "")
    text = clean_answer_span(text)
    values = []
    # yes/no answers.
    yes_no = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    for item in yes_no:
        values.append(item.lower())
    # Time or ratio-like answers, e.g., 12:50 or 2:3.
    colon_values = re.findall(r"(?<!\d)(\d{1,4}\s*:\s*\d{1,4})(?!\d)", text)
    for item in colon_values:
        values.append(item.replace(" ", ""))
    # Fractions, e.g., 17/7.
    fractions = re.findall(r"(?<!\d)([-+]?\d+\s*/\s*[-+]?\d+)(?!\d)", text)
    for item in fractions:
        values.append(item.replace(" ", ""))
    # Comma numbers, e.g., 1,059,955.
    comma_numbers = re.findall(
        r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?",
        text,
    )
    for item in comma_numbers:
        values.append(item.replace(",", ""))
    # Plain integers / decimals.
    # Avoid capturing pieces already captured inside comma numbers, fractions, or colon forms.
    protected = text
    for item in colon_values + fractions + comma_numbers:
        protected = protected.replace(item, " ")
    plain_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", protected)
    values.extend(plain_numbers)
    return [normalize_answer_token(v) for v in values if normalize_answer_token(v)]

def clean_answer_span(text: str) -> str:
    """
    Remove common trailing explanation text from a final-answer span.
    """
    text = str(text).strip()
    # Remove Markdown/code artifacts.
    text = text.strip("`").strip()
    # Remove common unit parentheses after the answer, e.g., "9 (apples)".
    text = re.sub(r"\([^)]*\)", "", text).strip()
    # Keep only before obvious sentence continuation when possible.
    # Do not split on ":" because it may be part of a time or ratio.
    text = re.split(
        r"\s+(?:because|since|which means|therefore|so the answer is)\s+",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    return text

def normalize_answer_token(token: str) -> str:
    """
    Normalize a single extracted answer token.
    """
    token = str(token).strip()
    token = token.strip(".。;,，")
    token = token.replace(" ", "")
    if not token:
        return ""
    if token.lower() in {"yes", "no"}:
        return token.lower()
    # Normalize comma numbers.
    token = token.replace(",", "")
    # Keep fractions and colon formats as strings.
    if re.fullmatch(r"[-+]?\d+/\d+", token):
        return token
    if re.fullmatch(r"\d{1,4}:\d{1,4}", token):
        return token
    # Normalize 12.0 -> 12.
    try:
        value = float(token)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return token

def extract_numbers(text: str) -> List[str]:
    """
    Extract integers and decimals for symbolic/constraint checking.

    Unlike the old version, this keeps comma-formatted numbers together:
      1,059,955 -> 1059955
    """
    text = normalize_math_symbols(text or "")
    pattern = r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?"
    nums = re.findall(pattern, text)
    return [normalize_number(n) for n in nums]

def normalize_number(token: str) -> str:
    token = str(token).strip().replace(",", "")
    try:
        value = float(token)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return token

def extract_equations(text: str) -> List[str]:
    """
    Extract simple arithmetic equations.

    Supports examples:
    2 * 60 = 120
    2 × 60 = 120
    720 ÷ 120 = 6
    24 - 14 = 10
    3 + 5 + 2 = 10
    1,410 + 6,908 = 8,318
    56 / 84 = 2/3

    Important:
    This function must return only full equation strings.
    It must not return standalone numbers such as step indices.
    """
    text = normalize_math_symbols(text or "")
    integer_or_decimal = (
        r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
        r"|[-+]?\d+(?:\.\d+)?"
    )

    fraction = r"[-+]?\d+\s*/\s*[-+]?\d+"
    # Fraction must appear before ordinary number, otherwise 2/3 becomes 2.
    value_pattern = rf"(?:{fraction}|{integer_or_decimal})"
    equation_pattern = (
        rf"{value_pattern}"
        rf"(?:\s*[\+\-\*/]\s*{value_pattern})+"
        rf"\s*=\s*"
        rf"{value_pattern}"
    )
    equations = []
    for match in re.finditer(equation_pattern, text):
        equation = match.group(0).strip()
        if "=" in equation:
            equations.append(equation)
    return equations

def extract_assignment_equations(text: str) -> List[str]:
    """
    Extract simple assignment-style calculations.
    """
    return extract_equations(text)


def has_final_answer(reasoning: str) -> bool:
    return bool(extract_final_answer(reasoning))

def is_empty_reasoning(reasoning: str) -> bool:
    return reasoning is None or not str(reasoning).strip()

if __name__ == "__main__":
    demos = [
        "Final Answer: 1,059,955",
        "Final Answer: 12:50",
        "Final Answer: 2:3",
        "Final Answer: 17/7",
        "Final Answer: 12.0",
        "Final Answer: No",
    ]
    for demo in demos:
        print(demo, "=>", extract_final_answer(demo))