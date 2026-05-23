import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple


INPUT_PATH = Path("data/asdiv_full.jsonl")
OUTPUT_PATH = Path("data/asdiv_numeric_full.jsonl")
REJECTED_PATH = Path("data/asdiv_numeric_rejected.jsonl")


NON_NUMERIC_QUESTION_PATTERNS = [
    r"\bwho\b",
    r"\bwhich\b",
    r"\bwhat color\b",
    r"\bwhat colour\b",
    r"\bdoes\b.*\benough\b",
    r"\bdo\b.*\benough\b",
    r"\bdid\b.*\benough\b",
    r"\bis\b.*\benough\b",
    r"\bare\b.*\benough\b",
    r"\bdoes\b.*\bequal\b",
    r"\bdo\b.*\bequal\b",
    r"\bequal the latter\b",
    r"\btrue or false\b",
    r"\byes or no\b",
]


NON_NUMERIC_ANSWER_VALUES = {
    "yes",
    "no",
    "true",
    "false",
}


# Common categorical/string answers seen in ASDiv-style problems.
# This is not meant to solve them; it excludes them from numeric repair evaluation.
CATEGORICAL_ANSWER_PATTERNS = [
    r"^[A-Za-z][A-Za-z\s\.\-']*$",   # names, colors, words
]


def is_plain_numeric_answer(answer: str) -> bool:
    """
    Keep answers that are directly numeric and suitable for numeric evaluation.

    Accepted:
      42
      -3
      12.5
      1059955
      17/7

    Rejected:
      yes/no
      Mrs. Hilt
      Purple
      12:50
      2:3
    """

    if answer is None:
        return False

    ans = str(answer).strip()
    ans = ans.strip(".。;,，")
    ans = ans.replace(",", "")

    if not ans:
        return False

    if ans.lower() in NON_NUMERIC_ANSWER_VALUES:
        return False

    # Reject time / ratio style. It is ambiguous whether 2:3 is ratio or time.
    if re.fullmatch(r"\d{1,4}:\d{1,4}", ans):
        return False

    # Accept integer / decimal.
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", ans):
        return True

    # Accept fractions.
    if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", ans):
        return True

    return False


def question_looks_non_numeric(question: str) -> bool:
    """
    Detect question types where the desired output is likely categorical
    rather than numeric.
    """

    q = str(question or "").strip().lower()

    for pattern in NON_NUMERIC_QUESTION_PATTERNS:
        if re.search(pattern, q):
            return True

    return False


def is_money_decimal_answer(answer: str) -> bool:
    """
    Detect dollar-style decimal answers such as 0.46.
    These are numeric, but they often conflict with model outputs in cents.
    We keep them by default because they are still numeric, but record them
    in the metadata so they can be analyzed separately.
    """

    ans = str(answer or "").strip().replace(",", "")

    return bool(re.fullmatch(r"0\.\d+", ans))


def classify_sample(sample: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Return:
      keep: bool
      reason: str
    """

    question = str(sample.get("question", ""))
    answer = str(sample.get("answer", ""))

    if not answer.strip():
        return False, "empty_answer"

    if answer.lower().strip() in NON_NUMERIC_ANSWER_VALUES:
        return False, "yes_no_answer"

    if question_looks_non_numeric(question):
        return False, "non_numeric_question_type"

    if not is_plain_numeric_answer(answer):
        return False, "non_numeric_or_ambiguous_answer_format"

    return True, "kept_numeric"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}. Run prepare_asdiv.py first."
        )

    kept = []
    rejected = []

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            sample = json.loads(line)
            keep, reason = classify_sample(sample)

            sample = dict(sample)
            sample["numeric_subset_filter_reason"] = reason
            sample["is_money_decimal_answer"] = is_money_decimal_answer(
                sample.get("answer", "")
            )

            if keep:
                sample["numeric_subset_id"] = len(kept)
                kept.append(sample)
            else:
                rejected.append(sample)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for sample in kept:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with REJECTED_PATH.open("w", encoding="utf-8") as f:
        for sample in rejected:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("Input:", INPUT_PATH)
    print("Output numeric subset:", OUTPUT_PATH)
    print("Rejected output:", REJECTED_PATH)
    print("Total input:", len(kept) + len(rejected))
    print("Kept numeric:", len(kept))
    print("Rejected:", len(rejected))

    money_decimal_count = sum(
        1 for x in kept if x.get("is_money_decimal_answer")
    )

    print("Kept money decimal answers:", money_decimal_count)

    print("\nFirst kept sample:")
    print(json.dumps(kept[0], ensure_ascii=False, indent=2) if kept else "None")

    print("\nFirst rejected sample:")
    print(json.dumps(rejected[0], ensure_ascii=False, indent=2) if rejected else "None")


if __name__ == "__main__":
    main()