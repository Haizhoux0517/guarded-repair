import json
import os
from typing import Any, Dict, List, Optional, Tuple
import re
from datasets import load_dataset


SAVE_PATH = "data/asdiv_full.jsonl"


DATASET_CANDIDATES: List[Tuple[str, Optional[str]]] = [
    ("EleutherAI/asdiv", None),
    ("asdiv", None),
    ("ChilleD/ASDiv", None),
    ("MU-NLPC/Calc-asdiv_a", None),
]


QUESTION_KEYS = [
    "question",
    "Question",
    "body",
    "Body",
    "problem",
    "Problem",
    "sQuestion",
    "input",
    "text",
]

ANSWER_KEYS = [
    "answer",
    "Answer",
    "final_answer",
    "final_ans",
    "target",
    "Target",
    "solution",
    "Solution",
    "answers",
]


BODY_KEYS = [
    "body",
    "Body",
    "context",
    "Context",
    "sBody",
]


def normalize_answer(answer: Any) -> str:
    """
    Normalize ASDiv answers without destroying time, ratio, or fraction formats.
    """

    if isinstance(answer, list):
        if len(answer) == 0:
            return ""
        answer = answer[0]

    answer = str(answer).strip()
    answer = answer.replace("，", ",")
    answer = answer.replace("####", "").strip()

    # Remove unit parentheses, e.g., "9 (apples)" -> "9".
    answer = re.sub(r"\([^)]*\)", "", answer).strip()

    # yes/no answers.
    if answer.lower() in {"yes", "no"}:
        return answer.lower()

    # Preserve time / ratio format.
    match = re.search(r"(?<!\d)(\d{1,4}\s*:\s*\d{1,4})(?!\d)", answer)
    if match:
        return match.group(1).replace(" ", "")

    # Preserve fractions.
    match = re.search(r"(?<!\d)([-+]?\d+\s*/\s*[-+]?\d+)(?!\d)", answer)
    if match:
        return match.group(1).replace(" ", "")

    # Preserve comma numbers.
    match = re.search(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", answer)
    if match:
        return match.group(0).replace(",", "")

    # Plain integer / decimal.
    match = re.search(r"-?\d+(?:\.\d+)?", answer)
    if match:
        value = float(match.group(0))
        if value.is_integer():
            return str(int(value))
        return str(value)

    return answer


def get_first_existing(item: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def load_asdiv_dataset():
    last_error = None

    for dataset_name, config_name in DATASET_CANDIDATES:
        try:
            print(f"Trying dataset: {dataset_name}")

            if config_name:
                dataset_dict = load_dataset(dataset_name, config_name)
            else:
                dataset_dict = load_dataset(dataset_name)

            print(dataset_dict)

            if "test" in dataset_dict:
                return dataset_dict["test"], dataset_name, "test"

            if "validation" in dataset_dict:
                return dataset_dict["validation"], dataset_name, "validation"

            if "train" in dataset_dict:
                return dataset_dict["train"], dataset_name, "train"

            # Some datasets may return a single Dataset instead of DatasetDict.
            return dataset_dict, dataset_name, "unknown"

        except Exception as e:
            last_error = e
            print(f"Failed to load {dataset_name}: {e}")

    raise RuntimeError(
        f"Could not load ASDiv from known candidates. Last error: {last_error}"
    )


def build_question(item: Dict[str, Any]) -> str:
    body = get_first_existing(item, BODY_KEYS)
    question = get_first_existing(item, QUESTION_KEYS)

    if body is not None and question is not None and str(body).strip() not in str(question):
        return (str(body).strip() + " " + str(question).strip()).strip()

    if question is not None:
        return str(question).strip()

    if body is not None:
        return str(body).strip()

    raise ValueError(f"Could not find question/body field. Keys: {list(item.keys())}")


def main() -> None:
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    dataset, dataset_name, split_name = load_asdiv_dataset()

    print("Loaded dataset:", dataset_name)
    print("Using split:", split_name)
    print("Columns:", dataset.column_names)
    print("Total raw rows:", len(dataset))

    samples = []

    for idx, item in enumerate(dataset):
        question = build_question(item)
        answer = get_first_existing(item, ANSWER_KEYS)

        if answer is None:
            raise ValueError(
                f"Could not find answer field at row {idx}. "
                f"Available keys: {list(item.keys())}. Item: {item}"
            )

        norm_answer = normalize_answer(answer)

        if not question:
            raise ValueError(f"Empty question at row {idx}: {item}")

        if not norm_answer:
            raise ValueError(f"Empty normalized answer at row {idx}: {item}")

        samples.append(
            {
                "id": idx,
                "question": question,
                "answer": norm_answer,
                "raw_answer": answer if isinstance(answer, str) else str(answer),
                "source": "asdiv",
                "split": split_name,
                "dataset_name": dataset_name,
            }
        )

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("Saved:", SAVE_PATH)
    print("Total:", len(samples))
    print("First sample:")
    print(json.dumps(samples[0], ensure_ascii=False, indent=2))
    print("Last sample:")
    print(json.dumps(samples[-1], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()