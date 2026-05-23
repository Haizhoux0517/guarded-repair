import json
import os
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


SAVE_PATH = "data/multiarith_full.jsonl"


DATASET_CANDIDATES: List[Tuple[str, Optional[str]]] = [
    ("ChilleD/MultiArith", None),
    ("MultiArith", None),
    ("multi_arith", None),
]


QUESTION_KEYS = [
    "question",
    "Question",
    "body",
    "Body",
    "problem",
    "Problem",
    "sQuestion",
]

ANSWER_KEYS = [
    "answer",
    "Answer",
    "final_ans",
    "final_answer",
    "target",
    "Target",
    "lSolutions",
]


def normalize_answer(answer: Any) -> str:
    """
    Normalize numeric answers into the same style used by the current pipeline.
    """

    if isinstance(answer, list):
        if len(answer) == 0:
            return ""
        answer = answer[0]

    answer = str(answer).strip()
    answer = answer.replace(",", "")

    # Remove common wrapper artifacts.
    answer = answer.replace("####", "").strip()

    try:
        value = float(answer)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return answer


def get_first_existing(item: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def load_multiarith_dataset():
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

            raise ValueError(f"No usable split found in {dataset_name}")

        except Exception as e:
            last_error = e
            print(f"Failed to load {dataset_name}: {e}")

    raise RuntimeError(f"Could not load MultiArith from known candidates. Last error: {last_error}")


def main() -> None:
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    dataset, dataset_name, split_name = load_multiarith_dataset()

    print("Loaded dataset:", dataset_name)
    print("Using split:", split_name)
    print("Columns:", dataset.column_names)
    print("Total raw rows:", len(dataset))

    samples = []

    for idx, item in enumerate(dataset):
        question = get_first_existing(item, QUESTION_KEYS)
        answer = get_first_existing(item, ANSWER_KEYS)

        if question is None:
            raise ValueError(
                f"Could not find question field at row {idx}. "
                f"Available keys: {list(item.keys())}. Item: {item}"
            )

        if answer is None:
            raise ValueError(
                f"Could not find answer field at row {idx}. "
                f"Available keys: {list(item.keys())}. Item: {item}"
            )

        # Some datasets store answer as a list.
        norm_answer = normalize_answer(answer)

        samples.append(
            {
                "id": idx,
                "question": str(question).strip(),
                "answer": norm_answer,
                "raw_answer": answer if isinstance(answer, str) else str(answer),
                "source": "multiarith",
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