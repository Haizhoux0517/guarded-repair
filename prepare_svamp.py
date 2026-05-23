import json
import os
from typing import Any

from datasets import load_dataset


SAVE_PATH = "data/svamp_full.jsonl"


def normalize_answer(answer: Any) -> str:
    """
    Normalize SVAMP numeric answers into the same style used by the GSM8K pipeline.
    """
    answer = str(answer).strip()
    answer = answer.replace(",", "")

    # Convert 12.0 -> 12 when safe.
    try:
        value = float(answer)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return answer


def get_field(item: dict, candidates: list[str], default: str = "") -> str:
    """
    Read a field robustly because different SVAMP mirrors may use slightly
    different column names.
    """
    for key in candidates:
        if key in item and item[key] is not None:
            return str(item[key])
    return default


def main() -> None:
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # Common HuggingFace SVAMP mirror.
    # If this fails, send me the full error and we will switch dataset source.
    dataset = load_dataset("ChilleD/SVAMP", split="test")

    samples = []

    for idx, item in enumerate(dataset):
        body = get_field(item, ["Body", "body", "context", "Context"])
        question = get_field(item, ["Question", "question"])
        answer = get_field(item, ["Answer", "answer", "Result", "result"])

        problem = (body.strip() + " " + question.strip()).strip()

        if not problem:
            raise ValueError(f"Empty problem at row {idx}: {item}")

        if answer == "":
            raise ValueError(f"Empty answer at row {idx}: {item}")

        samples.append(
            {
                "id": idx,
                "question": problem,
                "answer": normalize_answer(answer),
                "raw_answer": str(answer),
                "source": "svamp",
                "split": "test",
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