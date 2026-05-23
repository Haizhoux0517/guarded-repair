import json
import os
import random
from typing import Any, Dict, List

from datasets import load_dataset


OUTPUT_PATH = "data/gsm8k_sample_500.jsonl"
SAMPLE_SIZE = 500
SEED = 42


def extract_gold_answer(answer_text: str) -> str:
    """
    GSM8K answers usually contain final answer after ####.
    Example:
        "... #### 42"
    """
    if answer_text is None:
        return ""

    answer_text = str(answer_text)

    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()

    return answer_text.strip()


def normalize_gsm8k_item(item: Dict[str, Any], sample_id: int) -> Dict[str, Any]:
    question = item.get("question", "")
    full_answer = item.get("answer", "")
    gold_answer = extract_gold_answer(full_answer)

    return {
        "id": sample_id,
        "problem": question,
        "question": question,
        "answer": gold_answer,
        "gold_answer": gold_answer,
        "full_solution": full_answer,
        "source": "gsm8k",
        "split": "test",
    }


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading GSM8K test split from HuggingFace...")
    dataset = load_dataset("gsm8k", "main", split="test")

    total = len(dataset)
    print(f"Total GSM8K test samples: {total}")

    if SAMPLE_SIZE > total:
        raise ValueError(
            f"SAMPLE_SIZE={SAMPLE_SIZE} is larger than test split size={total}"
        )

    rng = random.Random(SEED)
    indices: List[int] = list(range(total))
    sampled_indices = rng.sample(indices, SAMPLE_SIZE)

    rows = [
        normalize_gsm8k_item(dataset[idx], sample_id=i)
        for i, idx in enumerate(sampled_indices)
    ]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Saved sampled dataset.")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Sample size: {len(rows)}")
    print(f"Seed: {SEED}")

    print("\nPreview first item:")
    print(json.dumps(rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()