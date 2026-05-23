import json
import os
from datasets import load_dataset

from src.dataset_loader import extract_gold_answer


SAVE_PATH = "data/gsm8k_test_full.jsonl"


def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    dataset = load_dataset("gsm8k", "main", split="test")

    samples = []
    for idx, item in enumerate(dataset):
        samples.append(
            {
                "id": idx,
                "question": item["question"],
                "answer": extract_gold_answer(item["answer"]),
                "raw_answer": item["answer"],
                "source": "gsm8k",
                "split": "test",
                "subset": "full_test",
            }
        )

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("Saved:", SAVE_PATH)
    print("Total:", len(samples))
    print("First ID:", samples[0]["id"])
    print("Last ID:", samples[-1]["id"])


if __name__ == "__main__":
    main()