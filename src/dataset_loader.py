import json
import os
from typing import List, Dict, Any

from datasets import load_dataset


DATA_DIR = "data"
DEFAULT_LOCAL_PATH = os.path.join(DATA_DIR, "gsm8k_sample_200.jsonl")


def extract_gold_answer(answer_text: str) -> str:
    """
    Extract the final numeric answer from GSM8K answer field.

    GSM8K format usually looks like:
    "... reasoning ... #### 42"
    """

    if "####" in answer_text:
        answer = answer_text.split("####")[-1].strip()
    else:
        answer = answer_text.strip()

    return answer.replace(",", "")


def save_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
    """
    Save data as jsonl.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load data from jsonl.
    """

    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return data


def build_gsm8k_subset(
    num_samples: int = 200,
    seed: int = 42,
    split: str = "test",
    save_path: str = DEFAULT_LOCAL_PATH,
) -> List[Dict[str, Any]]:
    """
    Download GSM8K from HuggingFace, shuffle with fixed seed,
    select a fixed subset, and save it locally.
    """

    dataset = load_dataset("gsm8k", "main", split=split)

    dataset = dataset.shuffle(seed=seed)
    subset = dataset.select(range(num_samples))

    samples = []

    for idx, item in enumerate(subset):
        samples.append(
            {
                "id": idx,
                "question": item["question"],
                "answer": extract_gold_answer(item["answer"]),
                "raw_answer": item["answer"],
                "source": "gsm8k",
                "split": split,
                "seed": seed,
            }
        )

    save_jsonl(save_path, samples)

    return samples


def load_gsm8k(
    num_samples: int = 200,
    seed: int = 42,
    split: str = "test",
    local_path: str = DEFAULT_LOCAL_PATH,
    rebuild: bool = False,
) -> List[Dict[str, Any]]:
    """
    Main dataset loading function.

    If local file exists, load from local file.
    Otherwise, download from HuggingFace and create a fixed subset.

    Args:
        num_samples: number of GSM8K examples to use
        seed: random seed for reproducibility
        split: usually "test"
        local_path: saved jsonl path
        rebuild: if True, ignore local cache and rebuild subset

    Returns:
        List of samples:
        [
            {
                "id": 0,
                "question": "...",
                "answer": "42",
                "raw_answer": "... #### 42"
            }
        ]
    """

    if os.path.exists(local_path) and not rebuild:
        return load_jsonl(local_path)

    return build_gsm8k_subset(
        num_samples=num_samples,
        seed=seed,
        split=split,
        save_path=local_path,
    )


if __name__ == "__main__":
    samples = load_gsm8k(num_samples=200, seed=42, rebuild=True)

    print(f"Loaded {len(samples)} samples.")
    print(samples[0])