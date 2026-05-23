import os
import json
from tqdm import tqdm

from config import (
    FLASH_MODEL,
    NUM_SAMPLES,
    DATASET_SEED,
    DATASET_SPLIT,
    LOCAL_DATASET_PATH,
    INITIAL_CACHE_PATH,
)

from src.dataset_loader import load_gsm8k
from src.deepseek_client import call_deepseek
from src.prompts import REASONER_PROMPT
from src.evaluator import is_correct
from src.reasoning_parser import extract_final_answer


def read_jsonl(path):
    items = []
    if not os.path.exists(path):
        return items

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def generate_one(problem):
    messages = [
        {
            "role": "user",
            "content": REASONER_PROMPT.format(problem=problem),
        }
    ]

    return call_deepseek(
        model=FLASH_MODEL,
        messages=messages,
        temperature=0.0,
    )


def main():
    samples = load_gsm8k(
        num_samples=NUM_SAMPLES,
        seed=DATASET_SEED,
        split=DATASET_SPLIT,
        local_path=LOCAL_DATASET_PATH,
        rebuild=False,
    )

    cache_items = read_jsonl(INITIAL_CACHE_PATH)

    if not cache_items:
        raise RuntimeError(
            f"No cache found at {INITIAL_CACHE_PATH}. Run generate_initial_cache.py first."
        )

    cache_by_id = {item["id"]: item for item in cache_items}

    updated = []
    refill_count = 0
    still_failed = 0

    for idx, sample in enumerate(tqdm(samples, desc="Refilling cache failures")):
        old_item = cache_by_id.get(idx)

        if old_item is None:
            old_item = {
                "id": idx,
                "problem": sample["question"],
                "gold_answer": sample["answer"],
                "initial_reasoning": "",
                "initial_answer": None,
                "initial_correct": False,
                "generation_failed": True,
            }

        needs_refill = (
            old_item.get("generation_failed") is True
            or not old_item.get("initial_reasoning", "").strip()
        )

        if not needs_refill:
            updated.append(old_item)
            continue

        try:
            reasoning = generate_one(sample["question"])

            if reasoning and reasoning.strip():
                answer = extract_final_answer(reasoning)
                correct = is_correct(answer, sample["answer"])

                new_item = {
                    "id": idx,
                    "problem": sample["question"],
                    "gold_answer": sample["answer"],
                    "initial_reasoning": reasoning.strip(),
                    "initial_answer": answer,
                    "initial_correct": correct,
                    "generation_failed": False,
                }

                refill_count += 1
                updated.append(new_item)

            else:
                old_item["generation_failed"] = True
                still_failed += 1
                updated.append(old_item)

        except RuntimeError as e:
            print(f"[Still failed] sample_id={idx}, error={e}")
            old_item["generation_failed"] = True
            still_failed += 1
            updated.append(old_item)

    updated = sorted(updated, key=lambda x: x["id"])
    write_jsonl(INITIAL_CACHE_PATH, updated)

    total = len(updated)
    correct = sum(1 for item in updated if item.get("initial_correct"))
    failures = sum(1 for item in updated if item.get("generation_failed"))

    print("Cache refill finished.")
    print(f"Saved to: {INITIAL_CACHE_PATH}")
    print(f"Total: {total}")
    print(f"Refilled: {refill_count}")
    print(f"Still failed this run: {still_failed}")
    print(f"Total generation failures remaining: {failures}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct / total if total else 0}")


if __name__ == "__main__":
    main()