from tqdm import tqdm

from config import (
    FLASH_MODEL,
    NUM_SAMPLES,
    DATASET_SEED,
    DATASET_SPLIT,
    LOCAL_DATASET_PATH,
)

from src.dataset_loader import load_gsm8k
from src.deepseek_client import call_deepseek
from src.prompts import REASONER_PROMPT
from src.reasoning_parser import extract_final_answer
from src.evaluator import is_correct
from src.utils import save_jsonl


INITIAL_CACHE_PATH = "outputs/cache/initial_reasoning_cache.jsonl"


def generate_initial_reasoning(problem):
    messages = [
        {
            "role": "user",
            "content": REASONER_PROMPT.format(problem=problem),
        }
    ]

    try:
        reasoning = call_deepseek(
            model=FLASH_MODEL,
            messages=messages,
            temperature=0.0,
        )

        if reasoning and reasoning.strip():
            return reasoning.strip(), False

        return "", True

    except RuntimeError as e:
        print(f"[Initial generation failed] {e}")
        return "", True


def main():
    samples = load_gsm8k(
        num_samples=NUM_SAMPLES,
        seed=DATASET_SEED,
        split=DATASET_SPLIT,
        local_path=LOCAL_DATASET_PATH,
        rebuild=False,
    )

    cache_items = []

    for idx, sample in enumerate(tqdm(samples, desc="Generating initial cache")):
        problem = sample["question"]
        gold_answer = sample["answer"]

        initial_reasoning, generation_failed = generate_initial_reasoning(problem)
        initial_answer = extract_final_answer(initial_reasoning)

        item = {
            "id": idx,
            "problem": problem,
            "gold_answer": gold_answer,

            "initial_reasoning": initial_reasoning,
            "initial_answer": initial_answer,
            "initial_correct": is_correct(initial_answer, gold_answer),

            "generation_failed": generation_failed,
            "model": FLASH_MODEL,
        }

        cache_items.append(item)

    save_jsonl(INITIAL_CACHE_PATH, cache_items)

    total = len(cache_items)
    correct = sum(item["initial_correct"] for item in cache_items)
    failed = sum(item["generation_failed"] for item in cache_items)

    print("Initial cache generated.")
    print(f"Saved to: {INITIAL_CACHE_PATH}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct / total if total else 0}")
    print(f"Generation failures: {failed}")


if __name__ == "__main__":
    main()