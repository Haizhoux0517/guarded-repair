from tqdm import tqdm
import os

from config import (
    NUM_SAMPLES,
    DATASET_SEED,
    DATASET_SPLIT,
    LOCAL_DATASET_PATH,
)

from src.dataset_loader import load_gsm8k
from src.ollama_client import call_ollama
from src.reasoning_parser import extract_final_answer
from src.evaluator import is_correct
from src.utils import save_jsonl

INITIAL_CACHE_PATH = "outputs/cache/initial_reasoning_cache.jsonl"

def build_small_model_prompt(problem: str) -> str:
    """
    Prompt for small initial reasoner.

    Keep this simple. Small models often fail when the prompt is too complex
    or asks for JSON.
    """

    return f"""You are a careful mathematical reasoning assistant.

Solve the following grade-school math problem step by step.

You must output in exactly this format:
Step 1: ...
Step 2: ...
Step 3: ...
Final Answer: <number>

Problem:
{problem}
"""


def generate_initial_reasoning(problem: str):
    small_model = os.getenv("SMALL_INITIAL_MODEL", "qwen2.5:1.5b")
    prompt = build_small_model_prompt(problem)
    try:
        reasoning = call_ollama(
            prompt=prompt,
            model=small_model,
            temperature=0.0,
            max_tokens=512,
        )
        if reasoning and reasoning.strip() and not reasoning.startswith("GENERATION_ERROR"):
            return reasoning.strip(), False
        print(f"[Initial generation failed] Empty or invalid Ollama response: {reasoning}")
        return "", True
    except Exception as e:
        print(f"[Initial generation failed] {e}")
        return "", True

def main():
    small_model = os.getenv("SMALL_INITIAL_MODEL", "qwen2.5:1.5b")
    samples = load_gsm8k(
        num_samples=NUM_SAMPLES,
        seed=DATASET_SEED,
        split=DATASET_SPLIT,
        local_path=LOCAL_DATASET_PATH,
        rebuild=False,
    )
    cache_items = []
    for idx, sample in enumerate(tqdm(samples, desc="Generating small-model initial cache")):
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
            "model": small_model,
            "initial_reasoner_type": "ollama_small_model",
        }
        cache_items.append(item)

    save_jsonl(INITIAL_CACHE_PATH, cache_items)
    total = len(cache_items)
    correct = sum(item["initial_correct"] for item in cache_items)
    failed = sum(item["generation_failed"] for item in cache_items)
    print("Small-model initial cache generated.")
    print(f"Model: {small_model}")
    print(f"Saved to: {INITIAL_CACHE_PATH}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct / total if total else 0}")
    print(f"Generation failures: {failed}")

if __name__ == "__main__":
    main()