import json
from collections import Counter


RAW_RESULTS_PATH = "outputs/raw_results.jsonl"
ERROR_ANALYSIS_PATH = "outputs/error_analysis.jsonl"


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def classify_error(item):
    evaluation = item.get("evaluation", {})
    meta = item.get("initial_meta_diagnosis", {})

    if not item.get("initial_reasoning", "").strip():
        return "generation_failure"

    if evaluation.get("initial_correct") is True:
        return "correct"

    error_type = meta.get("error_type", "unknown")

    if error_type == "arithmetic_error":
        return "arithmetic_error"

    if error_type == "missing_constraint":
        return "missing_constraint"

    if error_type == "generation_failure":
        return "generation_failure"

    return "semantic_or_other_error"


def main():
    results = load_jsonl(RAW_RESULTS_PATH)

    counter = Counter()

    wrong_cases = []

    for item in results:
        label = classify_error(item)
        counter[label] += 1

        if label != "correct":
            wrong_cases.append(
                {
                    "id": item.get("id"),
                    "problem": item.get("problem"),
                    "gold_answer": item.get("gold_answer"),
                    "initial_answer": item.get("evaluation", {}).get("initial_answer"),
                    "final_answer": item.get("evaluation", {}).get("final_answer"),
                    "error_label": label,
                    "meta_error_type": item.get("initial_meta_diagnosis", {}).get("error_type"),
                    "meta_score": item.get("initial_meta_diagnosis", {}).get("global_consistency_score"),
                    "initial_reasoning": item.get("initial_reasoning"),
                }
            )

    print("=== Error Analysis Summary ===")
    for k, v in counter.items():
        print(f"{k}: {v}")

    print("\n=== Wrong Cases ===")
    for case in wrong_cases:
        print("-" * 80)
        print(f"ID: {case['id']}")
        print(f"Label: {case['error_label']}")
        print(f"Gold: {case['gold_answer']}")
        print(f"Initial: {case['initial_answer']}")
        print(f"Final: {case['final_answer']}")
        print(f"Meta error type: {case['meta_error_type']}")
        print(f"Meta score: {case['meta_score']}")
        print(f"Problem: {case['problem']}")


if __name__ == "__main__":
    main()