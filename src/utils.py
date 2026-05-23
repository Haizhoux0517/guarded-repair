import json
import os
import re



def save_jsonl(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_json_parse(text):

    """

    Robust JSON parser for LLM outputs.

    It first tries direct parsing, then extracts the first JSON object.

    """
    if not text:
        return {
            "is_consistent": False,
            "error_step": None,
            "error_type": "unknown",
            "explanation": "Empty response.",
        }
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass

    return {
        "is_consistent": False,
        "error_step": None,
        "error_type": "unknown",
        "explanation": text[:500],
    }