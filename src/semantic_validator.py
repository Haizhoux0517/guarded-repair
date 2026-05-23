import json
import re
from typing import Dict, Any
from config import PRO_MODEL
from src.deepseek_client import call_deepseek
from src.prompts import SEMANTIC_VALIDATOR_PROMPT

DEFAULT_SEMANTIC_RESULT = {
    "is_semantically_valid": True,
    "semantic_score": 1.0,
    "error_type": "none",
    "needs_repair": False,
    "explanation": "Semantic validation skipped or failed safely.",
    "suspected_issue": "",
}

def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robustly extract the first JSON object from model output.
    """
    if not text:
        raise ValueError("Empty semantic validator response")
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in semantic validator response")

    return json.loads(match.group(0))

def normalize_semantic_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the semantic validator result has stable fields.
    """
    result = DEFAULT_SEMANTIC_RESULT.copy()
    result["is_semantically_valid"] = bool(
        data.get("is_semantically_valid", result["is_semantically_valid"])
    )
    try:
        result["semantic_score"] = float(
            data.get("semantic_score", result["semantic_score"])
        )
    except Exception:
        result["semantic_score"] = 1.0
    result["semantic_score"] = max(0.0, min(1.0, result["semantic_score"]))
    result["error_type"] = str(data.get("error_type", result["error_type"]))
    result["needs_repair"] = bool(data.get("needs_repair", result["needs_repair"]))
    result["explanation"] = str(data.get("explanation", result["explanation"]))
    result["suspected_issue"] = str(data.get("suspected_issue", result["suspected_issue"]))
    return result

def validate_semantics(problem: str, reasoning: str) -> Dict[str, Any]:
    """
    LLM-assisted semantic validation.

    This checks whether the reasoning correctly interprets the problem statement.
    It does not use the gold answer.
    """
    if reasoning is None or not reasoning.strip():
        return {
            "is_semantically_valid": False,
            "semantic_score": 0.0,
            "error_type": "generation_failure",
            "needs_repair": True,
            "explanation": "Reasoning is empty, so semantic validity cannot be established.",
            "suspected_issue": "empty_reasoning",
        }
    messages = [
        {
            "role": "user",
            "content": SEMANTIC_VALIDATOR_PROMPT.format(
                problem=problem,
                reasoning=reasoning,
            ),
        }
    ]
    try:
        response = call_deepseek(
            model=PRO_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        )
        data = extract_json_object(response)
        return normalize_semantic_result(data)
    except Exception as e:
        fallback = DEFAULT_SEMANTIC_RESULT.copy()
        fallback["explanation"] = f"Semantic validation failed safely: {e}"
        return fallback