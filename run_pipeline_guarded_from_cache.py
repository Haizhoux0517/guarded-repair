import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from config import PRO_MODEL, LOCAL_DATASET_PATH
from src.deepseek_client import call_deepseek
from src.ollama_client import call_ollama

REPAIR_BACKEND = os.getenv("REPAIR_BACKEND", "deepseek").strip().lower()
REPAIR_MODEL_NAME = os.getenv("REPAIR_MODEL", PRO_MODEL)
from src.symbolic_checker import check_reasoning
from src.constraint_checker import check_constraint_coverage
from src.meta_diagnoser import compute_meta_diagnosis
from src.evaluator import evaluate_result, summarize
from src.reasoning_parser import extract_final_answer
from src.semantic_graph_checker import check_semantic_graph
from deterministic_semantic_repair import deterministic_semantic_repair

def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

INITIAL_CACHE_PATH = "outputs/cache/initial_reasoning_cache.jsonl"
OUTPUT_DIR = os.getenv("GUARDED_OUTPUT_DIR", "outputs/from_cache_guarded")
RAW_RESULTS_PATH = os.path.join(OUTPUT_DIR, "raw_results.jsonl")
FINAL_REPORT_PATH = os.path.join(OUTPUT_DIR, "final_report.json")
ERROR_SET_PATH = os.path.join(OUTPUT_DIR, "error_set.jsonl")
ENABLE_GRAPH_GUARD = env_bool("ENABLE_GRAPH_GUARD", True)
ENABLE_DETERMINISTIC_HINT = env_bool("ENABLE_DETERMINISTIC_HINT", True)
ENABLE_RELAXED_SUPPORT_ACCEPT = env_bool(
    "ENABLE_RELAXED_SUPPORT_ACCEPT",
    False,
)
ENABLE_DIRECT_DETERMINISTIC_ACCEPT = env_bool(
    "ENABLE_DIRECT_DETERMINISTIC_ACCEPT",
    False,
)
DISABLE_EQUATION_SUPPORT_GUARD = env_bool(
    "DISABLE_EQUATION_SUPPORT_GUARD",
    False,
)

RELAX_MISSING_CONSTRAINT_ACCEPT = env_bool(
    "RELAX_MISSING_CONSTRAINT_ACCEPT",
    False,
)
ENABLE_WEAK_REASONER_RELAXED_ACCEPT = env_bool(
    "ENABLE_WEAK_REASONER_RELAXED_ACCEPT",
    False,
)
ENABLE_LLM_REPAIR = env_bool("ENABLE_LLM_REPAIR", True)
MEDIUM_REPAIR_TRIGGER = env_bool("MEDIUM_REPAIR_TRIGGER", True)
ENABLE_JSON_REPAIR_OUTPUT = env_bool("ENABLE_JSON_REPAIR_OUTPUT", True)
ENABLE_FORMAT_RETRY = env_bool("ENABLE_FORMAT_RETRY", True)
ENABLE_CLEAN_SEMANTIC_IMPROVEMENT_ACCEPT = env_bool(
    "ENABLE_CLEAN_SEMANTIC_IMPROVEMENT_ACCEPT",
    True,
)

GRAPH_TRIGGER_THRESHOLD = 0.80
GRAPH_ACCEPT_MIN_SCORE = 0.60
GRAPH_MIN_SCORE_DROP = 0.05
MIN_REPAIR_LENGTH = 20
LLM_REPAIR_MAX_RETRIES = 1
LLM_REPAIR_MAX_TOKENS = 768
LLM_FORMAT_RETRY_MAX_RETRIES = 1
LLM_FORMAT_RETRY_MAX_TOKENS = 512

# Number of independent LLM repair candidates to generate per triggered sample.
# Keep default at 1 for backward compatibility; set to 3 for best-of-N experiments.
LLM_REPAIR_NUM_CANDIDATES = max(1, int(os.getenv("LLM_REPAIR_NUM_CANDIDATES", "1")))


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_local_jsonl_dataset(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    dataset: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSONL at {path}, line {line_idx}: {e}"
                ) from e
            if "problem" not in item and "question" in item:
                item["problem"] = item["question"]
            if "gold_answer" not in item and "answer" in item:
                item["gold_answer"] = item["answer"]
            if "answer" not in item and "gold_answer" in item:
                item["answer"] = item["gold_answer"]
            if "id" not in item:
                item["id"] = line_idx - 1
            dataset.append(item)
    return dataset

def load_initial_cache(path: str) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Initial cache not found: {path}\n"
            "Please run: python generate_initial_cache.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = str(item.get("id"))
            cache[sample_id] = item
    return cache

def fallback_extract_final_answer_from_rambling(text: str) -> str:
    """
    Conservative fallback for rambling repair outputs.

    This only extracts answers from explicit answer-like phrases.
    It does not simply take the last number from the whole text, because that
    can accept unrelated intermediate quantities.
    """

    if not text:
        return ""

    patterns = [
        r"(?:final answer|answer)\s*(?:is|:)\s*[-$]?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(?:so|therefore|thus),?\s*(?:the\s+)?answer\s*(?:is|:)\s*[-$]?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(?:final_answer)\s*[\"']?\s*[:=]\s*[\"']?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return str(matches[-1]).replace(",", "").strip()

    return ""

def safe_final_answer(reasoning: str) -> str:
    return extract_final_answer(reasoning)


def normalize_answer_for_compare(ans: Any) -> str:
    text = str(ans or "").strip()
    text = text.replace(",", "")

    try:
        value = float(text)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except Exception:
        return text


def answers_differ(a: Any, b: Any) -> bool:
    return normalize_answer_for_compare(a) != normalize_answer_for_compare(b)


def run_diagnostics(problem: str, reasoning: str) -> Dict[str, Any]:
    if not reasoning or not reasoning.strip():
        return {
            "checker_results": None,
            "constraint_results": None,
            "meta_diagnosis": {
                "is_consistent": False,
                "global_consistency_score": 0.0,
                "average_step_score": 0.0,
                "symbolic_coverage": 0.0,
                "constraint_coverage_score": 0.0,
                "missing_constraints": [],
                "error_step": None,
                "error_type": "generation_failure",
                "step_scores": [],
                "explanation": "Reasoning is empty.",
            },
            "semantic_graph": check_semantic_graph(problem, reasoning),
        }

    checker_results = check_reasoning(reasoning)
    constraint_results = check_constraint_coverage(problem, reasoning)
    meta_diagnosis = compute_meta_diagnosis(
        checker_results,
        constraint_results,
    )
    semantic_graph = check_semantic_graph(problem, reasoning)

    return {
        "checker_results": checker_results,
        "constraint_results": constraint_results,
        "meta_diagnosis": meta_diagnosis,
        "semantic_graph": semantic_graph,
    }


def has_high_risk_semantic_issue(graph: Dict[str, Any]) -> bool:
    error_type = str(graph.get("error_type", "none"))

    if error_type == "generation_failure":
        return True
    if graph.get("semantic_pattern_issues"):
        return True
    if graph.get("format_issues"):
        return True
    high_risk_patterns = [
        "times_more_interpretation",
        "per_entity_rate_missing",
        "equally_split_interpretation",
        "change_event_misinterpretation",
    ]
    if any(pattern in error_type for pattern in high_risk_patterns):
        return True
    return False


def should_attempt_repair(
    initial_meta: Dict[str, Any],
    initial_graph: Dict[str, Any],
    initial_reasoning: str,
) -> bool:
    if not initial_reasoning or not initial_reasoning.strip():
        return True

    meta_error_type = str(initial_meta.get("error_type", "none"))
    graph_error_type = str(initial_graph.get("error_type", "none"))
    meta_score = float(initial_meta.get("global_consistency_score", 1.0))

    if MEDIUM_REPAIR_TRIGGER:
        if meta_error_type in {
            "generation_failure",
            "arithmetic_error",
            "logical_contradiction",
        }:
            return True

        if graph_error_type == "generation_failure":
            return True
        if has_high_risk_semantic_issue(initial_graph):
            return True
        if meta_error_type == "missing_constraint" and meta_score < 0.90:
            return True
        if meta_score < 0.65:
            return True
        return False
    if not initial_meta.get("is_consistent", False):
        return True
    if meta_score < 0.75:
        return True
    if initial_graph.get("needs_repair", False):
        return True
    if float(initial_graph.get("semantic_graph_score", 1.0)) < GRAPH_TRIGGER_THRESHOLD:
        return True
    return False


def is_valid_repair_text(candidate: str) -> bool:
    if not candidate or not candidate.strip():
        return False
    candidate = candidate.strip()
    if len(candidate) < MIN_REPAIR_LENGTH:
        return False
    if "Final Answer" not in candidate:
        return False
    if not safe_final_answer(candidate):
        return False
    return True


def is_clean_llm_repair(candidate: str) -> Dict[str, Any]:
    text = candidate or ""
    lower = text.lower()

    banned_phrases = [
        "meta-diagnosis",
        "meta diagnosis",
        "previous reasoning",
        "we are asked to repair",
        "the diagnosis says",
        "i think",
        "maybe",
        "let's re-read",
        "why does",
        "the instruction says",
        "deterministic semantic hint",
        "provided hint",
        "we need to produce",
        "the problem:",
        "original problem:",
        "this is ambiguous",
        "at first glance",
        "repair hint",
        "semantic error type",
    ]

    for phrase in banned_phrases:
        if phrase in lower:
            return {
                "clean": False,
                "reason": f"Rejected because candidate contains meta-discussion phrase: {phrase}",
            }
    if text.count("Final Answer") != 1:
        return {
            "clean": False,
            "reason": "Rejected because candidate must contain exactly one Final Answer line.",
        }
    if len(text) > 1200:
        return {
            "clean": False,
            "reason": "Rejected because candidate is too long and likely contains rambling analysis.",
        }
    final_answer_pos = text.rfind("Final Answer")
    tail = text[final_answer_pos:].strip()

    if len(tail.splitlines()) > 2:
        return {
            "clean": False,
            "reason": "Rejected because text continues too much after Final Answer.",
        }
    return {
        "clean": True,
        "reason": "Candidate is clean.",
    }

def candidate_output_quality(candidate: str) -> Dict[str, Any]:
    if not is_valid_repair_text(candidate):
        return {
            "valid": False,
            "reason": "invalid_or_missing_final_answer",
        }
    clean_result = is_clean_llm_repair(candidate)
    if not clean_result.get("clean", False):
        return {
            "valid": False,
            "reason": clean_result.get("reason", "not_clean"),
        }

    return {
        "valid": True,
        "reason": "valid_clean_candidate",
    }


def run_deterministic_hint(
    problem: str,
    initial_reasoning: str,
    meta_diagnosis: Dict[str, Any],
    semantic_graph: Dict[str, Any],
) -> str:
    if not ENABLE_DETERMINISTIC_HINT:
        return ""

    try:
        candidate = deterministic_semantic_repair(
            problem=problem,
            reasoning=initial_reasoning,
            meta_diagnosis=meta_diagnosis,
            semantic_graph=semantic_graph,
        )
        return candidate or ""
    except TypeError:
        try:
            candidate = deterministic_semantic_repair(
                problem=problem,
                initial_reasoning=initial_reasoning,
                graph_result=semantic_graph,
            )
            return candidate or ""
        except TypeError:
            try:
                candidate = deterministic_semantic_repair(problem, initial_reasoning)
                return candidate or ""
            except Exception:
                return ""
        except Exception:
            return ""
    except Exception:
        return ""


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)

    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
    except Exception:
        return None

    return None


def json_repair_to_reasoning(raw_text: str) -> str:
    data = extract_json_object(raw_text)

    if not data:
        return raw_text.strip() if raw_text else ""
    steps = data.get("steps", [])
    final_answer = data.get("final_answer", "")
    if not isinstance(steps, list):
        return raw_text.strip()
    final_answer = str(final_answer).strip()
    if not final_answer:
        return raw_text.strip()
    lines: List[str] = []
    for idx, step in enumerate(steps, start=1):
        step_text = str(step).strip()
        if not step_text:
            continue
        step_text = step_text.rstrip(".")
        lines.append(f"Step {idx}: {step_text}.")
    lines.append(f"Final Answer: {final_answer}")
    return "\n".join(lines)


def build_json_repair_prompt(
    problem: str,
    initial_reasoning: str,
    meta_diagnosis: Dict[str, Any],
    semantic_graph: Dict[str, Any],
    deterministic_hint: str,
    attempt_idx: int = 0,
) -> str:
    semantic_error_type = semantic_graph.get("error_type", "none")
    meta_error_type = meta_diagnosis.get("error_type", "none")

    if attempt_idx == 0:
        attempt_style = (
            "Use the repair hint as diagnostic guidance when it is helpful. "
            "Correct the reasoning only if the previous answer is not supported by the problem."
        )
    elif attempt_idx == 1:
        attempt_style = (
            "Prioritize strict output formatting and concise arithmetic. "
            "Do not discuss ambiguity or diagnostics. If the original answer is defensible, preserve it."
        )
    else:
        attempt_style = (
            "Solve from the original problem from scratch using at most 4 compact steps. "
            "Use the previous reasoning only as a warning signal; do not copy it blindly. "
            "If the problem is ambiguous and the original answer is defensible, preserve the original answer."
        )

    return f"""
You are a strict math repair engine.

Return ONLY valid JSON.
Do NOT use markdown.
Do NOT write any explanation outside JSON.
Do NOT discuss diagnosis, previous reasoning, hint, uncertainty, ambiguity, or possible interpretations.

The JSON schema must be exactly:
{{
  "steps": ["short arithmetic equation or short factual statement"],
  "final_answer": "number"
}}

Rules:
- Use at most 4 steps.
- Prefer number-only arithmetic equations.
- Do not put units inside equations.
- Do not use chained equations such as "8 + 24 = 32 = ...".
- final_answer must contain only the answer number.
- If a repair hint is provided, use it as guidance, but do not mention the hint.
- Do not optimize for any specific dataset; solve the problem using its stated quantities and relations.
- Attempt strategy: {attempt_style}

Problem:
{problem}

Initial reasoning:
{initial_reasoning}

Repair hint:
{deterministic_hint}

Semantic error type:
{semantic_error_type}

Meta error type:
{meta_error_type}
""".strip()


def build_format_retry_prompt(
    problem: str,
    initial_reasoning: str,
    deterministic_hint: str,
    first_raw_response: str,
) -> str:
    return f"""
Your previous output was not valid for the required format.

Task:
Rewrite the solution as ONLY valid JSON.
Do not include markdown.
Do not include explanation outside JSON.
Do not discuss the prompt, diagnosis, previous reasoning, ambiguity, or uncertainty.

Required JSON schema:
{{
  "steps": ["short arithmetic equation or short factual statement"],
  "final_answer": "number"
}}

Rules:
- Use at most 4 steps.
- Use the original problem as the source of truth.
- You may use the repair hint as guidance, but do not mention it.
- Prefer clean arithmetic equations.
- final_answer must contain only the answer number.
- Do not optimize for any specific dataset or benchmark.

Problem:
{problem}

Initial reasoning:
{initial_reasoning}

Repair hint:
{deterministic_hint}

Invalid previous output:
{first_raw_response}
""".strip()


def call_repair_json_prompt(prompt: str, max_tokens: int, max_retries: int) -> str:
    system_prompt = (
        "You are a strict JSON-only math solver. "
        "Return only a valid JSON object. No markdown. No prose outside JSON."
    )

    if REPAIR_BACKEND in {"ollama", "local", "qwen"}:
        local_prompt = f"{system_prompt}\n\n{prompt}\n\nReturn only valid JSON."
        return call_ollama(
            prompt=local_prompt,
            model=REPAIR_MODEL_NAME,
            temperature=0.0,
            max_tokens=max_tokens,
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return call_deepseek(
        model=REPAIR_MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )


def call_llm_repair(
    problem: str,
    initial_reasoning: str,
    meta_diagnosis: Dict[str, Any],
    semantic_graph: Dict[str, Any],
    deterministic_hint: str = "",
    attempt_idx: int = 0,
) -> Tuple[str, Dict[str, Any]]:
    first_prompt = build_json_repair_prompt(
        problem=problem,
        initial_reasoning=initial_reasoning,
        meta_diagnosis=meta_diagnosis,
        semantic_graph=semantic_graph,
        deterministic_hint=deterministic_hint,
        attempt_idx=attempt_idx,
    )
    first_raw_response = call_repair_json_prompt(
        prompt=first_prompt,
        max_tokens=LLM_REPAIR_MAX_TOKENS,
        max_retries=LLM_REPAIR_MAX_RETRIES,
    )
    first_candidate = (
        json_repair_to_reasoning(first_raw_response)
        if ENABLE_JSON_REPAIR_OUTPUT
        else first_raw_response
    )
    first_quality = candidate_output_quality(first_candidate)
    metadata = {
        "attempt_idx": attempt_idx,
        "format_retry_used": False,
        "first_raw_response": first_raw_response,
        "first_candidate": first_candidate,
        "first_quality": first_quality,
        "retry_raw_response": "",
        "retry_candidate": "",
        "retry_quality": None,
    }
    if first_quality.get("valid", False):
        return first_candidate, metadata
    if not ENABLE_FORMAT_RETRY:
        return first_candidate, metadata
    retry_prompt = build_format_retry_prompt(
        problem=problem,
        initial_reasoning=initial_reasoning,
        deterministic_hint=deterministic_hint,
        first_raw_response=first_raw_response,
    )
    retry_raw_response = call_repair_json_prompt(
        prompt=retry_prompt,
        max_tokens=LLM_FORMAT_RETRY_MAX_TOKENS,
        max_retries=LLM_FORMAT_RETRY_MAX_RETRIES,
    )
    retry_candidate = (
        json_repair_to_reasoning(retry_raw_response)
        if ENABLE_JSON_REPAIR_OUTPUT
        else retry_raw_response
    )
    retry_quality = candidate_output_quality(retry_candidate)
    metadata.update(
        {
            "format_retry_used": True,
            "retry_raw_response": retry_raw_response,
            "retry_candidate": retry_candidate,
            "retry_quality": retry_quality,
        }
    )
    if retry_quality.get("valid", False):
        return retry_candidate, metadata
    return first_candidate, metadata


def is_date_like_binding_issue(issue: Dict[str, Any]) -> bool:
    context = str(issue.get("context", "")).lower()
    number = str(issue.get("number", "")).strip()
    month_words = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    if any(month in context for month in month_words):
        return True
    if "birthday" in context or "today" in context:
        return True
    if number in {"1", "19", "28", "29", "30", "31"} and (
        "day" in context or "date" in context or "year" in context
    ):
        return True

    return False

def graph_guard_accept(
    initial_graph: Dict[str, Any],
    candidate_graph: Dict[str, Any],
) -> Dict[str, Any]:
    initial_score = float(initial_graph.get("semantic_graph_score", 0.0))
    candidate_score = float(candidate_graph.get("semantic_graph_score", 0.0))
    candidate_error_type = str(candidate_graph.get("error_type", ""))
    semantic_pattern_issues = candidate_graph.get("semantic_pattern_issues", [])
    binding_issues = candidate_graph.get("binding_issues", [])
    format_issues = candidate_graph.get("format_issues", [])
    severe_candidate_error = False
    if "generation_failure" in candidate_error_type:
        severe_candidate_error = True
    if format_issues:
        severe_candidate_error = True
    if semantic_pattern_issues:
        severe_candidate_error = True
    non_date_binding_issues = [
        issue for issue in binding_issues
        if not is_date_like_binding_issue(issue)
    ]
    equation_numbers = candidate_graph.get("equation_numbers", [])
    if non_date_binding_issues and not equation_numbers:
        severe_candidate_error = True
    if severe_candidate_error:
        return {
            "accept": False,
            "reason": "Rejected because candidate still has severe semantic graph issue.",
            "initial_graph_score": initial_score,
            "candidate_graph_score": candidate_score,
            "candidate_error_type": candidate_error_type,
            "non_date_binding_issues": non_date_binding_issues,
        }
    if candidate_score < GRAPH_ACCEPT_MIN_SCORE:
        only_date_binding = (
            binding_issues
            and not non_date_binding_issues
            and not semantic_pattern_issues
            and not format_issues
        )
        if not only_date_binding:
            return {
                "accept": False,
                "reason": "Rejected because candidate semantic graph score is below threshold.",
                "initial_graph_score": initial_score,
                "candidate_graph_score": candidate_score,
                "candidate_error_type": candidate_error_type,
            }
    if candidate_score + GRAPH_MIN_SCORE_DROP < initial_score:
        initial_had_severe_issue = bool(
            initial_graph.get("semantic_pattern_issues")
            or initial_graph.get("format_issues")
            or initial_graph.get("error_type") == "generation_failure"
        )
        if not initial_had_severe_issue:
            return {
                "accept": False,
                "reason": "Rejected because candidate semantic graph score drops too much.",
                "initial_graph_score": initial_score,
                "candidate_graph_score": candidate_score,
                "candidate_error_type": candidate_error_type,
            }

    return {
        "accept": True,
        "reason": "Accepted by semantic graph guard.",
        "initial_graph_score": initial_score,
        "candidate_graph_score": candidate_score,
        "candidate_error_type": candidate_error_type,
    }


def deterministic_accept_repair(
    initial_reasoning: str,
    candidate: str,
    initial_meta: Dict[str, Any],
    candidate_meta: Dict[str, Any],
    initial_graph: Dict[str, Any],
    candidate_graph: Dict[str, Any],
) -> Dict[str, Any]:
    initial_answer = safe_final_answer(initial_reasoning)
    candidate_answer = safe_final_answer(candidate)

    initial_score = float(initial_meta.get("global_consistency_score", 0.0))
    candidate_score = float(candidate_meta.get("global_consistency_score", 0.0))
    candidate_graph_score = float(candidate_graph.get("semantic_graph_score", 0.0))
    candidate_graph_error_type = str(candidate_graph.get("error_type", ""))

    if not candidate_answer:
        return {
            "accept": False,
            "reason": "Rejected deterministic repair because candidate has no parseable final answer.",
            "initial_score": initial_score,
            "candidate_score": candidate_score,
            "candidate_graph_score": candidate_graph_score,
            "initial_answer": initial_answer,
            "candidate_answer": candidate_answer,
        }
    if not answers_differ(initial_answer, candidate_answer):
        return {
            "accept": False,
            "reason": "Rejected deterministic repair because final answer did not change.",
            "initial_score": initial_score,
            "candidate_score": candidate_score,
            "candidate_graph_score": candidate_graph_score,
            "initial_answer": initial_answer,
            "candidate_answer": candidate_answer,
        }
    if "generation_failure" in candidate_graph_error_type:
        return {
            "accept": False,
            "reason": "Rejected deterministic repair because candidate has severe semantic graph error.",
            "initial_score": initial_score,
            "candidate_score": candidate_score,
            "candidate_graph_score": candidate_graph_score,
            "initial_answer": initial_answer,
            "candidate_answer": candidate_answer,
        }
    if candidate_score < 0.75:
        return {
            "accept": False,
            "reason": "Rejected deterministic repair because candidate meta score is too low.",
            "initial_score": initial_score,
            "candidate_score": candidate_score,
            "candidate_graph_score": candidate_graph_score,
            "initial_answer": initial_answer,
            "candidate_answer": candidate_answer,
        }
    return {
        "accept": True,
        "reason": "Accepted deterministic repair under upper-bound mode.",
        "initial_score": initial_score,
        "candidate_score": candidate_score,
        "candidate_graph_score": candidate_graph_score,
        "initial_answer": initial_answer,
        "candidate_answer": candidate_answer,
    }

def normalize_numeric_answer_for_support(ans: str) -> str:
    """
    Normalize numeric answers for equation-support checking.

    Examples:
        "1,200" -> "1200"
        "23.0" -> "23"
    """

    if ans is None:
        return ""
    ans = str(ans).strip()
    ans = ans.replace(",", "")
    ans = ans.strip(".。;；,，")
    try:
        value = float(ans)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except Exception:
        return ans

def candidate_answer_is_supported_by_equation(
    candidate: str,
    candidate_answer: str,
) -> bool:
    """
    Return True if the candidate final answer is explicitly supported by
    arithmetic or standard number-theoretic derivation in the candidate reasoning.

    Supports:
        276 / 12 = 23
        Number of trays = 23
        LCM(6, 5) = 30
        GCD(72, 90) = 18
        5220 + 783 = 6003
        The greatest common divisor is 15

    Rejects:
        answers copied from the problem without derivational support,
        such as "Time saved = 64" when 64 is directly given in the problem.
    """

    if not candidate or not candidate_answer:
        return False
    target = normalize_numeric_answer_for_support(candidate_answer)
    if not target:
        return False
    text = candidate.replace("×", "*").replace("÷", "/")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace(",", "")
    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)
    number = r"[-+]?\d+(?:\.\d+)?"
    # 1. Standard arithmetic equation:
    #    276 / 12 = 23
    #    5220 + 783 = 6003
    arithmetic_equation_pattern = (
        rf"({number}(?:\s*[\+\-\*/]\s*{number})+)"
        rf"\s*=\s*"
        rf"({number})"
    )
    for match in re.finditer(arithmetic_equation_pattern, text):
        rhs = normalize_numeric_answer_for_support(match.group(2))
        if rhs == target:
            return True

    # 2. LCM/GCD equation:
    #    LCM(6, 5) = 30
    #    GCD(72,90) = 18
    function_equation_pattern = (
        rf"\b(?:LCM|GCD|lcm|gcd)\s*\([^\)]*\)"
        rf"\s*=\s*"
        rf"({number})"
    )
    for match in re.finditer(function_equation_pattern, text):
        rhs = normalize_numeric_answer_for_support(match.group(1))
        if rhs == target:
            return True

    # 3. Explicit result statement from a derivational keyword:
    #    Number of trays = 23.
    #    The greatest common divisor is 15.
    #    The smallest common multiple is 35.
    #
    # This is intentionally limited to derivational phrases, not arbitrary
    # "time saved = 64" statements, because the latter can just copy a problem number.
    derivational_statement_patterns = [
        rf"(?:number of trays|number of packages|number of groups|number of rows)\s*=\s*({number})",
        rf"(?:greatest common divisor|least common multiple|smallest common multiple)\s+(?:is|=)\s+({number})",
        rf"(?:gcd|lcm)\s+(?:is|=)\s+({number})",
    ]

    for pattern in derivational_statement_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            rhs = normalize_numeric_answer_for_support(match.group(1))
            if rhs == target:
                return True

    # 4. Final standalone numeric derivation step:
    #    Step 4: 35.
    #
    # Allow this only when the previous text explicitly mentions LCM/GCD/common multiple/divisor.
    # This catches LCM/GCD problems without allowing arbitrary copied numbers.
    has_number_theory_context = bool(
        re.search(
            r"\b(?:LCM|GCD|least common multiple|greatest common divisor|common multiple|common divisor)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    if has_number_theory_context:
        standalone_step_pattern = rf"Step\s+\d+\s*:\s*({number})\s*[\.\n ]"
        for match in re.finditer(standalone_step_pattern, text, flags=re.IGNORECASE):
            rhs = normalize_numeric_answer_for_support(match.group(1))
            if rhs == target:
                return True

    return False

def llm_accept_repair(
    initial_reasoning: str,
    candidate: str,
    initial_meta: Dict[str, Any],
    candidate_meta: Dict[str, Any],
    initial_graph: Dict[str, Any],
    candidate_graph: Dict[str, Any],
) -> Dict[str, Any]:
    initial_answer = safe_final_answer(initial_reasoning)
    candidate_answer = safe_final_answer(candidate)
    initial_score = float(initial_meta.get("global_consistency_score", 0.0))
    candidate_score = float(candidate_meta.get("global_consistency_score", 0.0))
    initial_meta_error = str(initial_meta.get("error_type", "none"))
    candidate_meta_error = str(candidate_meta.get("error_type", "none"))
    candidate_consistent = bool(candidate_meta.get("is_consistent", False))
    initial_graph_score = float(initial_graph.get("semantic_graph_score", 0.0))
    candidate_graph_score = float(candidate_graph.get("semantic_graph_score", 0.0))
    initial_graph_error = str(initial_graph.get("error_type", "none"))
    candidate_graph_error = str(candidate_graph.get("error_type", "none"))
    candidate_has_semantic_pattern_issue = bool(
        candidate_graph.get("semantic_pattern_issues")
    )
    candidate_has_format_issue = bool(candidate_graph.get("format_issues"))
    candidate_has_binding_issue = bool(candidate_graph.get("binding_issues"))
    candidate_graph_clean = bool(
        candidate_graph_error == "none"
        and not candidate_has_semantic_pattern_issue
        and not candidate_has_format_issue
        and not candidate_has_binding_issue
    )
    candidate_graph_benign_warning = bool(
        candidate_graph_error == "comparison_warning"
        and not candidate_has_semantic_pattern_issue
        and not candidate_has_format_issue
        and not candidate_has_binding_issue
    )
    high_risk_initial_graph = bool(
        initial_graph.get("semantic_pattern_issues")
        or initial_graph.get("format_issues")
        or initial_graph.get("error_type") == "generation_failure"
    )
    severe_candidate_meta_error = candidate_meta_error in {
        "arithmetic_error",
        "logical_contradiction",
        "generation_failure",
    }
    answer_changed = bool(
        candidate_answer
        and answers_differ(initial_answer, candidate_answer)
    )
    initial_has_explicit_error_signal = bool(
        initial_meta_error in {
            "arithmetic_error",
            "missing_constraint",
            "logical_contradiction",
            "generation_failure",
        }
        or initial_graph_error not in {"none", ""}
        or initial_graph_score < 0.80
    )
    initial_has_explicit_high_risk_signal = bool(
        initial_graph.get("semantic_pattern_issues")
        or initial_graph.get("format_issues")
        or initial_graph.get("error_type") == "generation_failure"
    )
    initial_has_quantity_binding_error = bool(
        "quantity_binding_error" in initial_graph_error
        or bool(initial_graph.get("binding_issues"))
    )
    candidate_answer_supported = candidate_answer_is_supported_by_equation(
        candidate,
        candidate_answer,
    )
    candidate_support_ok = bool(
        candidate_answer_supported or DISABLE_EQUATION_SUPPORT_GUARD
    )
    common_payload = {
        "initial_answer": initial_answer,
        "candidate_answer": candidate_answer,
        "initial_score": initial_score,
        "candidate_score": candidate_score,
        "initial_meta_error_type": initial_meta_error,
        "candidate_meta_error_type": candidate_meta_error,
        "initial_graph_score": initial_graph_score,
        "candidate_graph_score": candidate_graph_score,
        "initial_graph_error_type": initial_graph_error,
        "candidate_graph_error_type": candidate_graph_error,
        "candidate_graph_clean": candidate_graph_clean,
        "candidate_graph_benign_warning": candidate_graph_benign_warning,
        "candidate_meta_consistent": candidate_consistent,
        "initial_has_explicit_error_signal": initial_has_explicit_error_signal,
        "initial_has_explicit_high_risk_signal": initial_has_explicit_high_risk_signal,
        "initial_has_quantity_binding_error": initial_has_quantity_binding_error,
        "candidate_answer_supported_by_equation": candidate_answer_supported,
        "candidate_support_ok": candidate_support_ok,
        "equation_support_guard_disabled": DISABLE_EQUATION_SUPPORT_GUARD,
        "relax_missing_constraint_accept": RELAX_MISSING_CONSTRAINT_ACCEPT,
    }
    # 1. Reject no-op repairs.
    if (
        initial_answer
        and candidate_answer
        and not answers_differ(initial_answer, candidate_answer)
    ):
        return {
            "accept": False,
            "reason": "Rejected LLM repair because final answer did not change.",
            **common_payload,
        }
    # 2. High-risk semantic repair path.
    if (
        answer_changed
        and candidate_score >= 0.75
        and high_risk_initial_graph
        and not severe_candidate_meta_error
    ):
        return {
            "accept": True,
            "reason": (
                "Accepted LLM repair because answer changed under explicit high-risk "
                "semantic diagnosis and candidate has high meta score."
            ),
            **common_payload,
        }
    # 3. Empty-generation rescue path.
    if (
        not initial_answer
        and candidate_answer
        and candidate_score >= 0.70
        and not severe_candidate_meta_error
    ):
        return {
            "accept": True,
            "reason": (
                "Accepted LLM repair because initial generation was empty "
                "and candidate has parseable answer with acceptable meta score."
            ),
            **common_payload,
        }

    # 4. Tightened very-low-confidence rescue path.
    if (
        initial_answer
        and candidate_answer
        and answer_changed
        and initial_score <= 0.30
        and candidate_score >= 0.95
        and candidate_consistent
        and initial_has_explicit_error_signal
        and not severe_candidate_meta_error
    ):
        return {
            "accept": True,
            "reason": (
                "Accepted LLM repair because initial reasoning has extremely low "
                "meta score with explicit diagnostic evidence and candidate is highly consistent."
            ),
            **common_payload,
        }

    # 5. Gate-audit rule:
    #    low initial confidence + clean or benign-warning candidate.
    clean_low_confidence_candidate = bool(
        candidate_graph_clean
        and candidate_graph_score >= 0.95
        and candidate_score >= 0.80
    )
    benign_warning_low_confidence_candidate = bool(
        candidate_graph_benign_warning
        and candidate_graph_score >= 0.70
        and candidate_score >= 0.90
    )
    if (
        answer_changed
        and initial_answer
        and initial_score <= 0.30
        and initial_has_explicit_error_signal
        and candidate_consistent
        and candidate_meta_error == "none"
        and (
            clean_low_confidence_candidate
            or benign_warning_low_confidence_candidate
        )
    ):
        return {
            "accept": True,
            "reason": (
                "Accepted LLM repair because initial reasoning has low confidence with "
                "explicit diagnostic evidence, and candidate is answer-changing, "
                "meta-consistent, and either graph-clean or has only a benign semantic graph warning."
            ),
            **common_payload,
        }

    # 6. Restricted clean semantic-improvement path.
    if (
        ENABLE_CLEAN_SEMANTIC_IMPROVEMENT_ACCEPT
        and initial_answer
        and candidate_answer
        and answer_changed
        and initial_has_explicit_high_risk_signal
        and candidate_score >= 0.85
        and candidate_graph_clean
        and candidate_graph_score >= 0.95
        and candidate_graph_score >= initial_graph_score + 0.50
        and candidate_consistent
        and not severe_candidate_meta_error
    ):
        return {
            "accept": True,
            "reason": (
                "Accepted LLM repair because candidate has clean semantic graph, "
                "high meta score, substantially improves the initial semantic graph, "
                "and the initial reasoning has explicit high-risk diagnostic evidence."
            ),
            **common_payload,
        }

    # 7. General relaxed support acceptance path.
    #
    # Disabled by default. Intended for strong-reasoner GSM8K relaxed-support
    # experiments. This accepts only meta-clean, graph-clean, answer-changing
    # candidates whose final answer is explicitly supported by an equation.
    if (
        ENABLE_RELAXED_SUPPORT_ACCEPT
        and initial_answer
        and candidate_answer
        and answer_changed
        and candidate_meta_error == "none"
        and candidate_consistent
        and candidate_score >= 0.70
        and candidate_graph_clean
        and candidate_graph_score >= 0.95
        and candidate_support_ok
        and not severe_candidate_meta_error
    ):
        return {
            "accept": True,
            "reason": (
                "Accepted LLM repair under relaxed support gate because the "
                "candidate is answer-changing, meta-clean, graph-clean, and its "
                "final answer is explicitly supported by an arithmetic equation."
            ),
            **common_payload,
        }
    
    # Ablation only: relaxed missing-constraint acceptance path.
    #
    # This is intentionally disabled by default. It is used to test whether
    # accepting missing-constraint candidates increases repair recall at the
    # cost of safety.
    if (
        RELAX_MISSING_CONSTRAINT_ACCEPT
        and ENABLE_RELAXED_SUPPORT_ACCEPT
        and initial_answer
        and candidate_answer
        and answer_changed
        and candidate_meta_error == "missing_constraint"
        and candidate_score >= 0.70
        and candidate_graph_clean
        and candidate_graph_score >= 0.95
        and candidate_support_ok
        and not severe_candidate_meta_error
    ):
        return {
            "accept": True,
            "reason": (
                "Accepted LLM repair under ablation-only relaxed missing-constraint "
                "gate because the candidate is answer-changing, graph-clean, and "
                "passes the current support condition."
            ),
            **common_payload,
        }

    # 8. General relaxed low-symbolic-coverage support acceptance path.
    #
    # Disabled by default. Intended for GSM8K relaxed-support experiments.
    # This accepts only low_symbolic_coverage as a mild meta warning.
    #
    # Important: do NOT accept missing_constraint here. In GSM8K, missing_constraint
    # can correspond to a real omitted quantity or condition.
    if (
        ENABLE_RELAXED_SUPPORT_ACCEPT
        and initial_answer
        and candidate_answer
        and answer_changed
        and candidate_meta_error == "low_symbolic_coverage"
        and candidate_score >= 0.70
        and candidate_graph_clean
        and candidate_graph_score >= 0.95
        and candidate_support_ok
        and not severe_candidate_meta_error
    ):
        return {
            "accept": True,
            "reason": (
                "Accepted LLM repair under relaxed low-symbolic-coverage support gate "
                "because the candidate is answer-changing, graph-clean, has only low "
                "symbolic coverage, and its final answer is explicitly supported by "
                "an equation."
            ),
            **common_payload,
        }

    # 9. Weak-reasoner relaxed acceptance path.
    #
    # Disabled by default. Use only for weak-initial-reasoner experiments.
    # This path accepts strong clean candidates when the initial reasoner is weak.
    if (
        ENABLE_WEAK_REASONER_RELAXED_ACCEPT
        and initial_answer
        and candidate_answer
        and answer_changed
        and candidate_meta_error == "none"
        and candidate_consistent
        and candidate_score >= 0.70
        and candidate_graph_clean
        and candidate_graph_score >= 0.95
        and not severe_candidate_meta_error
    ):
        return {
            "accept": True,
            "reason": (
                "Accepted LLM repair under weak-reasoner relaxed gate because "
                "the candidate is answer-changing, meta-clean, graph-clean, "
                "and sufficiently consistent."
            ),
            **common_payload,
        }

    # 10. Weak-reasoner benign-meta relaxed acceptance path.
    #
    # Disabled by default. Use only for weak-initial-reasoner experiments.
    # This keeps missing_constraint enabled for the ASDiv weak-reasoner setting,
    # where the support guard previously removed measured broken cases.
    if (
        ENABLE_WEAK_REASONER_RELAXED_ACCEPT
        and initial_answer
        and candidate_answer
        and answer_changed
        and candidate_meta_error in {
            "low_symbolic_coverage",
            "missing_constraint",
        }
        and candidate_score >= 0.58
        and candidate_graph_clean
        and candidate_graph_score >= 0.95
        and candidate_answer_supported
        and not severe_candidate_meta_error
    ):
        return {
            "accept": True,
            "reason": (
                "Accepted LLM repair under weak-reasoner benign-meta relaxed gate "
                "because the candidate is answer-changing, graph-clean, has only "
                "a mild meta-diagnostic warning, and its final answer is explicitly "
                "supported by an arithmetic equation."
            ),
            **common_payload,
        }

    # 11. Default rejection.
    return {
        "accept": False,
        "reason": (
            "Rejected LLM repair because candidate does not satisfy high-risk, "
            "empty-generation, tightened very-low-confidence, low-initial-confidence "
            "clean/benign-warning, restricted clean semantic-improvement, relaxed "
            "support, relaxed low-symbolic-coverage support, weak-reasoner clean "
            "relaxed, or weak-reasoner benign-meta relaxed policy."
        ),
        **common_payload,
    }

def try_candidate_repair(
    problem: str,
    initial_reasoning: str,
    initial_meta: Dict[str, Any],
    initial_graph: Dict[str, Any],
    candidate: str,
    source: str,
) -> Dict[str, Any]:
    if not is_valid_repair_text(candidate):
        return {
            "accepted": False,
            "source": source,
            "candidate": candidate or "",
            "candidate_diagnostics": None,
            "acceptance_decision": {
                "accept": False,
                "reason": "Rejected because candidate repair text is empty, too short, missing Final Answer, or has no parseable answer.",
            },
            "graph_guard_decision": None,
        }
    if source.startswith("llm"):
        clean_check = is_clean_llm_repair(candidate)
        if not clean_check["clean"]:
            return {
                "accepted": False,
                "source": source,
                "candidate": candidate or "",
                "candidate_diagnostics": None,
                "acceptance_decision": {
                    "accept": False,
                    "reason": clean_check["reason"],
                },
                "graph_guard_decision": None,
            }
    candidate_diagnostics = run_diagnostics(problem, candidate)
    candidate_meta = candidate_diagnostics["meta_diagnosis"]
    candidate_graph = candidate_diagnostics["semantic_graph"]
    if source == "deterministic":
        acceptance_decision = deterministic_accept_repair(
            initial_reasoning=initial_reasoning,
            candidate=candidate,
            initial_meta=initial_meta,
            candidate_meta=candidate_meta,
            initial_graph=initial_graph,
            candidate_graph=candidate_graph,
        )
    else:
        acceptance_decision = llm_accept_repair(
            initial_reasoning=initial_reasoning,
            candidate=candidate,
            initial_meta=initial_meta,
            candidate_meta=candidate_meta,
            initial_graph=initial_graph,
            candidate_graph=candidate_graph,
        )

    if ENABLE_GRAPH_GUARD:
        graph_guard_decision = graph_guard_accept(
            initial_graph=initial_graph,
            candidate_graph=candidate_graph,
        )
    else:
        graph_guard_decision = {
            "accept": True,
            "reason": "Graph guard disabled.",
        }
    accepted = bool(
        acceptance_decision.get("accept", False)
        and graph_guard_decision.get("accept", False)
    )

    return {
        "accepted": accepted,
        "source": source,
        "candidate": candidate,
        "candidate_diagnostics": candidate_diagnostics,
        "acceptance_decision": acceptance_decision,
        "graph_guard_decision": graph_guard_decision,
    }

def apply_repair_result(
    repair_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "repair_candidate": repair_result.get("candidate", ""),
        "candidate_diagnostics": repair_result.get("candidate_diagnostics"),
        "acceptance_decision": repair_result.get("acceptance_decision"),
        "graph_guard_decision": repair_result.get("graph_guard_decision"),
        "repair_source": repair_result.get("source", ""),
        "repair_accepted": bool(repair_result.get("accepted", False)),
    }



def candidate_result_rank(repair_result: Dict[str, Any]) -> Tuple[int, float, float, int]:
    """
    Rank candidate attempts for logging when no candidate is accepted.

    Accepted attempts are always best. For rejected attempts, prefer valid candidates
    with higher meta score, higher semantic graph score, and parseable final answer.
    This ranking does NOT apply a rejected repair; it only selects the most informative
    rejected candidate to store in the top-level fields.
    """
    if not repair_result:
        return (-1, 0.0, 0.0, 0)

    accepted = 1 if repair_result.get("accepted", False) else 0
    diagnostics = repair_result.get("candidate_diagnostics") or {}
    meta = diagnostics.get("meta_diagnosis") or {}
    graph = diagnostics.get("semantic_graph") or {}

    try:
        meta_score = float(meta.get("global_consistency_score", 0.0))
    except Exception:
        meta_score = 0.0

    try:
        graph_score = float(graph.get("semantic_graph_score", 0.0))
    except Exception:
        graph_score = 0.0

    has_answer = 1 if safe_final_answer(repair_result.get("candidate", "")) else 0
    return (accepted, meta_score, graph_score, has_answer)


def process_sample(sample: Dict[str, Any], cache_item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    sample_id = sample.get("id")
    problem = sample.get("problem", "")
    gold_answer = str(sample.get("answer", sample.get("gold_answer", ""))).strip()

    if cache_item is None:
        initial_reasoning = ""
        initial_diagnostics = run_diagnostics(problem, initial_reasoning)
        initial_diagnostics["meta_diagnosis"]["explanation"] = "Initial reasoning missing from cache."
    else:
        initial_reasoning = (
            cache_item.get("reasoning", "")
            or cache_item.get("initial_reasoning", "")
            or ""
        )
        initial_diagnostics = run_diagnostics(problem, initial_reasoning)

    initial_checker_results = initial_diagnostics["checker_results"]
    initial_constraint_results = initial_diagnostics["constraint_results"]
    initial_meta = initial_diagnostics["meta_diagnosis"]
    initial_graph = initial_diagnostics["semantic_graph"]
    deterministic_hint = ""
    repair_candidate = ""
    repair_failed = False
    repair_source = ""
    candidate_diagnostics = None
    acceptance_decision = None
    graph_guard_decision = None
    llm_repair_metadata: Dict[str, Any] = {}
    repair_attempts: List[Dict[str, Any]] = []
    final_reasoning = initial_reasoning
    was_repaired = False
    repair_accepted = False
    if should_attempt_repair(initial_meta, initial_graph, initial_reasoning):
        was_repaired = True

        deterministic_hint = run_deterministic_hint(
            problem=problem,
            initial_reasoning=initial_reasoning,
            meta_diagnosis=initial_meta,
            semantic_graph=initial_graph,
        )

        if ENABLE_DIRECT_DETERMINISTIC_ACCEPT and deterministic_hint:
            deterministic_result = try_candidate_repair(
                problem=problem,
                initial_reasoning=initial_reasoning,
                initial_meta=initial_meta,
                initial_graph=initial_graph,
                candidate=deterministic_hint,
                source="deterministic",
            )

            if deterministic_result["accepted"]:
                applied = apply_repair_result(deterministic_result)

                repair_candidate = applied["repair_candidate"]
                candidate_diagnostics = applied["candidate_diagnostics"]
                acceptance_decision = applied["acceptance_decision"]
                graph_guard_decision = applied["graph_guard_decision"]
                repair_source = applied["repair_source"]
                repair_accepted = applied["repair_accepted"]
                final_reasoning = repair_candidate

        if not repair_accepted and ENABLE_LLM_REPAIR:
            try:
                selected_llm_result = None
                selected_llm_metadata: Dict[str, Any] = {}

                for attempt_idx in range(LLM_REPAIR_NUM_CANDIDATES):
                    llm_candidate, attempt_metadata = call_llm_repair(
                        problem=problem,
                        initial_reasoning=initial_reasoning,
                        meta_diagnosis=initial_meta,
                        semantic_graph=initial_graph,
                        deterministic_hint=deterministic_hint,
                        attempt_idx=attempt_idx,
                    )

                    llm_result = try_candidate_repair(
                        problem=problem,
                        initial_reasoning=initial_reasoning,
                        initial_meta=initial_meta,
                        initial_graph=initial_graph,
                        candidate=llm_candidate,
                        source=f"llm_hint_guided_attempt_{attempt_idx}",
                    )

                    attempt_record = {
                        "attempt_idx": attempt_idx,
                        "candidate": llm_result.get("candidate", ""),
                        "accepted": bool(llm_result.get("accepted", False)),
                        "source": llm_result.get("source", ""),
                        "metadata": attempt_metadata,
                        "acceptance_decision": llm_result.get("acceptance_decision"),
                        "graph_guard_decision": llm_result.get("graph_guard_decision"),
                        "candidate_meta_diagnosis": (
                            (llm_result.get("candidate_diagnostics") or {}).get("meta_diagnosis")
                        ),
                        "candidate_semantic_graph": (
                            (llm_result.get("candidate_diagnostics") or {}).get("semantic_graph")
                        ),
                    }
                    repair_attempts.append(attempt_record)

                    if selected_llm_result is None or (
                        candidate_result_rank(llm_result) > candidate_result_rank(selected_llm_result)
                    ):
                        selected_llm_result = llm_result
                        selected_llm_metadata = attempt_metadata

                    if llm_result.get("accepted", False):
                        selected_llm_result = llm_result
                        selected_llm_metadata = attempt_metadata
                        break

                if selected_llm_result is not None:
                    llm_repair_metadata = {
                        **selected_llm_metadata,
                        "num_candidates_requested": LLM_REPAIR_NUM_CANDIDATES,
                        "num_candidates_generated": len(repair_attempts),
                        "selected_attempt_idx": (
                            selected_llm_metadata.get("attempt_idx")
                            if isinstance(selected_llm_metadata, dict)
                            else None
                        ),
                    }

                    repair_candidate = selected_llm_result["candidate"]
                    candidate_diagnostics = selected_llm_result["candidate_diagnostics"]
                    acceptance_decision = selected_llm_result["acceptance_decision"]
                    graph_guard_decision = selected_llm_result["graph_guard_decision"]
                    repair_source = selected_llm_result["source"]

                    if selected_llm_result.get("accepted", False):
                        final_reasoning = repair_candidate
                        repair_accepted = True

            except Exception as e:
                repair_failed = True
                print(f"[Repair failed] {e}")

    result = {
        "id": sample_id,
        "problem": problem,
        "gold_answer": gold_answer,

        "initial_reasoning": initial_reasoning,
        "initial_checker_results": initial_checker_results,
        "initial_constraint_results": initial_constraint_results,
        "initial_meta_diagnosis": initial_meta,
        "initial_semantic_graph": initial_graph,

        "deterministic_hint": deterministic_hint,

        "repair_candidate": repair_candidate,
        "repair_failed": repair_failed,
        "repair_source": repair_source,
        "llm_repair_metadata": llm_repair_metadata,
        "repair_attempts": repair_attempts,
        "num_repair_attempts": len(repair_attempts),

        "candidate_checker_results": (
            candidate_diagnostics["checker_results"] if candidate_diagnostics else None
        ),
        "candidate_constraint_results": (
            candidate_diagnostics["constraint_results"] if candidate_diagnostics else None
        ),
        "candidate_meta_diagnosis": (
            candidate_diagnostics["meta_diagnosis"] if candidate_diagnostics else None
        ),
        "candidate_semantic_graph": (
            candidate_diagnostics["semantic_graph"] if candidate_diagnostics else None
        ),

        "acceptance_decision": acceptance_decision,
        "graph_guard_decision": graph_guard_decision,

        "final_reasoning": final_reasoning,
        "was_repaired": was_repaired,
        "repair_accepted": repair_accepted,
    }

    result["evaluation"] = evaluate_result(result)
    return result


def main() -> None:
    ensure_output_dir()
    dataset = load_local_jsonl_dataset(LOCAL_DATASET_PATH)
    cache = load_initial_cache(INITIAL_CACHE_PATH)
    results: List[Dict[str, Any]] = []

    with open(RAW_RESULTS_PATH, "w", encoding="utf-8") as raw_f:
        for sample in tqdm(dataset, desc="Running guarded pipeline from cache"):
            sample_id = str(sample.get("id"))
            cache_item = cache.get(sample_id)

            result = process_sample(sample, cache_item)
            results.append(result)

            raw_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            raw_f.flush()

    report = summarize(results)
    format_retry_used = sum(
        1 for r in results
        if (r.get("llm_repair_metadata") or {}).get("format_retry_used", False)
    )
    total_llm_repair_attempts = sum(
        int(r.get("num_repair_attempts", 0) or 0)
        for r in results
    )

    report.update(
        {
            "cache_path": INITIAL_CACHE_PATH,
            "from_cache": True,
            "guarded_pipeline": True,
            "medium_repair_trigger": MEDIUM_REPAIR_TRIGGER,
            "graph_guard_enabled": ENABLE_GRAPH_GUARD,
            "deterministic_hint_enabled": ENABLE_DETERMINISTIC_HINT,
            "direct_deterministic_accept_enabled": ENABLE_DIRECT_DETERMINISTIC_ACCEPT,
            "llm_repair_enabled": ENABLE_LLM_REPAIR,
            "repair_backend": REPAIR_BACKEND,
            "repair_model": REPAIR_MODEL_NAME,
            "llm_repair_max_retries": LLM_REPAIR_MAX_RETRIES,
            "llm_repair_max_tokens": LLM_REPAIR_MAX_TOKENS,
            "json_repair_output_enabled": ENABLE_JSON_REPAIR_OUTPUT,
            "format_retry_enabled": ENABLE_FORMAT_RETRY,
            "format_retry_used": format_retry_used,
            "format_retry_max_retries": LLM_FORMAT_RETRY_MAX_RETRIES,
            "format_retry_max_tokens": LLM_FORMAT_RETRY_MAX_TOKENS,
            "llm_repair_num_candidates": LLM_REPAIR_NUM_CANDIDATES,
            "total_llm_repair_attempts": total_llm_repair_attempts,
            "clean_semantic_improvement_accept_enabled": ENABLE_CLEAN_SEMANTIC_IMPROVEMENT_ACCEPT,
            "graph_trigger_threshold": GRAPH_TRIGGER_THRESHOLD,
            "graph_accept_min_score": GRAPH_ACCEPT_MIN_SCORE,
            "repair_mode": "medium_trigger_json_hint_guided_best_of_n_llm_repair_with_generic_format_retry",
            "llm_clean_output_guard": True,
            "llm_accept_policy": (
                "answer_change_under_high_risk_or_empty_or_very_low_confidence_"
                "or_low_initial_confidence_clean_or_benign_warning_"
                "or_restricted_clean_semantic_improvement"
            ),
        }
    )

    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    error_results = [
        r for r in results
        if not r.get("evaluation", {}).get("final_correct", False)
    ]
    with open(ERROR_SET_PATH, "w", encoding="utf-8") as f:
        for r in error_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Guarded pipeline from cache finished.")
    print(report)
    print(f"Saved raw results to: {RAW_RESULTS_PATH}")
    print(f"Saved final report to: {FINAL_REPORT_PATH}")
    print(f"Saved error set to: {ERROR_SET_PATH}")

if __name__ == "__main__":
    main()
