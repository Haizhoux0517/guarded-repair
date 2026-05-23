from typing import Any, Dict, List, Optional
import json
import time
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    timeout=90.0,
)

def _safe_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert OpenAI SDK objects to dict for debugging / robust parsing.
    """

    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    try:
        return json.loads(str(obj))
    except Exception:
        return {"raw": str(obj)}

def _extract_message_text(response: Any) -> str:
    """
    Extract useful text from OpenAI-compatible chat completion response.

    Handles:
    - response.choices[0].message.content
    - response.choices[0].message.reasoning_content
    - dict-style equivalents
    """

    if response is None:
        return ""
    # 1. SDK object path
    try:
        choices = getattr(response, "choices", None)
        if choices:
            message = choices[0].message
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
            # Some reasoning models expose this field.
            reasoning_content = getattr(message, "reasoning_content", None)
            if isinstance(reasoning_content, str) and reasoning_content.strip():
                return reasoning_content.strip()
            # Fallback: check message as dict.
            message_dict = _safe_to_dict(message)
            for key in ["content", "reasoning_content", "text"]:
                value = message_dict.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            # Very rare fallback: tool calls as text.
            tool_calls = message_dict.get("tool_calls")
            if tool_calls:
                return json.dumps(tool_calls, ensure_ascii=False)

    except Exception:
        pass

    # 2. Dict path
    data = _safe_to_dict(response)
    try:
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})

            for key in ["content", "reasoning_content", "text"]:
                value = msg.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

            if "text" in choices[0] and isinstance(choices[0]["text"], str):
                if choices[0]["text"].strip():
                    return choices[0]["text"].strip()

            tool_calls = msg.get("tool_calls")
            if tool_calls:
                return json.dumps(tool_calls, ensure_ascii=False)
    except Exception:
        pass

    return ""


def _compact_response_debug(response: Any, max_chars: int = 1000) -> str:
    """
    Return a compact string representation of raw response for debugging.
    """

    data = _safe_to_dict(response)

    try:
        text = json.dumps(data, ensure_ascii=False)
    except Exception:
        text = str(data)

    if len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"

    return text


def call_deepseek(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_retries: int = 3,
    timeout: Optional[float] = None,
) -> str:
    """
    Call DeepSeek / OpenAI-compatible chat completion API.

    Compatible with your existing runner:
        call_deepseek(model, messages, temperature, max_tokens, max_retries)

    Notes:
    - If message.content is empty, this function tries reasoning_content.
    - If still empty, it includes compact raw response in the error.
    """

    if not model:
        raise ValueError("Model name is empty.")

    if not messages:
        raise ValueError("Messages are empty.")

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Some OpenAI-compatible SDK versions accept timeout per request.
            if timeout is not None:
                kwargs["timeout"] = timeout

            response = client.chat.completions.create(**kwargs)

            content = _extract_message_text(response)

            if content:
                return content

            raw_debug = _compact_response_debug(response)
            last_error = f"Empty model response. Raw response: {raw_debug}"

        except Exception as e:
            last_error = str(e)

        if attempt < max_retries:
            time.sleep(1.5 * attempt)

    raise RuntimeError(
        f"DeepSeek call failed after {max_retries} retries. Last error: {last_error}"
    )