import json
import urllib.request
from typing import Optional

def call_ollama(
    prompt: str,
    model: str = "qwen2.5:1.5b",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    Call a local Ollama model.
    Requires: ollama serve
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "").strip()
    except Exception as e:
        return f"GENERATION_ERROR: {e}"