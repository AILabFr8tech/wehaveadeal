"""Thin wrapper around Ollama's `/api/chat` endpoint with structured
(JSON-schema-constrained) output. Deliberately isolated in one small
function so `wehaveadeal/tests/test_llm_engine.py` can monkeypatch `call_ollama`
directly instead of needing a real Ollama server -- there is none available
in this development environment; the real models only run on the remote
server this was built for (see the paper).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional, Union

import requests

DEFAULT_HOST = "http://localhost:11434"


@dataclass
class LLMResponse:
    raw_text: str
    parsed: Optional[dict]
    latency_s: float
    error: Optional[str] = None


def call_ollama(
    model: str,
    messages: list[dict],
    schema: dict,
    host: str = DEFAULT_HOST,
    timeout_s: float = 90.0,
    think: Optional[Union[bool, str]] = None,
) -> LLMResponse:
    """One structured-output call. Never raises for an ordinary failure
    (timeout, connection error, invalid JSON) -- those come back as
    `LLMResponse.error` so the caller (the LLM engine's retry loop) can
    decide what to do, the same defense-in-depth every other engine in this
    package already has around its own decision logic."""
    payload: dict = {"model": model, "messages": messages, "format": schema, "stream": False}
    if think is not None:
        payload["think"] = think

    start = time.monotonic()
    try:
        resp = requests.post(f"{host.rstrip('/')}/api/chat", json=payload, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return LLMResponse(raw_text="", parsed=None, latency_s=time.monotonic() - start, error=f"request_error: {exc}")
    except ValueError as exc:  # resp.json() failed to parse
        return LLMResponse(raw_text=resp.text, parsed=None, latency_s=time.monotonic() - start, error=f"bad_response_json: {exc}")

    latency = time.monotonic() - start
    content = data.get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return LLMResponse(raw_text=content, parsed=None, latency_s=latency, error="not_a_json_object")
    except (json.JSONDecodeError, TypeError) as exc:
        return LLMResponse(raw_text=content, parsed=None, latency_s=latency, error=f"invalid_json: {exc}")
    return LLMResponse(raw_text=content, parsed=parsed, latency_s=latency, error=None)
