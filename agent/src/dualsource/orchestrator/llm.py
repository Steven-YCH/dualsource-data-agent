"""LLM call utilities for DualSource Agent orchestrator.

Uses Databricks Foundation Model APIs via the serving endpoint invocation URL.
Two models:
- ROUTER_MODEL: fast model for guard-rail + routing (~1-3s)
- SYNTH_MODEL: primary model for answer synthesis (~2-4s)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from dualsource.tools.auth import auth_headers, workspace_host

ROUTER_MODEL = os.getenv("CI_ROUTER_MODEL", "databricks-meta-llama-3-3-70b-instruct")
SYNTH_MODEL = os.getenv("CI_LLM_ENDPOINT", "databricks-gpt-oss-120b")


def llm_call(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 1024,
    temperature: float = 0,
    timeout: float = 45,
) -> str:
    """Call a Databricks Foundation Model API and return the text content."""
    host = workspace_host()
    headers = auth_headers()
    resp = requests.post(
        f"{host}/serving-endpoints/{model}/invocations",
        headers=headers,
        json={
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"LLM returned no choices: {data}")
    content = choices[0].get("message", {}).get("content", "")

    # gpt-oss-120b returns extended-thinking as a list of blocks.
    # The content may be a native Python list (from SDK) or a JSON string.
    parts = None
    if isinstance(content, list):
        parts = content
    elif isinstance(content, str) and content.strip().startswith("["):
        try:
            parts = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass

    if parts and isinstance(parts, list):
        text_parts = []
        for part in parts:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                # Skip reasoning blocks entirely
        if text_parts:
            return "\n".join(text_parts).strip()

    return str(content).strip()


def llm_json(
    model: str,
    system: str,
    user_content: str,
    *,
    max_tokens: int = 512,
    timeout: float = 30,
) -> dict[str, Any]:
    """Call LLM and parse response as JSON."""
    raw = llm_call(
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user_content}],
        max_tokens=max_tokens,
        timeout=timeout,
    )
    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)
