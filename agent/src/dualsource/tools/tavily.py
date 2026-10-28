from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class TavilyResult:
    status: str  # ok | error | skipped
    text: str
    sources: list[dict[str, str]]
    raw: dict[str, Any] | None = None


def _env_search_depth() -> str:
    # Keep advanced first (accuracy). Set TAVILY_SEARCH_DEPTH=basic later for speed.
    depth = (os.getenv("TAVILY_SEARCH_DEPTH") or "advanced").strip().lower()
    return depth if depth in {"basic", "advanced"} else "advanced"


def _env_max_results(default: int = 5) -> int:
    raw = (os.getenv("TAVILY_MAX_RESULTS") or str(default)).strip()
    try:
        return max(1, min(int(raw), 8))
    except Exception:
        return default


class TavilyClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY") or ""

    def search(self, query: str, max_results: int | None = None) -> TavilyResult:
        if not self.api_key:
            return TavilyResult(status="error", text="TAVILY_API_KEY missing", sources=[])
        n = int(max_results) if max_results is not None else _env_max_results(5)
        n = max(1, min(n, 8))
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": _env_search_depth(),
                    "include_answer": True,
                    "max_results": n,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = (data.get("answer") or "").strip()
            results = data.get("results") or []
            sources: list[dict[str, str]] = []
            snippets: list[str] = []
            for item in results[:n]:
                title = str(item.get("title") or "public source")
                url = str(item.get("url") or "")
                content = str(item.get("content") or "").strip()
                sources.append({"label": title, "url": url})
                if content:
                    snippets.append(f"- {title}: {content[:400]}")
            text_parts = []
            if answer:
                text_parts.append(answer)
            if snippets:
                text_parts.append("Source notes:\n" + "\n".join(snippets))
            text = "\n\n".join(text_parts).strip() or "No public web result found."
            return TavilyResult(status="ok", text=text, sources=sources, raw=data)
        except Exception as exc:  # noqa: BLE001
            return TavilyResult(status="error", text=f"Tavily error: {exc}", sources=[])
