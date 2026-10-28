"""Progress stage + tool-card helpers aligned with PROGRESS_EVENTS_CONTRACT.md."""

from __future__ import annotations

import time
from typing import Any

STAGE_LABELS = {
    "route": "Understanding question",
    "genie": "Querying internal data",
    "sql": "Querying internal data",
    "web": "Checking online sources",
    "synthesize": "Drafting answer",
    "files": "Preparing export",
    "complete": "Complete",
}

# User-safe labels (no Wren/Genie/Tavily/MCP jargon).
TOOL_LABELS = {
    "wren_get_instructions": "Loading Internal metric rules",
    "wren_describe_schema": "Checking Internal data shape",
    "wren_dry_run": "Validating Internal query",
    "wren_run_sql": "Querying internal data",
    "web_search": "Checking online sources",
    "chart_plan": "Preparing chart",
    "genie_ask": "Querying internal data",
}


def stage_event(stage: str, status: str, label: str | None = None) -> dict[str, Any]:
    return {
        "type": "stage",
        "stage": stage,
        "status": status,
        "label": label or STAGE_LABELS.get(stage, stage),
        "ts": time.time(),
    }


def tool_label(name: str) -> str:
    return TOOL_LABELS.get(name, "Working")


def tool_event(
    *,
    tool_id: str,
    name: str,
    status: str,
    detail: str | None = None,
    sql: str | None = None,
    query: str | None = None,
    ok: bool | None = None,
    elapsed_ms: float | None = None,
    rows: int | None = None,
    sources_n: int | None = None,
) -> dict[str, Any]:
    """LibreChat-style per-tool card event for FE.

    Normal UI uses label/detail only. Debug fields (tool_name/sql/query) are optional.
    """
    stage = "web" if name == "web_search" else "genie" if name != "chart_plan" else "synthesize"
    if name == "chart_plan":
        stage = "synthesize"
    ev: dict[str, Any] = {
        "type": "tool_start" if status == "start" else "tool_end",
        "stage": stage,
        "status": status,
        "label": tool_label(name),
        "detail": detail,
        "ts": time.time(),
        "tool_id": tool_id,
        "tool_name": name,
    }
    if sql is not None:
        ev["sql"] = sql[:4000]
    if query is not None:
        ev["query"] = query[:500]
    if ok is not None:
        ev["ok"] = ok
    if elapsed_ms is not None:
        ev["elapsed_ms"] = elapsed_ms
    if rows is not None:
        ev["rows"] = rows
    if sources_n is not None:
        ev["sources_n"] = sources_n
    return ev
