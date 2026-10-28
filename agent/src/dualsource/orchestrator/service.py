"""Orchestrator: thin wrapper over the LangGraph state-graph.

The Orchestrator.handle() interface is unchanged so that
responses_agent.py and agent_client.py keep working.
"""

from __future__ import annotations

import logging
from typing import Any

from dualsource.contracts.response import (
    AgentResponse,
    ChartPayload,
    FilePayload,
    Source,
    StageEvent,
    TablePayload,
    ToolStatus,
)
from dualsource.orchestrator.graph import GraphState, get_graph
from dualsource.tracing.mlflow_trace import start_trace

log = logging.getLogger(__name__)


class Orchestrator:
    """LangGraph-backed orchestrator for DualSource Agent.

    Graph: guard_and_route → execute_tools → synthesize [→ generate_files] → END
    """

    def handle(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        user_email: str = "unknown",
        session_id: str = "session",
        files_dir: str | None = None,
    ) -> AgentResponse:
        with start_trace("dualsource-turn") as trace_id:
            initial_state: GraphState = {
                "question": question,
                "history": history or [],
                "user_email": user_email,
                "session_id": session_id,
                "files_dir": files_dir or "/tmp/ci_files",
            }

            graph = get_graph()
            try:
                result = graph.invoke(initial_state)
            except Exception:
                log.exception("LangGraph execution failed")
                return AgentResponse(
                    answer_markdown="An unexpected error occurred. Please try again.",
                    status=ToolStatus(genie="error", web="error", files="skipped"),
                    trace_id=trace_id,
                )

            # Convert dicts back into Pydantic models for AgentResponse
            sources = [Source(**s) for s in (result.get("sources") or [])]
            tables = []
            for t in result.get("tables") or []:
                if isinstance(t, dict):
                    tables.append(TablePayload(**t))
                elif isinstance(t, TablePayload):
                    tables.append(t)
            charts = []
            for c in result.get("charts") or []:
                if isinstance(c, dict):
                    charts.append(ChartPayload(**c))
                elif isinstance(c, ChartPayload):
                    charts.append(c)
            files = []
            for f in result.get("files") or []:
                if isinstance(f, dict):
                    files.append(FilePayload(**f))
                elif isinstance(f, FilePayload):
                    files.append(f)

            status_raw = result.get("status") or {}
            if isinstance(status_raw, ToolStatus):
                status = status_raw
            elif isinstance(status_raw, dict):
                status = ToolStatus(**{k: v for k, v in status_raw.items() if k in ("genie", "web", "files")})
            else:
                status = ToolStatus()

            # Map export_tables (full rows for Excel, up to 500)
            export_tables = []
            for t in result.get("export_tables") or []:
                if isinstance(t, dict):
                    export_tables.append(TablePayload(**t))
                elif isinstance(t, TablePayload):
                    export_tables.append(t)

            # Map progress events (WS4)
            progress_events = []
            for ev in result.get("progress_events") or []:
                if isinstance(ev, dict):
                    progress_events.append(StageEvent(**ev))
                elif isinstance(ev, StageEvent):
                    progress_events.append(ev)

            return AgentResponse(
                answer_markdown=result.get("answer_markdown", "No answer produced."),
                sources=sources,
                tables=tables,
                export_tables=export_tables,
                charts=charts,
                files=files,
                followups=result.get("followups", []),
                status=status,
                trace_id=trace_id,
                progress_events=progress_events,
            )
