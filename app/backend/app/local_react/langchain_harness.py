"""Opt-in supported ``create_agent`` retrieval harness for local_react.

Wren/Tavily tools deliberately reuse ToolRuntime's MCP clients and guards. This
adapter does not replace metric policy, chart synthesis, UC history, or SSE schema.
No checkpointer is installed: the existing per-user history is authoritative.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from typing import Any, Callable, Iterator

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer

from app.local_react.langchain_model import FoundryChatModel
from app.local_react.progress import tool_event, tool_label
from app.local_react.tools import _tavily_max_workers

_COMPLETE = "complete_retrieval"
_EXPLORE = {"wren_get_instructions", "wren_describe_schema", "wren_dry_run"}
_RETRIEVAL_PROMPT = """
You gather evidence for a separate final-answer writer. Do not write the user's
final answer. When the requested evidence is sufficient, call complete_retrieval.
If tools cannot gather the evidence, stop with the short text EVIDENCE_UNAVAILABLE.
Never invent missing numbers. Prefer wren_run_sql immediately for the hub KPIs;
do not start with wren_get_instructions, describe_schema, or SELECT 1 probes.
On hub_kpi_monthly use EXACT columns:
  passengers (never passengers/pax),
  cargo_tonnes (never cargo/cargo_tonnes),
  year_month for the period filter.
For YTD, return monthly rows with SUM(passengers) or SUM(cargo_tonnes)
in one query for the completed-month window from RUNTIME FACTS.
hub_kpi_monthly is route x month. the hub system total = SUM with NO airport_code
filter. Filter airport_code only when the user named one route partner.
A single route row is not the hub airport-wide cargo or passengers.
Run dependent SQL and chart tools in separate rounds; independent peer web searches
may run together. Retain completed-month YTD policy, Internal for the hub, and public web
only for other airports' own statistics.
Call complete_retrieval only when the question is covered. Do not stop early:
- If the user named several airports, issue one web_search for each named airport
  (same turn when you can). One peer result is not coverage. Say a figure is
  unavailable only after that airport was searched.
- Year-on-year or growth: query the same months for both years before
  complete_retrieval. One year of rows is not enough, and do not treat the other
  year as missing unless the query included it and returned no rows.
  If you UNION the two years, include the year as its own column. Do not alias both years with the same name.
- Baggage pieces: hub_kpi_monthly is one row per bag per flight. year_month is YYYY-MM text.
  Pieces = SUM(bag_count). There is no baggage_pieces column. bag_id is reused across months,
  so never COUNT(DISTINCT bag_id). A monthly series uses SUM(bag_count) per month, and the
  headline is the sum of those rows (or one SUM(bag_count) for the same months).
- SHA in a peer list is Shanghai Hongqiao, not Pudong. Search Pudong only when the user said Pudong or PVG.
- Passengers and cargo are SUM on hub_kpi_monthly (no airport_code filter
  unless the user named a route). ATM is a separate query on hub_kpi_monthly.
  If one statement fails, retry that metric on its own table. Do not drop a metric
  you have not successfully queried.
"""


def _has_web(sources: list[dict[str, Any]]) -> bool:
    return any(source.get("type") == "public" or source.get("url") for source in sources)


def _asked_question(messages: list[dict[str, Any]]) -> str:
    """The user question, without the runtime-facts preamble the App prepends."""
    content = str((messages[-1] or {}).get("content") or "")
    marker = "\nQuestion:\n"
    if marker in content:
        content = content.split(marker, 1)[1]
        for stop in ("\n\nConversation continuity:", "\n\nRouting reminder:"):
            if stop in content:
                content = content.split(stop, 1)[0]
    return content.strip()


class RetrievalPolicy(AgentMiddleware):
    """Bounded decisions plus the existing KPI, exploration and YTD guards."""

    def __init__(
        self, runtime: Any, *, kpi_ask: bool, peer_ask: bool, ytd_ask: bool,
        facts: dict[str, Any], max_rounds: int, coverage: Callable,
        web_snippets: list[str], max_tool_calls: int = 24,
    ) -> None:
        self.runtime = runtime
        self.kpi_ask, self.peer_ask, self.ytd_ask = kpi_ask, peer_ask, ytd_ask
        self.facts, self.coverage = facts, coverage
        self.max_rounds, self.max_tool_calls = max_rounds, max_tool_calls
        self.model_calls = self.tool_calls = self.explore_calls = 0
        self.stop_reason = "model_finished"
        self.ytd_nudged = False
        self.repair_ytd = False
        self.question = ""
        self.gap_nudges = 0
        self.lock = threading.Lock()
        # CI_TAVILY_MAX_WORKERS (default 8). The older orchestrator already read this;
        # this harness used to ignore it and stop at 4.
        self.web_parallelism = _tavily_max_workers()
        self.web_slots = threading.Semaphore(self.web_parallelism)
        self.web_snippets = web_snippets

    def _missing_evidence(self) -> bool:
        return (self.kpi_ask and not self.runtime.tables) or (
            self.peer_ask and self.runtime.tables and not _has_web(self.runtime.sources)
        )

    def _evidence_gap(self) -> str:
        """Metrics the question asked for that are not yet in the tables."""
        question = self.question.lower()
        blob = " ".join(
            " ".join(str(col) for col in (table.get("columns") or []))
            + " "
            + str(table.get("title") or "")
            for table in self.runtime.tables
        ).lower()
        missing: list[str] = []
        if re.search(r"baggage|pieces of bag", question) and "baggage_pieces" not in blob:
            missing.append(
                "one SUM(bag_count) AS baggage_pieces for the whole window"
            )
        wanted: list[tuple[str, str]] = []
        if re.search(r"\b(pax|passenger)", question):
            wanted.append(("passengers", "passenger"))
        if "cargo" in question:
            wanted.append(("cargo", "cargo"))
        if re.search(r"\batm\b|air traffic movement", question):
            wanted.append(("ATM on hub_kpi_monthly", "atm"))
        if len(wanted) >= 2:
            for label, needle in wanted:
                if needle not in blob:
                    missing.append(label)
        return "; ".join(missing)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        # Budget termination avoids an extra FAST final-answer generation. The
        # existing STRONG step gets any partial evidence and applies accuracy rules.
        if self.model_calls >= self.max_rounds or self.tool_calls >= self.max_tool_calls:
            self.stop_reason = "budget"
            return {"jump_to": "end"}
        expected = int(self.facts.get("expected_ytd_months") or "9")
        if (
            self.ytd_ask and self.runtime.tables and not self.ytd_nudged
            and self.coverage(self.runtime.tables) < max(6, expected - 1)
        ):
            self.ytd_nudged = self.repair_ytd = True
            return {"messages": [HumanMessage(content=(
                "Your SQL window is incomplete for YTD. Re-run wren_run_sql for ALL completed "
                f"months ({self.facts['ytd_window']}) using year_month >= TIMESTAMP '{self.facts['ytd_sql_start']}' "
                f"AND year_month < TIMESTAMP '{self.facts['ytd_sql_end_exclusive']}'. "
                f"Expect about {expected} monthly rows, then SUM the total."
            ))]}
        messages = state.get("messages") or []
        completed = any(
            isinstance(message, ToolMessage) and message.name == _COMPLETE
            for message in reversed(messages[-10:])
        )
        gap = self._evidence_gap() if completed else ""
        if completed and gap and self.gap_nudges < 2:
            self.gap_nudges += 1
            return {"messages": [HumanMessage(content=(
                "Do not stop. Still missing: " + gap + ". "
                "Run that query, then call complete_retrieval."
            ))]}
        if completed and not self._missing_evidence() and not self.repair_ytd and not gap:
            self.stop_reason = "evidence_ready"
            return {"jump_to": "end"}
        # Do not end just because one SQL row or one web result exists. The retrieval
        # prompt decides when every named airport, both YoY years, and each requested
        # metric are covered, then calls complete_retrieval.
        return None

    def wrap_model_call(self, request, handler):
        choice: Any = "auto"
        if self.repair_ytd:
            choice = {"type": "function", "function": {"name": "wren_run_sql"}}
            self.repair_ytd = False
        elif (self.kpi_ask or self.peer_ask) and not self.runtime.tables and self.model_calls < 2:
            # Force SQL directly. tool_choice=required let gpt-5.4-mini burn a round on
            # wren_get_instructions / SELECT 1 probes before any useful query.
            choice = {"type": "function", "function": {"name": "wren_run_sql"}}
        self.model_calls += 1
        return handler(request.override(tool_choice=choice))

    @hook_config(can_jump_to=["model", "end"])
    def after_model(self, state, runtime):
        message = state["messages"][-1]
        if isinstance(message, AIMessage) and not message.tool_calls:
            # Evidence already gathered: never spend another FAST prose round.
            if not self._missing_evidence() and not self.repair_ytd:
                self.stop_reason = "evidence_ready"
                return {"jump_to": "end"}
            if self._missing_evidence() and self.model_calls < self.max_rounds:
                return {"messages": [HumanMessage(content=(
                    "Evidence is still missing. Use wren_run_sql for the hub KPIs and "
                    "web_search for other airports. Do not answer from memory."
                ))], "jump_to": "model"}
        return None

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        name, args = call["name"], call.get("args") or {}
        call_id = call.get("id") or uuid.uuid4().hex
        if name == _COMPLETE:
            return handler(request)
        writer = get_stream_writer()
        detail = str(args.get("title") or args.get("query") or tool_label(name))[:160]
        writer(tool_event(tool_id=call_id, name=name, status="start", detail=detail,
                          sql=args.get("sql"), query=args.get("query")))
        with self.lock:
            skipped = self.tool_calls >= self.max_tool_calls
            self.tool_calls += 1
            spare_explore = self.kpi_ask and name in _EXPLORE and self.explore_calls >= 1 and not self.runtime.tables
            if name in _EXPLORE:
                self.explore_calls += 1
        if skipped or spare_explore:
            result = {"ok": not skipped, "skipped": True, "elapsed_ms": 0.0,
                      "note": "Tool budget reached" if skipped else "Explore step skipped for latency. Call wren_run_sql next."}
            message = ToolMessage(name=name, tool_call_id=call_id, content=json.dumps(result))
        else:
            try:
                if name == "web_search":
                    with self.web_slots:
                        message = handler(request)
                else:
                    message = handler(request)
                if isinstance(message.artifact, dict):
                    result = message.artifact
                elif message.status == "error":
                    result = {"ok": False, "error": str(message.content), "elapsed_ms": 0.0}
                else:
                    result = json.loads(message.content)
            except Exception as exc:
                # Tool validation/transport errors remain visible observations. No
                # automatic retry changes server side effects or hides latency.
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "elapsed_ms": 0.0}
                message = ToolMessage(name=name, tool_call_id=call_id, content=json.dumps(result), status="error")
        if not isinstance(result, dict):
            result = {"ok": False, "error": "Invalid tool result", "elapsed_ms": 0.0}
        if name == "web_search" and result.get("text"):
            with self.lock:
                self.web_snippets.append(str(result["text"])[:600])
        ok = bool(result.get("ok", True))
        rows = len((result.get("table") or {}).get("rows") or []) if name == "wren_run_sql" else None
        sources_n = int(result.get("sources_n") or len(result.get("sources") or [])) if name == "web_search" else None
        detail = str(result.get("error") or "Done")[:160]
        if ok and rows is not None:
            detail = f"{rows} rows returned"
        elif ok and sources_n is not None:
            detail = f"{sources_n} sources found"
        writer(tool_event(tool_id=call_id, name=name, status="end", detail=detail,
                          sql=result.get("sql") or args.get("sql"), query=result.get("query") or args.get("query"),
                          ok=ok, elapsed_ms=float(result.get("elapsed_ms") or 0), rows=rows, sources_n=sources_n))
        return message


def stream_retrieval(
    *, foundry: Any, runtime: Any, tools: list[dict[str, Any]],
    messages: list[dict[str, Any]], kpi_ask: bool, peer_ask: bool, ytd_ask: bool,
    facts: dict[str, Any], max_rounds: int, coverage: Callable,
    web_snippets: list[str], session_id: str = "", callbacks: list[Any] | None = None, stats: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream tool events as tools start/finish, retaining the runtime artifacts."""
    def make_tool(schema):
        function = schema["function"]
        name = function["name"]
        def execute(**args):
            result = runtime.execute(name, args)
            return json.dumps(result, ensure_ascii=False, default=str)[:12000], result
        return StructuredTool(name=name, description=function["description"],
                              args_schema=function["parameters"], func=execute, response_format="content_and_artifact")

    agent_tools = [make_tool(schema) for schema in tools]
    agent_tools.append(StructuredTool(
        name=_COMPLETE, description="Stop evidence retrieval once sufficient; a separate writer answers the user.",
        args_schema={"type": "object", "properties": {}, "additionalProperties": False},
        func=lambda: '{"ok": true}',
    ))
    policy = RetrievalPolicy(runtime, kpi_ask=kpi_ask, peer_ask=peer_ask, ytd_ask=ytd_ask,
                             facts=facts, max_rounds=max_rounds, coverage=coverage, web_snippets=web_snippets)
    policy.question = _asked_question(messages)
    system_prompt = str(messages[0]["content"]) + "\n\n" + _RETRIEVAL_PROMPT
    agent = create_agent(model=FoundryChatModel(client=foundry, model_name=foundry.fast_model),
                         tools=agent_tools, system_prompt=system_prompt, middleware=[policy],
                         name="dualsource_evidence_agent")
    try:
        yield from agent.stream(
            {"messages": messages[1:]}, stream_mode="custom",
            config={"recursion_limit": max_rounds * 5 + 10, "max_concurrency": policy.web_parallelism,
                    "run_name": "local_react.langchain_retrieval",
                    "metadata": {"harness": "langchain", "session_id": session_id}, "callbacks": callbacks},
        )
    finally:
        if stats is not None:
            stats.update(model_calls=policy.model_calls, tool_calls=policy.tool_calls, stop_reason=policy.stop_reason)
