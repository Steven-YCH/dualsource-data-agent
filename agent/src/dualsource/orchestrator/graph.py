"""LangGraph state-graph orchestrator for DualSource Agent.

Graph:
  START → guard_and_route → execute_tools → synthesize [→ generate_files] → END

Nodes:
  guard_and_route: LLM-based guardrail + intent routing  (Llama 70B, ~1-3s)
  execute_tools:   Parallel Genie + Tavily execution     (5-30s)
  synthesize:      LLM merges results into answer         (GPT-OSS-120B, ~2-4s)
  generate_files:  PPT / Excel if requested               (~0.5s)
"""

from __future__ import annotations

import operator
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

import mlflow

mlflow.langchain.autolog()  # Enable node-level tracing for LangGraph spans

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# NOTE: contracts used at the service.py layer, not inside graph nodes
from dualsource.orchestrator.llm import (
    ROUTER_MODEL,
    SYNTH_MODEL,
    llm_call,
    llm_json,
)
from dualsource.tools.charts import maybe_chart_from_table
from dualsource.tools.documents import generate_pptx, generate_xlsx
from dualsource.tools.genie import GenieClient
from dualsource.tools.tavily import TavilyClient
# tracing handled by service.py wrapper

# ---- State schema ----


# ---- Genie space ----

# Single unified Genie space (all domains: traffic, flights, baggage)
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID") or os.getenv("PROD002_SPACE_ID") or "01f1c3b038ba160db274ed2aa5f651b9"


# ---- Genie conversation reuse (WS2) ----
# Safety-net only — primary reset path is router's reset_genie_conversation field
_RESET_KW = {"start over", "ignore previous", "new topic", "forget", "reset", "fresh start", "begin again"}

_CHART_TYPE_RE = re.compile(r"\b(bar|line|pie)\b", re.IGNORECASE)
_CHART_ASK_RE = re.compile(r"\b(chart|charts|graph|graphs|plot|plots|visuali[sz]\w*)\b", re.IGNORECASE)


def _detect_chart_type(question: str) -> str:
    """Deterministic fallback for the router's chart_type field.

    Returns the explicitly requested chart type — "bar", "line", or "pie" —
    or "" when the question names no chart-type keyword.
    """
    match = _CHART_TYPE_RE.search(question or "")
    return match.group(1).lower() if match else ""


def _detect_chart_ask(question: str) -> bool:
    """Deterministic safety net: question mentions charts AND a chart-type word."""
    q = question or ""
    return bool(_CHART_ASK_RE.search(q) and _CHART_TYPE_RE.search(q))


_genie_sessions: dict[str, str] = {}  # session_id -> last successful genie conversation_id
_genie_session_topics: dict[str, str] = {}  # session_id -> last topic (pax/cargo/atm/baggage)
_genie_session_sql: dict[str, str] = {}  # session_id -> last Genie SQL for sql_request

_STAGE_LABELS = {
    "route": "Understanding question",
    "genie": "Querying internal data",
    "web": "Checking online sources",
    "synthesize": "Drafting answer",
    "files": "Preparing export",
    "complete": "Complete",
}


def _stage_event(stage: str, status: str) -> dict:
    """Create a progress stage event (WS4)."""
    return {
        "type": "stage",
        "stage": stage,
        "status": status,
        "label": _STAGE_LABELS.get(stage, stage),
        "ts": time.time(),
    }


# ---- Deterministic NLU removed (v25) ----
# _rewrite_genie_question(), _detect_follow_up_type(), and all keyword sets
# deleted. Router LLM now owns language understanding. follow_up_type is a router output field.

def _extract_prior_summary(history: list) -> str:
    """Extract a useful summary from the last assistant message for context carry-forward."""
    if not history:
        return ""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # Keep a meaningful chunk — enough for context, not full export
            return content[:1500].strip()
    return ""


def _normalize_table_title(title: str) -> str:
    """Clean Genie-style verbose titles into short business titles.

    'The total number of passengers handled at the hub for the latest 12 months'
    → 'Total passengers — latest 12 months'
    """
    import re as _re
    t = title.strip()
    # 1. Strip leading verbose restatement patterns
    t = _re.sub(
        r"^(?:"
        r"You (?:want(?:ed)? to|asked to) (?:see|know|view)"
        r"|Here (?:is|are)"
        r"|This (?:shows?|displays?|provides?|presents?)"
        r"|Below (?:is|are)"
        r"|The following (?:table |chart |data )?(?:shows?|displays?|presents?|contains?)"
        r"|A (?:table|chart|breakdown|summary|view) (?:showing|of|for|with)"
        r"|Showing"
        r")[:\s]+",
        "", t, flags=_re.IGNORECASE,
    )
    # 2. Strip leading "The total number of" → "Total"
    t = _re.sub(r"^The\s+total\s+(?:number|count|amount|volume|sum)\s+of\s+", "Total ", t, flags=_re.IGNORECASE)
    # 3. Strip leading "The top / The list of / The breakdown of"
    t = _re.sub(r"^The\s+(?:top|list|breakdown|distribution|split|composition)\s+(?:of\s+)?", "", t, flags=_re.IGNORECASE)
    # 4. Strip trailing noise
    t = _re.sub(r"\s+at\s+(?:the hub|Hong Kong International Airport)\s*$", "", t, flags=_re.IGNORECASE)
    t = _re.sub(r"\s+(?:as requested|as shown below|as follows|for your reference)\s*\.?\s*$", "", t, flags=_re.IGNORECASE)
    # 5. Capitalize first letter
    if t:
        t = t[0].upper() + t[1:]
    # 6. Enforce max length
    if len(t) > 80:
        t = t[:75].rsplit(" ", 1)[0] + "\u2026"
    return t or title


class GraphState(TypedDict, total=False):
    # Input
    question: str
    history: list
    user_email: str
    session_id: str
    files_dir: str
    # Router
    rejected: bool
    reject_reason: str
    need_genie: bool
    need_web: bool
    need_chart: bool
    chart_type: str  # "" | "bar" | "line" | "pie" (explicit user request)
    need_pptx: bool
    need_xlsx: bool
    genie_question: str
    genie_space_id: str  # which sub-agent to use
    web_queries: list
    follow_up_type: str  # attribution / compare_sidebyside / sql_request / breakdown / normal
    prior_summary: str   # last assistant answer for multi-turn context
    # Tool results
    genie_result: dict | None
    genie_sql: str | None  # real Genie SQL for sql_request (cached per session)
    tavily_results: list
    # Synthesis
    answer_markdown: str
    sources: list
    tables: list
    export_tables: list  # full rows for Excel (up to 500)
    charts: list
    files: list
    followups: list
    status: dict
    trace_id: str | None
    current_step: str
    # Genie conversation reuse (WS2)
    genie_conversation_id: str | None
    reuse_genie_conversation: bool
    # Progress events (WS4) -- Annotated with operator.add for LangGraph list append
    progress_events: Annotated[list, operator.add]

# ---- Prompts ----

ROUTE_SYSTEM = """You are a routing classifier for DualSource Agent, an the hub (Hong Kong International Airport) analytics assistant.

Given the user's question (and any previous-answer context), output ONLY valid JSON (no markdown fences, no explanation):
{"rejected": bool, "reject_reason": string_or_null, "need_genie": bool, "need_web": bool, "need_chart": bool, "chart_type": string_or_null, "need_xlsx": bool, "need_pptx": bool, "genie_question": string_or_null, "web_queries": list_of_strings_or_null, "reuse_genie_conversation": bool, "reset_genie_conversation": bool, "follow_up_type": string}

Rules:
- "rejected": true ONLY if the question has absolutely nothing to do with airports, aviation, Hong Kong, travel, cargo, logistics, trade, tourism, or related business topics. Be LOOSE.
- "need_genie": true when the hub / Hong Kong airport internal data is needed (passengers, cargo, flights, baggage, airlines, destinations, ATM, air traffic movements, route data).
- "reuse_genie_conversation": true when this is a follow-up to a previous the hub data question in the same topic area. false when changing topic or starting fresh.
- "reset_genie_conversation": true when the user explicitly wants to start fresh (e.g. "start over", "new topic", "forget previous", "reset"). false otherwise.
- "need_web": true when external/public information is needed.
- When both the hub data AND external context are needed, set both to true.
- "follow_up_type": classify the intent. One of: "normal", "attribution", "compare_sidebyside", "sql_request", "breakdown".
  - "attribution": user asks why/drivers/causes/reasons. Set need_web=true. Only also set need_genie=true if asking for a specific metric breakdown.
  - "compare_sidebyside": user wants side-by-side comparison. Set need_genie=true AND need_web=true for peer data.
  - "sql_request": user wants to see the SQL/query. Set need_genie=false, need_web=false.
  - "breakdown": user wants dimension breakdown. Set need_genie=true.
  - "normal": standard new question or follow-up.

GENIE QUESTION QUALITY (critical — this replaces all deterministic rewriting):
- "genie_question": rewrite the question to focus ONLY on the hub/HKG internal metrics. Make it specific, data-oriented, and unambiguous.
- For vague trend asks without explicit dates, specify "latest 12 complete months" and order chronologically.
- For YTD asks without explicit year, specify the current year and "January through the last complete month". Sum all available months.
- For ATM (air traffic movements): instruct Genie to count true movements using COUNT(DISTINCT CONCAT_WS('|', flight_id, DATE(flight_date), direction)). Do NOT use COUNT(DISTINCT flight_id) — that counts flight numbers, not movements. Always wrap flight_date in DATE() because it is a timestamp.
- MULTI-METRIC: if the user asks for MULTIPLE metrics together (e.g. passengers AND cargo), keep ALL requested metrics in genie_question. Never drop one metric because another is easier. Ask for both in side-by-side columns. Note that freshness may differ by metric.
- Prefer total metrics by default; do not split into sub-components unless the user explicitly asks.
- Include formatting hints: format year_month as YYYY-MM, use human-readable column aliases, show percentages as percent values, round floats to at most 2 decimal places.
- For market share: share = airline_pax / total_pax_same_period * 100. No share can exceed 100%.
- For short/vague questions without dates (<=4 words), add "latest 12 months" recency.

FOLLOW-UP / MULTI-TURN RULES:
- If user says "export" or "download" without format, assume follow-up: set need_genie=false, need_web=false.
- "need_chart": true if user explicitly asks for chart, trend, plot, graph, visualization, or visual.
- "chart_type": the chart type the user EXPLICITLY requested: "bar" (bar chart / bar graph), "line" (line chart / trend line), or "pie" (pie chart). null when no explicit type is requested or need_chart is false.
- "need_xlsx": true if user asks for Excel, spreadsheet, or workbook export.
- "need_pptx": true ONLY if user explicitly asks for PowerPoint, PPT, or presentation.

BREAKDOWN / ATTRIBUTE REQUESTS:
- Default to the total metric grouped by the natural dimension. Do NOT request mixed total + component splits unless user asks.

AMBIGUOUS / VAGUE PROMPTS:
If the question is vague / context-dependent AND no previous-answer context is provided, set need_genie=false and need_web=false."""

SYNTH_SYSTEM = """You are the DualSource Agent analyst for the hub (Hong Kong International Airport).
Your audience is airport senior management — write crisp, insight-led answers.

FORMAT RULES:
1. Lead with a single-sentence headline insight (bold the key number).
2. Then provide concise analysis — 2-4 short paragraphs or bullet points max.
3. State the time period being analysed.
4. Use **bold** for key figures. Include absolute values AND percentage changes when data allows.
5. Separate observed facts from possible explanations.
6. Use markdown: headers (##), bold, bullet points. Keep it scannable.
7. Do NOT create sections called "Key Figures", "Observations", or "Notes" — integrate naturally.
8. CHARTS: When the context says charts are available, place [[CHART:0]], [[CHART:1]], etc. each on its own line DIRECTLY BELOW the paragraph they illustrate. Interleave text and charts — do NOT group all charts at the end. Do NOT output [[CHART:n]] unless told charts are available.
9. Round all numeric values: integers for counts, 1 decimal for tonnes, 1-2 decimals for percentages. Never show more than 2 decimal places.
10. Never expose internal table names in user-facing answers. Use business terms.

TABLES / CHARTS — CRITICAL:
- Structured tables[] from tools are GROUND TRUTH for the hub internal numbers. Genie prose is secondary context only and is often incomplete (showing only example rows). When structured rows are provided, they are authoritative — every figure must come from them.
- Do NOT claim a month/period is missing or unavailable if it appears in the structured rows. Do NOT invent placeholder dashes ("–") for periods present in structured rows. Do NOT stamp fabricated gaps with [Internal].
- Do NOT restate the same detailed tool table as a markdown data table in your answer. The UI already renders structured tables below the answer. Narrate the key insights in prose instead.
- Markdown tables ARE allowed when they add value beyond the structured table, especially: compact summary/compare matrices, cross-source KPI cards (the hub + public peer figures), short summary tables (averages, shares, peaks) that are NOT row-clones of the structured table.
- ANTI-DUPLICATION: Do NOT present the same detailed series as BOTH a markdown data table AND [[CHART:n]].
  * Small/compact sets (roughly ≤8–12 rows, or a summary compare matrix): prefer a markdown table. Do NOT also place [[CHART:n]] for that same series unless the user explicitly asked for a chart/trend/visualization.
  * Larger/trend series (roughly >12 points, long monthly trend, multi-series): prefer [[CHART:n]] inline. Do NOT also emit a markdown table cloning the same month-by-month series.
  * If the user explicitly asks for BOTH table and chart, allow both — but the markdown table must match the authoritative structured rows exactly, no invented blanks.
- When choosing chart as primary, place [[CHART:n]] on its own line directly below the paragraph it illustrates. When choosing markdown summary as primary, omit [[CHART:n]] for that same series.

INLINE CITATIONS — CRITICAL:
Each data source has a citation tag provided in the context. Place the citation tag IMMEDIATELY after every claim it supports.
- ALWAYS cite. Every data-backed claim MUST have a citation tag.
- Use the EXACT citation tags provided — do not invent your own.
- Do NOT create a separate Sources section — the inline tags ARE the citations.

PERCENTAGE / SHARE MATH — CRITICAL:
- Market share = airline passengers / total passengers for the SAME period x 100. Shares MUST sum to ~100%.
- YoY % change = (current - prior) / prior x 100. A decline of 1.5% is -1.5%, NOT -150%.
- If a percentage looks implausible, flag it. Never double-scale.

PERIOD POLICY — CRITICAL:
- "YTD" = January through the LAST COMPLETE month. Currently the last complete month is September 2026. October 2026 is partial/preliminary.
- State the exact months included in every aggregate.
- Freshness varies by metric — state the period for each metric when comparing.

ATM / FLIGHTS DEFINITION:
- ATM = total aircraft takeoffs and landings. A round-trip = 2 ATMs.
- If ATM count seems suspiciously low for a full month, flag it.

CAUSAL HONESTY:
- NEVER invent explanations for zero or unusual values.
- If the user asked for multiple metrics but the tool result only contains one, say the returned extract is partial — do NOT claim the missing metric is absent from the internal dataset.
- Only attribute causes when backed by a cited source.

ZERO / MISSING DATA:
- All-zero values = data not yet available.
- Sudden drops in final 1-2 months = flag as preliminary, do NOT extrapolate as trend.

COMPARISON / SIDE-BY-SIDE:
- Produce a compact aligned markdown SUMMARY comparison table (e.g. the hub vs peer KPI matrix with averages, shares, peaks).
- This means a compact summary matrix — NOT a row-clone of the Genie monthly detail table already in structured tables[].
- Disclose period coverage, unit definitions, and source types.
- If prior-turn context includes the hub data, reuse rather than re-querying.
- Every cell must be grounded in authoritative structured rows and/or cited web sources. No invented blanks.

SQL REQUEST:
- If real Genie SQL is provided in context, show it in a ```sql code block with a brief explanation.
- If no real SQL is available, honestly describe the query intent. Never invent SQL.

ATTRIBUTION / DRIVER FOLLOW-UPS:
- Focus on explaining WHY using web research. Do not re-state data as zero if previous answer had real values.

TABLE TITLES: Short business titles under 80 characters.

BREAKDOWN DEFAULTS: Default to most natural dimension. No mixed splits unless asked.

STYLE:
- Never show SQL or technical details unless explicitly asked.
- Never mention "Genie", "Tavily", "LangGraph", or other internal tool names.
- PowerPoint/PPT export is NOT available. Suggest Excel.

LANGUAGE: Reply in same language as user, or English if mixed.

PLEASANTRIES: Brief acknowledgment only (1 sentence).

CARGO UNITS: Always "tonnes" or "million tonnes" — never "t" or "mt".

EXPORT RULES: Only mention Excel when explicitly asked. Never unsolicited.

FOLLOWUPS:
After your answer, output [FOLLOWUPS] on its own line, then 2-3 natural follow-up questions as bullet points."""


# ---- Node functions ----


def guard_and_route(state: GraphState) -> dict:
    """LLM-based guardrail + intent routing (Llama 70B, ~1-3s)."""
    question = state["question"]
    history = state.get("history") or []

    # Multi-turn: extract prior summary for context carry-forward
    prior_summary = _extract_prior_summary(history)

    # Build router context with runtime facts + prior summary
    from datetime import date as _date
    _today = _date.today()
    runtime_facts = (
        f"RUNTIME FACTS:\n"
        f"- Today: {_today.isoformat()}\n"
        f"- Current year: {_today.year}\n"
        f"- YTD = January through the last complete month (September 2026)\n"
        f"- Data freshness: passengers and cargo data available through September 2026. October 2026 is partial/preliminary.\n"
        f"- Cargo units: always use tonnes or million tonnes\n"
        f"- ATM counting: COUNT(DISTINCT CONCAT_WS('|', flight_id, DATE(flight_date), direction))\n"
    )

    user_content = f"{runtime_facts}\n"
    if prior_summary:
        user_content += f"Previous answer context: {prior_summary[:600]}\n\n"
    user_content += f"New question: {question}"

    try:
        parsed = llm_json(ROUTER_MODEL, ROUTE_SYSTEM, user_content, max_tokens=500, timeout=15)
    except Exception:
        # Fallback: assume internal the hub question
        parsed = {
            "rejected": False,
            "need_genie": True,
            "need_web": False,
            "need_chart": False,
            "need_xlsx": False,
            "genie_question": question,
            "web_queries": None,
            "follow_up_type": "normal",
            "reset_genie_conversation": False,
        }

    need_genie = bool(parsed.get("need_genie", False))
    need_web = bool(parsed.get("need_web", False))
    follow_up_type = parsed.get("follow_up_type", "normal") or "normal"

    # Use router's genie_question directly — no deterministic rewrite
    genie_q = parsed.get("genie_question") or question

    # PPT detection: if user asked for PPT, short-circuit — don't fire tools or xlsx
    pptx_requested = bool(parsed.get("need_pptx", False))
    if pptx_requested:
        need_genie = False
        need_web = False

    # ---- Genie conversation reuse (WS2) ----
    session_id = state.get("session_id", "")
    reuse_genie = False
    genie_conv_id = None
    if need_genie and session_id:
        # Primary reset path: router's reset_genie_conversation field
        explicit_reset = bool(parsed.get("reset_genie_conversation", False))
        # Safety-net: keyword detection as fallback only
        if not explicit_reset:
            q_lower = question.lower()
            explicit_reset = any(kw in q_lower for kw in _RESET_KW)
        has_prior = session_id in _genie_sessions
        if explicit_reset:
            _genie_sessions.pop(session_id, None)
            _genie_session_topics.pop(session_id, None)
            _genie_session_sql.pop(session_id, None)
        elif has_prior:
            reuse_genie = bool(parsed.get("reuse_genie_conversation", False))
            if reuse_genie:
                genie_conv_id = _genie_sessions.get(session_id)

    need_chart = bool(parsed.get("need_chart", False))
    # Chart type: router's explicit answer, then deterministic keyword fallback.
    # E.g. "compare HKG with ICN ... and show in bar chart" -> "bar".
    chart_type = str(parsed.get("chart_type") or "").strip().lower()
    if chart_type not in {"bar", "line", "pie"}:
        chart_type = ""
    if not chart_type:
        chart_type = _detect_chart_type(question)
    if chart_type and _detect_chart_ask(question):
        need_chart = True  # safety net when the router missed the explicit ask

    # For sql_request, inject cached SQL into state for synthesize
    cached_sql = None
    if follow_up_type == "sql_request" and session_id:
        cached_sql = _genie_session_sql.get(session_id)

    return {
        "rejected": bool(parsed.get("rejected", False)),
        "reject_reason": parsed.get("reject_reason") or "",
        "need_genie": need_genie,
        "need_web": need_web,
        "need_chart": need_chart,
        "chart_type": chart_type,
        "need_pptx": pptx_requested,
        "need_xlsx": False if pptx_requested else bool(parsed.get("need_xlsx", False)),
        "genie_question": genie_q,
        "genie_space_id": GENIE_SPACE_ID,
        "web_queries": parsed.get("web_queries") or [],
        "follow_up_type": follow_up_type,
        "prior_summary": prior_summary,
        "genie_sql": cached_sql,
        "reuse_genie_conversation": reuse_genie,
        "genie_conversation_id": genie_conv_id,
        "current_step": "routing_done",
        "progress_events": [_stage_event("route", "start"), _stage_event("route", "end")],
    }


def execute_tools(state: GraphState) -> dict:
    """Run Genie and/or Tavily in parallel based on router decision."""
    genie_result = None
    tavily_results: list[dict[str, Any]] = []
    events: list[dict] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs: dict[str, Any] = {}

        if state.get("need_genie"):
            events.append(_stage_event("genie", "start"))
            space_id = state.get("genie_space_id") or GENIE_SPACE_ID
            genie = GenieClient(space_id=space_id)
            # WS2: pass conversation_id for reuse if available
            conv_id = state.get("genie_conversation_id") if state.get("reuse_genie_conversation") else None
            chart_prefer = state.get("chart_type") or "auto"
            futs["genie"] = pool.submit(
                genie.ask, state.get("genie_question", state["question"]), conv_id, chart_prefer
            )

        if state.get("need_web") and state.get("web_queries"):
            events.append(_stage_event("web", "start"))
            tavily = TavilyClient()
            for i, query in enumerate(state["web_queries"][:3]):
                futs[f"tavily_{i}"] = pool.submit(tavily.search, query)

        if "genie" in futs:
            genie_result = futs["genie"].result()
            # WS2: if reuse failed, retry with new conversation
            if genie_result and genie_result.status == "error" and state.get("reuse_genie_conversation"):
                genie = GenieClient(space_id=state.get("genie_space_id") or GENIE_SPACE_ID)
                genie_result = genie.ask(
                    state.get("genie_question", state["question"]),
                    chart_prefer=state.get("chart_type") or "auto",
                )
            # WS2: cache successful conversation_id for future reuse
            session_id = state.get("session_id", "")
            if session_id and genie_result and getattr(genie_result, "conversation_id", None):
                _genie_sessions[session_id] = genie_result.conversation_id
            # Cache Genie SQL for sql_request follow-ups
            if session_id and genie_result and getattr(genie_result, "genie_sql", None):
                _genie_session_sql[session_id] = genie_result.genie_sql
            events.append(_stage_event("genie", "end"))

        for key in sorted(k for k in futs if k.startswith("tavily_")):
            tavily_results.append(futs[key].result())
        if any(k.startswith("tavily_") for k in futs):
            events.append(_stage_event("web", "end"))

    primary_dict = None
    genie_sql_current = None
    if genie_result:
        primary_dict = genie_result.__dict__ if hasattr(genie_result, "__dict__") else {
            "status": genie_result.status, "text": genie_result.text,
            "tables": genie_result.tables,
            "export_tables": getattr(genie_result, "export_tables", []),
            "charts": genie_result.charts,
            "had_query_result": genie_result.had_query_result,
        }
        genie_sql_current = getattr(genie_result, "genie_sql", None)

    return {
        "genie_result": primary_dict,
        "genie_sql": genie_sql_current or state.get("genie_sql"),
        "tavily_results": [
            {"status": r.status, "text": r.text, "sources": r.sources}
            for r in tavily_results
            if r is not None
        ],
        "current_step": "tools_done",
        "progress_events": events,
    }


def synthesize(state: GraphState) -> dict:
    """LLM synthesizes all tool results into a coherent answer with inline citations."""
    _synth_events: list[dict] = [_stage_event("synthesize", "start")]
    follow_up_type = state.get("follow_up_type", "normal")
    prior_summary = state.get("prior_summary", "")

    # ---- PPT-request short-circuit ----
    if state.get("need_pptx"):
        return {
            "answer_markdown": (
                "PowerPoint export is not currently available.\n\n"
                "I can export data as **Excel** instead — just ask "
                "\"export as Excel\" after any data query."
            ),
            "sources": [],
            "tables": [],
            "export_tables": [],
            "charts": [],
            "followups": [
                "Show me the latest passenger trends",
                "Export this month's cargo data as Excel",
            ],
            "status": {"genie": "skipped", "web": "skipped", "files": "skipped"},
            "current_step": "synthesized",
        }

    if state.get("rejected"):
        return {
            "answer_markdown": (
                "I'm designed to help with the hub airport analytics, aviation data, "
                "and related business topics.\n\n"
                "Try asking about passenger traffic, cargo volumes, airline performance, "
                "baggage handling, or airport operations."
            ),
            "sources": [],
            "tables": [],
            "export_tables": [],
            "charts": [],
            "followups": [
                "How did the airport perform last month?",
                "Show me passenger traffic trends",
            ],
            "status": {"genie": "skipped", "web": "skipped", "files": "skipped"},
            "current_step": "synthesized",
        }

    # ---- Collect tool results ----
    genie_data = state.get("genie_result")
    tavily_data = state.get("tavily_results") or []

    sources: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    charts: list[dict[str, Any]] = []
    export_table: dict[str, Any] | None = None
    status_genie = "skipped"
    status_web = "skipped"
    has_chart = False

    synth_context_parts: list[str] = []
    citation_tags: list[str] = []
    # Explicit user chart type ("bar"/"line"/"pie") or "auto" heuristic.
    chart_prefer_user = state.get("chart_type") or "auto"

    # -- Genie (internal data) --
    if genie_data:
        status_genie = genie_data.get("status", "skipped")
        if genie_data.get("status") == "ok":
            cite_tag = "[Internal]"
            citation_tags.append(cite_tag)
            sources.append({
                "type": "hkia_internal",
                "label": "Internal warehouse",
                "cite_tag": cite_tag,
            })
            genie_text = genie_data.get('text', '')
            # Zero-result guard: detect all-zero / empty metric tables
            _genie_tables_raw = genie_data.get("tables") or []
            _all_zero = False
            if _genie_tables_raw:
                _numeric_vals = []
                for _t in _genie_tables_raw:
                    for _row in (_t.get("rows") or []):
                        for _v in _row:
                            if isinstance(_v, (int, float)):
                                _numeric_vals.append(_v)
                if _numeric_vals and all(v == 0 for v in _numeric_vals):
                    _all_zero = True
            if _all_zero:
                genie_text = (
                    "The query returned all-zero values. This likely means data "
                    "is not yet available for the requested period. "
                    "Do NOT invent explanations like operational shutdowns."
                )
                status_genie = "insufficient"
                if prior_summary and any(
                    kw in prior_summary.lower()
                    for kw in ("million", "passengers", "tonnes", "cargo", "bags")
                ):
                    genie_text += (
                        "\n\nIMPORTANT: The previous answer contained real non-zero values. "
                        "The current query returned zeros — this is likely a query issue, "
                        "not a real data change. Do NOT claim the metric is now zero or unavailable "
                        "when the previous answer already showed real values. "
                        "Use the previous answer's data as the authoritative source."
                    )
            synth_context_parts.append(
                f"the hub INTERNAL DATA (cite as {cite_tag}):\n{genie_text}"
            )
            for t in _genie_tables_raw:
                if t.get("title"):
                    t["title"] = _normalize_table_title(t["title"])
                tables.append(t)
            if genie_data.get("export_tables"):
                export_table = genie_data["export_tables"][0]
            elif tables:
                export_table = tables[0]
            for t in tables:
                chart = maybe_chart_from_table(t, prefer=chart_prefer_user)
                if chart:
                    charts.append(chart)
                    has_chart = True
            # ---- Inject authoritative structured rows into synth context (trust fix) ----
            _auth_source = export_table or (tables[0] if tables else None)
            if _auth_source and (_auth_source.get("columns") and _auth_source.get("rows")):
                _auth_cols = _auth_source["columns"]
                _auth_rows = _auth_source["rows"]
                _MAX_AUTH_ROWS = 24
                _truncated = len(_auth_rows) > _MAX_AUTH_ROWS
                _display_rows = _auth_rows[:_MAX_AUTH_ROWS]
                _tsv_lines = ["\t".join(str(c) for c in _auth_cols)]
                for _r in _display_rows:
                    _tsv_lines.append("\t".join(str(v) for v in _r))
                _tsv_block = "\n".join(_tsv_lines)
                _auth_note = f" (showing first {_MAX_AUTH_ROWS} of {len(_auth_rows)} rows)" if _truncated else ""
                synth_context_parts.append(
                    f"AUTHORITATIVE STRUCTURED RESULTS (ground truth, cite as [Internal]){_auth_note}:\n"
                    f"{_tsv_block}\n\n"
                    "RULES FOR THESE ROWS:\n"
                    "- Every the hub figure MUST come from these rows when present.\n"
                    "- If a month/value appears here, it IS available \u2014 never say missing.\n"
                    "- Do NOT invent dashes for periods present in these rows.\n"
                    "- Prefer these rows over Genie prose examples.\n"
                    "- Do NOT restate these rows as a markdown data table \u2014 the UI renders them.\n"
                    "- Apply the markdown-vs-chart anti-duplication policy from your system instructions."
                )
        elif genie_data.get("status") == "insufficient":
            synth_context_parts.append(
                f"the hub DATA NOTE: {genie_data.get('text', 'Insufficient data')}"
            )
        else:
            synth_context_parts.append(
                "the hub DATA NOTE: Internal data source is temporarily unavailable."
            )

    # -- Tavily (web research) --
    web_ok = False
    for tav in tavily_data:
        if tav.get("status") == "ok":
            web_ok = True
            tav_sources = tav.get("sources") or []
            cite_labels: list[str] = []
            for src in tav_sources:
                label = src.get("label", "online source")
                url = src.get("url", "")
                if url:
                    cite_tag = f"[{label}]({url})"
                else:
                    cite_tag = f"[{label}]"
                citation_tags.append(cite_tag)
                cite_labels.append(cite_tag)
                sources.append({"type": "public", "label": label, "url": url, "cite_tag": cite_tag})
            if not cite_labels:
                cite_tag = "[online source]"
                citation_tags.append(cite_tag)
                cite_labels.append(cite_tag)
                sources.append({"type": "public", "label": "online source", "cite_tag": cite_tag})
            tags_str = ", ".join(cite_labels[:3])
            synth_context_parts.append(
                f"EXTERNAL RESEARCH (cite as {tags_str}):\n{tav.get('text', '')}"
            )
    status_web = "ok" if web_ok else ("error" if tavily_data else "skipped")

    # -- Prior summary for multi-turn context --
    if prior_summary and follow_up_type in ("attribution", "compare_sidebyside", "breakdown", "sql_request"):
        synth_context_parts.insert(0,
            f"PREVIOUS ANSWER (for context — reuse these figures if current query returned zeros or is insufficient):\n{prior_summary[:2000]}"
        )

    # -- SQL request: inject real Genie SQL if available --
    if follow_up_type == "sql_request":
        cached_sql = state.get("genie_sql")
        if cached_sql:
            synth_context_parts.append(
                f"GENIE SQL (the real SQL generated for the previous data query):\n```sql\n{cached_sql}\n```"
            )

    # -- Fallback for export follow-ups or ambiguous prompts --
    if not synth_context_parts:
        if prior_summary:
            synth_context_parts.append(
                f"PREVIOUS ANSWER (for export):\n{prior_summary[:2000]}"
            )
        else:
            return {
                "answer_markdown": (
                    "Could you be more specific? For example, I can help with:\n\n"
                    "- **Passenger traffic** — monthly trends, YTD totals, airline breakdowns\n"
                    "- **Cargo volumes** — tonnage, import/export splits, growth rates\n"
                    "- **Flight operations** — daily flights, destinations, capacity\n"
                    "- **Baggage handling** — dwell times, system performance\n\n"
                    "What would you like to know?"
                ),
                "sources": [],
                "tables": [],
                "charts": [],
                "followups": [
                    "How many passengers did the hub handle this year?",
                    "Show me monthly cargo trends",
                    "What are the busiest routes?",
                ],
                "status": {"genie": status_genie, "web": status_web, "files": "skipped"},
                "current_step": "synthesized",
            }

    # ---- Build LLM prompt with citation tags ----
    question = state["question"]
    context = "\n\n---\n\n".join(synth_context_parts)
    show_chart = has_chart and state.get("need_chart", False)
    chart_hint = ""
    if show_chart and charts:
        n = len(charts)
        indices = ", ".join(f"[[CHART:{i}]]" for i in range(n))
        chart_hint = (
            f"\n\n{n} chart(s) are available: {indices}. "
            f"Place each on its own line directly below the paragraph it illustrates. "
            f"Interleave text and charts — do NOT group them at the end."
        )
    tags_summary = ", ".join(citation_tags[:6]) if citation_tags else "none"
    # Soft follow-up context (not hardcoded branch logic)
    follow_up_context = ""
    if follow_up_type and follow_up_type != "normal":
        follow_up_context = f"\n\nFollow-up type: {follow_up_type}. Apply the relevant guidance from your system instructions."
    user_prompt = (
        f"User question: {question}\n\n"
        f"{context}"
        f"{chart_hint}"
        f"{follow_up_context}\n\n"
        f"Available citation tags: {tags_summary}\n"
        "Write your answer following the system instructions. "
        "Cite every data-backed claim using the exact citation tags above."
    )

    try:
        raw = llm_call(
            SYNTH_MODEL,
            [
                {"role": "system", "content": SYNTH_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2000,
            temperature=0,
            timeout=45,
        )
    except Exception:
        raw = "\n\n".join(synth_context_parts).strip()

    # ---- Parse answer + followups ----
    answer, followups = _parse_answer_followups(raw)

    import re as _re
    answer = _re.sub(r"\u3010([^\u3011]+)\u3011", r"[\1]", answer)

    # ---- P2: Round floats in tables ----
    for t in tables:
        if t.get("rows"):
            new_rows = []
            for row in t["rows"]:
                new_row = []
                for v in row:
                    if isinstance(v, float):
                        if abs(v) < 1000 and abs(v) > 0.001:
                            new_row.append(round(v, 2))
                        else:
                            new_row.append(round(v, 1))
                    else:
                        new_row.append(v)
                new_rows.append(new_row)
            t["rows"] = new_rows
        if t.get("columns"):
            t["columns"] = [c.replace("_", " ").title() for c in t["columns"]]

    import re as _re_chart
    if charts:
        found = set(_re_chart.findall(r"\[\[CHART:(\d+)\]\]", answer))
        for i, c in enumerate(charts):
            c["index"] = i
            c["position"] = "inline" if str(i) in found else "end"
    def _strip_orphan(m: _re_chart.Match) -> str:
        idx = int(m.group(1))
        return m.group(0) if charts and idx < len(charts) else ""
    answer = _re_chart.sub(r"\[\[CHART:(\d+)\]\]", _strip_orphan, answer)
    answer = answer.replace("[[CHART]]", "[[CHART:0]]" if charts else "").strip()

    for c in charts:
        if c.get("title"):
            c["title"] = _normalize_table_title(c["title"])

    if status_genie == "ok" and not state.get("need_xlsx"):
        followups.append("Export this as Excel")

    export_tables_out: list[dict[str, Any]] = []
    if export_table:
        export_tables_out.append(export_table)

    return {
        "answer_markdown": answer or "No answer produced.",
        "sources": sources,
        "tables": tables,
        "export_tables": export_tables_out,
        "charts": charts,
        "followups": followups[:5],
        "status": {"genie": status_genie, "web": status_web, "files": "skipped"},
        "current_step": "synthesized",
        "progress_events": _synth_events + [_stage_event("synthesize", "end"), _stage_event("complete", "end")],
    }


def _parse_answer_followups(raw: str) -> tuple[str, list[str]]:
    """Split LLM output into answer markdown and followup suggestions."""
    import re as _re

    # Normalise LLM followup markers (may have spaces, brackets, etc.)
    normalised = _re.sub(r"\[\s*FOLLOWUPS?\s*\]", "[FOLLOWUPS]", raw)

    answer = normalised
    followups: list[str] = []

    if "[FOLLOWUPS]" in normalised:
        parts = normalised.split("[FOLLOWUPS]")
        answer = parts[0].strip()
        for line in parts[1].strip().split("\n"):
            line = line.strip().lstrip("- ").lstrip("* ").strip().rstrip("'\"}]")
            if line and len(line) > 10 and not line.startswith("["):
                followups.append(line)

    # Fallback: detect trailing bullet block
    if not followups and "\n- " in raw:
        lines = raw.strip().split("\n")
        trailing_bullets: list[str] = []
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.startswith("- ") or stripped.startswith("* "):
                trailing_bullets.insert(
                    0, stripped.lstrip("- ").lstrip("* ").strip()
                )
            elif trailing_bullets:
                break
        if len(trailing_bullets) >= 2:
            followups = [b for b in trailing_bullets if len(b) > 10]

    return answer, followups


def generate_files(state: GraphState) -> dict:
    """Generate Excel exports if requested. PPT disabled (Wave 1)."""
    files: list[dict[str, Any]] = []
    file_status = "skipped"
    answer = state.get("answer_markdown", "")[:1800]
    tables = state.get("tables") or []
    export_table = None

    # Prefer export_tables (full rows) over chat tables (truncated)
    export_tables = state.get("export_tables") or []
    if export_tables:
        export_table = export_tables[0]
    else:
        genie_data = state.get("genie_result")
        if genie_data and genie_data.get("export_tables"):
            export_table = genie_data["export_tables"][0]
        elif tables:
            export_table = tables[0]

    files_dir = state.get("files_dir") or "/tmp/ci_files"
    user_email = state.get("user_email", "unknown")
    session_id = state.get("session_id", "session")

    # PPT disabled (Wave 1) — Excel only
    if state.get("need_xlsx"):
        xls = generate_xlsx(
            title="DualSource Agent Export",
            summary=answer,
            table=export_table,
            output_dir=files_dir,
            user_email=user_email,
            session_id=session_id,
        )
        if xls.status == "ok":
            file_status = "ok" if file_status != "error" else "error"
            for f in xls.files:
                files.append(f)

    # Merge status
    existing_status = state.get("status") or {}
    existing_status["files"] = file_status

    result: dict[str, Any] = {
        "files": files,
        "status": existing_status,
        "current_step": "files_done",
        "progress_events": [_stage_event("files", "start"), _stage_event("files", "end")] if files else [_stage_event("complete", "end")],
    }

    # When files were produced, replace verbose answer with short success text
    if files:
        file_names = [f.get("filename", "") for f in files]
        names_str = ", ".join(f"**{n}**" for n in file_names if n)
        result["answer_markdown"] = (
            f"Your export is ready: {names_str}\n\n"
            "Click the download button below to save."
        )
        # Remove chart/followup clutter on export turns
        result["charts"] = []
        result["followups"] = ["Show me the latest passenger trends"]

    return result


# ---- Graph construction ----


def _route_after_guard(state: GraphState) -> str:
    if state.get("rejected"):
        return "synthesize"
    if state.get("need_genie") or state.get("need_web"):
        return "execute_tools"
    # Pure export / follow-up
    return "synthesize"


def _route_after_synth(state: GraphState) -> str:
    if state.get("need_xlsx"):
        return "generate_files"
    return END


def build_graph() -> StateGraph:
    """Build and compile the DualSource Agent LangGraph."""
    builder = StateGraph(GraphState)

    builder.add_node("guard_and_route", guard_and_route)
    builder.add_node("execute_tools", execute_tools)
    builder.add_node("synthesize", synthesize)
    builder.add_node("generate_files", generate_files)

    builder.add_edge(START, "guard_and_route")
    builder.add_conditional_edges("guard_and_route", _route_after_guard, ["execute_tools", "synthesize"])
    builder.add_edge("execute_tools", "synthesize")
    builder.add_conditional_edges("synthesize", _route_after_synth, ["generate_files", END])
    builder.add_edge("generate_files", END)

    return builder.compile()


# Singleton compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
