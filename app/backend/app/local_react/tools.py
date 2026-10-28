"""Tool schemas + executors for local ReAct."""

from __future__ import annotations

import html
import json
import math
import os
import re
import threading
import time
from typing import Any, Callable

from urllib.parse import urlparse

from app.local_react.chart_plan import build_chart_from_plan
from app.local_react.mcp_observability import timed_stage
from app.local_react.duckdb_client import DuckDBSQLClient


def _cite_domain(url: str | None) -> str:
    """Short host for markdown cites (safe inside table cells)."""
    try:
        host = (urlparse(str(url or "")).netloc or "").lower()
    except Exception:
        host = ""
    if host.startswith("www."):
        host = host[4:]
    return host or "source"


def _safe_source_label(label: str, url: str | None = None) -> str:
    text = str(label or "").replace("|", " - ")
    text = text.replace(chr(10), " ").replace(chr(13), " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 100:
        text = text[:97] + "..."
    return text or _cite_domain(url)


def _cite_tag_for(label: str, url: str | None) -> str:
    """Prefer [domain](url). Long titles with | break markdown tables."""
    if not url:
        return f"[{_safe_source_label(label)}]"
    return f"[{_cite_domain(url)}]({url})"


# Soft cap on billable SQL per turn. Peer compares often need YTD + H1/Q3 + monthly.
_SQL_TURN_CAP = 6
_SQL_REFUND_CAP = 3


def _looks_like_sql_engine_error(err: str) -> bool:
    """Warehouse/parse failures that should not permanently burn the turn SQL cap."""
    e = (err or "").upper()
    return any(
        token in e
        for token in (
            "BAD_REQUEST",
            "UNRESOLVED_COLUMN",
            "PARSE_SYNTAX",
            "SQLSTATE",
            "INCONSISTENT_BEHAVIOR",
            "DATETIME_PATTERN",
            "FAILED {",
            "SQL FAILED:",
        )
    )


def _mixed_grain_union_message(sql: str) -> str | None:
    """Reject period-total UNION monthly year_month in one statement (Spark BAD_REQUEST)."""
    s = str(sql or "")
    if not re.search(r"\bUNION\s+ALL\b", s, re.I):
        return None
    has_period_label = bool(
        re.search(r"SELECT\s+'[^']+'\s+AS\s+period\b", s, re.I)
        or re.search(r"AS\s+period\b.+\bUNION\s+ALL\b", s, re.I | re.S)
    )
    has_monthly = bool(
        re.search(r"\byear_month\b", s, re.I)
        and re.search(r"\bGROUP\s+BY\s+year_month\b", s, re.I)
    )
    if has_period_label and has_monthly:
        return (
            "Rejected before warehouse: do not UNION a period-label total with monthly "
            "year_month rows in one statement (ORDER BY/GROUP columns resolve against the "
            "first branch only → UNRESOLVED_COLUMN). Run the period SUM as one SQL, and "
            "monthly GROUP BY year_month as a separate SQL."
        )
    return None


def _rewrite_common_column_aliases(sql: str) -> str:
    """Light cleanup before DuckDB execute."""
    sql = html.unescape(str(sql or ""))
    out = re.sub(r"\btotal_passengers\b", "passengers", sql, flags=re.I)
    out = re.sub(r"\btotal_cargo_tonnes\b", "cargo_tonnes", out, flags=re.I)
    out = re.sub(r"\batm\b", "aircraft_movements", out, flags=re.I)
    out = re.sub(
        r"\b[\w]+\.[\w]+\.(hub_kpi_monthly|route_kpi_monthly|hub_retail_monthly|hub_otp_monthly)\b",
        r"\1",
        out,
        flags=re.I,
    )
    return out


CHAT_TABLE_MAX_ROWS = 20


def _tavily_max_results(default: int = 5) -> int:
    raw = (os.getenv("TAVILY_MAX_RESULTS") or str(default)).strip()
    try:
        return max(1, min(int(raw), 8))
    except Exception:
        return default


def _tavily_max_searches(default: int = 8) -> int:
    """Max web_search tool calls per turn (parallel peer airports).

    Separate from TAVILY_MAX_RESULTS (hits returned per single search).
    Production Tavily plans can sustain high RPM; keep a soft product cap.
    """
    raw = (os.getenv("CI_TAVILY_MAX_SEARCHES") or str(default)).strip()
    try:
        return max(1, min(int(raw), 16))
    except Exception:
        return default


def _tavily_max_workers(default: int | None = None) -> int:
    """Concurrent Tavily fetches. Default matches per-turn search cap."""
    fallback = _tavily_max_searches(8) if default is None else default
    raw = (os.getenv("CI_TAVILY_MAX_WORKERS") or str(fallback)).strip()
    try:
        return max(1, min(int(raw), 16))
    except Exception:
        return fallback


def bias_web_query(query: str, *, current_year: str) -> str:
    """Pin SHA to Hongqiao and the current year. Do not inject an H1 alternative."""
    q = (query or "").strip()
    mentions_pudong = re.search(r"pudong|\bpvg\b", q, re.I)
    mentions_sha = re.search(r"\bsha\b|hongqiao", q, re.I)
    mentions_shanghai = re.search(r"\bshanghai\b", q, re.I)
    if not mentions_pudong and (mentions_sha or mentions_shanghai):
        if mentions_sha:
            q = re.sub(r"\bsha\b", "Shanghai Hongqiao SHA", q, flags=re.I)
        if "hongqiao" not in q.lower():
            q = f"{q} Shanghai Hongqiao"
        q = f"{q} -Pudong -PVG"
    if not re.search(r"\b(20\d{2}|fy\s*\d{2})\b", q, re.I):
        q = f"{q} {current_year}"
    return q


def openai_tools(*, enable_genie: bool = True) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "wren_get_instructions",
                "description": "Load Internal semantic / metric instructions before writing SQL.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "wren_describe_schema",
                "description": "Describe gold-table schema for SQL drafting.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string", "description": "Optional model/table name"}
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "wren_dry_run",
                "description": "Validate SQL without returning rows.",
                "parameters": {
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                    "required": ["sql"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "wren_run_sql",
                "description": "Execute read-only SQL against Internal gold tables (row-capped).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string"},
                        "title": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["sql"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search public web sources for PEER airport own-stats / industry news. "
                    "Always include the target year in the query (prefer current year from RUNTIME FACTS). "
                    "Do NOT use this for the hub internal KPIs when Internal SQL is available."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "chart_plan",
                "description": (
                    "Plan an in-response ECharts chart from a prior SQL table. "
                    "Use combo + dual_y for mixed units (e.g. passengers vs cargo). "
                    "Use stacked_bar for composition; area for filled trends. "
                    "Peer airport ranked bars are rebuilt by the app — skip chart_plan for those."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["line", "bar", "pie", "area", "stacked_bar", "combo"],
                        },
                        "title": {"type": "string"},
                        "x": {"type": "string"},
                        "y": {"type": "array", "items": {"type": "string"}},
                        "dual_y": {
                            "type": "boolean",
                            "description": "Put later y series on a right-hand axis",
                        },
                        "y_axes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional [left_axis_name, right_axis_name]",
                        },
                        "series_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["line", "bar", "area"]},
                            "description": "Per-y series type (especially for combo)",
                        },
                        "y_axis_index": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Per-y axis index 0|1",
                        },
                        "table_index": {
                            "type": "integer",
                            "description": "0-based index into gathered tables",
                        },
                    },
                    "required": ["type", "x", "y"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    if enable_genie:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "genie_ask",
                    "description": "Optional fallback NL->SQL helper for hard/open-ended the hub asks.",
                    "parameters": {
                        "type": "object",
                        "properties": {"question": {"type": "string"}},
                        "required": ["question"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools


def _rows_as_objects(columns: list[str], rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append({str(c): row.get(c) for c in columns})
        elif isinstance(row, (list, tuple)):
            item = {str(columns[i]): row[i] if i < len(row) else None for i in range(len(columns))}
            out.append(item)
        else:
            out.append({str(columns[0]) if columns else "value": row})
    return out


def _as_table(raw: Any, title: str = "Query result") -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        if "columns" in raw and "rows" in raw:
            cols = [str(c) for c in list(raw.get("columns") or [])]
            raw_rows = list(raw.get("rows") or [])
            # Fallback MCP server returns rows as objects; normalize to arrays
            # aligned with columns so chart/evidence consumers see one shape.
            rows = [
                [row.get(c) for c in cols] if isinstance(row, dict) else row
                for row in raw_rows
            ]
            return {
                "title": str(raw.get("title") or title),
                "columns": cols,
                "rows": rows[:CHAT_TABLE_MAX_ROWS],
            }
        # common Wren shapes
        if "data" in raw and isinstance(raw["data"], list):
            data = raw["data"]
            if data and isinstance(data[0], dict):
                cols = list(data[0].keys())
                rows = [[row.get(c) for c in cols] for row in data]
                return {"title": title, "columns": cols, "rows": rows[:CHAT_TABLE_MAX_ROWS]}
        if "records" in raw and isinstance(raw["records"], list):
            data = raw["records"]
            if data and isinstance(data[0], dict):
                cols = list(data[0].keys())
                rows = [[row.get(c) for c in cols] for row in data]
                return {"title": title, "columns": cols, "rows": rows[:CHAT_TABLE_MAX_ROWS]}
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        cols = list(raw[0].keys())
        rows = [[row.get(c) for c in cols] for row in raw]
        return {"title": title, "columns": cols, "rows": rows[:CHAT_TABLE_MAX_ROWS]}
    return None


class ToolRuntime:
    """Stateful tool executor collecting tables/charts/sources/status."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self._timing_origin = time.perf_counter()
        self.sql = DuckDBSQLClient()
        self.wren = self.sql  # compat alias; no Wren MCP in portfolio build
        self.tables: list[dict[str, Any]] = []
        self.charts: list[dict[str, Any]] = []
        self.sources: list[dict[str, Any]] = []
        self.status = {"genie": "skipped", "web": "skipped", "files": "skipped"}
        self.last_sql: str | None = None
        self.timings_ms: dict[str, float] = {
            "sql": 0.0,
            "wren": 0.0,
            "web": 0.0,
            "genie": 0.0,
            "chart": 0.0,
        }
        self.tool_calls: list[dict[str, Any]] = []
        self._sql_runs = 0
        self._sql_refunds = 0
        self._web_runs = 0
        self._genie_runs = 0

    @property
    def tool_wall_ms(self) -> float:
        """Union of recorded intervals; concurrent work counts once.

        Legacy/fake executors may omit intervals. Those calls are skipped rather
        than invented; tool_timeline_complete signals incomplete timing coverage.
        """
        with self._lock:
            intervals = sorted(self._valid_tool_intervals())
        total = 0.0
        left = right = None
        for start, end in intervals:
            if left is None:
                left, right = start, end
            elif start <= right:
                right = max(right, end)
            else:
                total += right - left
                left, right = start, end
        return round(total + (right - left if left is not None else 0), 2)

    def _valid_tool_intervals(self) -> list[tuple[float, float]]:
        intervals = []
        for call in self.tool_calls:
            start, end = call.get("start_ms"), call.get("end_ms")
            if (isinstance(start, (int, float)) and isinstance(end, (int, float))
                    and math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
                intervals.append((start, end))
        return intervals

    @property
    def tool_timeline_complete(self) -> bool:
        with self._lock:
            return len(self._valid_tool_intervals()) == len(self.tool_calls)

    def enable_genie(self) -> bool:
        # Default OFF: dev stack is strictly Wren -> Databricks. Opt back in
        # explicitly with CI_ENABLE_GENIE_FALLBACK=true for hard/open-ended asks.
        flag = (os.getenv("CI_ENABLE_GENIE_FALLBACK") or "false").strip().lower()
        return flag in {"1", "true", "yes", "on"}

    @staticmethod
    def _bucket_for_tool(name: str) -> str:
        if name in {"wren_get_instructions", "wren_describe_schema", "wren_dry_run"}:
            return "wren"
        if name == "wren_run_sql":
            return "sql"
        if name == "web_search":
            return "web"
        if name == "genie_ask":
            return "genie"
        if name == "chart_plan":
            return "chart"
        return "other"

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a tool.

        web_search is parallel-safe: network IO runs outside the lock so multiple
        peer searches can overlap. SQL / chart / genie stay serialized.
        """
        from app.local_react.langsmith_trace import attach_outputs, tool_span

        self._local.invocation_start = time.perf_counter()
        args = arguments or {}
        safe_inputs: dict[str, Any] = {"tool": name}
        if isinstance(args.get("query"), str):
            safe_inputs["query"] = args["query"][:240]
        if isinstance(args.get("sql"), str):
            safe_inputs["sql"] = args["sql"][:500]
        if isinstance(args.get("question"), str):
            safe_inputs["question"] = args["question"][:240]

        with tool_span(
            name,
            inputs=safe_inputs,
            metadata={"bucket": self._bucket_for_tool(name)},
        ) as run:
            if name == "web_search":
                result = self._execute_web_parallel(args)
            else:
                with self._lock:
                    result = self._execute_unlocked(name, args)
            attach_outputs(
                run,
                {
                    "ok": bool(isinstance(result, dict) and result.get("ok", True)),
                    "elapsed_ms": (result.get("elapsed_ms") if isinstance(result, dict) else None),
                    "error": (result.get("error") if isinstance(result, dict) else None),
                    "row_count": (
                        len(result.get("rows") or [])
                        if isinstance(result, dict) and isinstance(result.get("rows"), list)
                        else None
                    ),
                },
            )
            return result

    def _record_tool(
        self,
        *,
        name: str,
        bucket: str,
        elapsed_ms: float,
        result: dict[str, Any],
        started_at: float | None = None,
    ) -> dict[str, Any]:
        if bucket in self.timings_ms:
            self.timings_ms[bucket] = round(self.timings_ms.get(bucket, 0.0) + elapsed_ms, 1)
        end = time.perf_counter()
        start = started_at if started_at is not None else end - elapsed_ms / 1000
        invocation = getattr(self._local, "invocation_start", start)
        sql_val = None
        query_val = None
        err_val = None
        if isinstance(result, dict):
            sql_val = result.get("sql") or self.last_sql
            query_val = result.get("query")
            err_val = result.get("error") or result.get("wren_error")
        self.tool_calls.append(
            {
                "name": name,
                "bucket": bucket,
                "ms": round(elapsed_ms, 1),
                "start_ms": round((start - self._timing_origin) * 1000, 2),
                "end_ms": round((end - self._timing_origin) * 1000, 2),
                "queue_ms": round(max(0.0, start - invocation) * 1000, 2),
                "observability": (result.get("_ci_observability", {}) if isinstance(result, dict) else {}),
                "backend": (result.get("backend") or result.get("source")) if isinstance(result, dict) else None,
                "ok": bool(isinstance(result, dict) and result.get("ok", True)),
                "sql": sql_val,
                "query": query_val,
                "error": (str(err_val)[:500] if err_val else None),
            }
        )
        out = dict(result) if isinstance(result, dict) else {"ok": False, "error": "bad result"}
        out.pop("_ci_observability", None)
        out["elapsed_ms"] = round(elapsed_ms, 1)
        return out

    def _execute_web_parallel(self, args: dict[str, Any]) -> dict[str, Any]:
        """True parallel web: reserve a slot under lock, fetch outside, merge under lock."""
        t0 = time.perf_counter()
        with self._lock:
            cap = _tavily_max_searches(8)
            if self._web_runs >= cap:
                return {
                    "ok": False,
                    "error": f"web search cap reached for this turn ({cap})",
                    "elapsed_ms": 0.0,
                }
            self._web_runs += 1

        try:
            fetched = self._web_search_fetch(
                str(args.get("query") or ""),
                max_results=int(args.get("max_results") or _tavily_max_results(5)),
            )
        except Exception as exc:  # noqa: BLE001
            fetched = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            if fetched.get("ok"):
                self.status["web"] = "ok"
                for src in list(fetched.get("sources") or []):
                    if not isinstance(src, dict):
                        continue
                    url = str(src.get("url") or "") or None
                    label = _safe_source_label(
                        str(src.get("label") or "public source"), url
                    )
                    self.sources.append(
                        {
                            "type": "public",
                            "label": label,
                            "url": url,
                            "cite_tag": _cite_tag_for(label, url),
                        }
                    )
            elif self.status.get("web") != "ok":
                self.status["web"] = "error" if fetched.get("error") else "skipped"
            return self._record_tool(
                name="web_search",
                bucket="web",
                elapsed_ms=elapsed_ms,
                result=fetched,
                started_at=t0,
            )

    def _execute_unlocked(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = arguments or {}
        bucket = self._bucket_for_tool(name)
        t0 = time.perf_counter()
        if name.startswith("wren_"):
            pass  # no Wren MCP
            self.sql.last_observability = {"transport": "duckdb", "stages": []}
        try:
            if name == "wren_get_instructions":
                result = self._wren_or_fallback_instructions()
            elif name == "wren_describe_schema":
                result = self._wren_or_fallback_schema(str(args.get("model") or "") or None)
            elif name == "wren_dry_run":
                result = self._dry_run(str(args.get("sql") or ""))
            elif name == "wren_run_sql":
                result = self._run_sql(
                    str(args.get("sql") or ""),
                    title=str(args.get("title") or "Query result"),
                    limit=int(args.get("limit") or CHAT_TABLE_MAX_ROWS),
                )
                # Attribute run_sql to wren when sidecar served it; else keep sql bucket.
                if isinstance(result, dict) and result.get("source") == "wren":
                    bucket = "wren"
            elif name == "web_search":
                # Prefer _execute_web_parallel; keep as fallback if called unlocked.
                result = self._web_search(
                    str(args.get("query") or ""),
                    max_results=int(args.get("max_results") or _tavily_max_results(5)),
                )
            elif name == "chart_plan":
                result = self._chart_plan(args)
            elif name == "genie_ask":
                result = self._genie_ask(str(args.get("question") or ""))
            else:
                result = {"ok": False, "error": f"Unknown tool: {name}"}
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if name.startswith("wren_") and isinstance(result, dict):
            attempts = [{}, getattr(self.sql, "last_observability", {})]
            result["_ci_observability"] = {"attempts": [a for a in attempts if a.get("stages")]}
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._record_tool(
            name=name,
            bucket=bucket,
            elapsed_ms=elapsed_ms,
            result=result if isinstance(result, dict) else {"ok": False, "error": "bad result"},
            started_at=t0,
        )

    def _wren_or_fallback_instructions(self) -> dict[str, Any]:
        return {
            "ok": True,
            "source": "duckdb",
            "instructions": {
                "metric_rules": [
                    "YTD = completed months only (use year_month bounds)",
                    "cargo_tonnes is already tonnes — never invent bare 't'",
                    "aircraft_movements = ATM / flights handled",
                    "Internal hub KPIs come from hub_kpi_monthly",
                    "Peer / non-Aether airport own stats = web_search only",
                    "Retail revenue is USD on hub_retail_monthly",
                    "On-time performance is on_time_pct on hub_otp_monthly",
                ],
                "gold_tables": [
                    "hub_kpi_monthly",
                    "route_kpi_monthly",
                    "hub_retail_monthly",
                    "hub_otp_monthly",
                ],
                "hubs": [
                    "NVA = Nova International",
                    "PRH = Pacific Rim Hub",
                    "ATG = Atlantic Gate",
                    "SKY = Skyline Central",
                ],
            },
        }

    def _wren_or_fallback_schema(self, model: str | None) -> dict[str, Any]:
        return {
            "ok": True,
            "source": "duckdb",
            "schema": {
                "note": "DuckDB local warehouse. year_month is DATE-like text YYYY-MM-01.",
                "model": model,
                "tables": [
                    "hub_kpi_monthly",
                    "route_kpi_monthly",
                    "hub_retail_monthly",
                    "hub_otp_monthly",
                ],
                "canonical_columns": {
                    "hub_kpi_monthly": [
                        "hub_code",
                        "hub_name",
                        "year_month",
                        "passengers",
                        "cargo_tonnes",
                        "aircraft_movements",
                    ],
                    "route_kpi_monthly": [
                        "origin_hub",
                        "dest_code",
                        "dest_name",
                        "year_month",
                        "passengers",
                        "cargo_tonnes",
                    ],
                    "hub_retail_monthly": [
                        "hub_code",
                        "year_month",
                        "category",
                        "revenue_usd",
                    ],
                    "hub_otp_monthly": [
                        "hub_code",
                        "year_month",
                        "on_time_pct",
                        "avg_delay_minutes",
                    ],
                },
            },
        }

    def _dry_run(self, sql: str) -> dict[str, Any]:
        if not sql.strip():
            return {"ok": False, "error": "sql is required"}
        sql = _rewrite_common_column_aliases(sql)
        # EXPLAIN-only validation against DuckDB
        try:
            probe = self.sql.run_sql(f"EXPLAIN {sql}", limit=5)
            if probe.get("ok"):
                return {"ok": True, "source": "duckdb", "result": {"valid": True}}
            return {"ok": False, "error": probe.get("error") or "dry-run failed", "sql": sql}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "sql": sql}


    def _refund_sql_run(self, err: str | None = None) -> bool:
        """Undo one billable SQL slot for engine/parse failures (limited)."""
        if self._sql_refunds >= _SQL_REFUND_CAP or self._sql_runs <= 0:
            return False
        if err is not None and not _looks_like_sql_engine_error(err):
            return False
        self._sql_runs -= 1
        self._sql_refunds += 1
        return True

    def _run_sql(self, sql: str, title: str = "Query result", limit: int = 20) -> dict[str, Any]:
        if not sql.strip():
            return {"ok": False, "error": "sql is required", "sql": sql}
        sql = _rewrite_common_column_aliases(sql)
        mixed = _mixed_grain_union_message(sql)
        if mixed:
            return {"ok": False, "error": mixed, "sql": sql}
        if self._sql_runs >= _SQL_TURN_CAP:
            return {"ok": False, "error": "SQL run cap reached for this turn", "sql": sql}
        self._sql_runs += 1
        self.last_sql = sql
        try:
            raw = self.sql.run_sql(sql, limit=limit)
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc)[:800]
            self.status["genie"] = "insufficient"
            refunded = self._refund_sql_run(err_msg)
            return {
                "ok": False,
                "error": err_msg,
                "source": "duckdb",
                "sql": sql,
                "sql_cap_refunded": refunded,
            }
        if not isinstance(raw, dict) or not raw.get("ok"):
            err_msg = str((raw or {}).get("error") if isinstance(raw, dict) else raw)[:800]
            self.status["genie"] = "insufficient"
            refunded = self._refund_sql_run(err_msg)
            return {
                "ok": False,
                "error": err_msg or "DuckDB SQL failed",
                "source": "duckdb",
                "sql": sql,
                "sql_cap_refunded": refunded,
            }
        table = _as_table(raw.get("table") or raw, title=title)
        if table is None:
            self.status["genie"] = "insufficient"
            err_msg = "Could not parse SQL result into a table"
            refunded = self._refund_sql_run(err_msg)
            return {
                "ok": False,
                "error": err_msg,
                "source": "duckdb",
                "sql": sql,
                "sql_cap_refunded": refunded,
                "raw": raw,
            }
        self.tables.append(table)
        table_index = len(self.tables) - 1
        self.status["genie"] = "ok"
        if not any(s.get("label") == "Internal" for s in self.sources):
            self.sources.append(
                {
                    "type": "internal",
                    "label": "Internal",
                    "url": None,
                    "cite_tag": "[Internal]",
                }
            )
        return {
            "ok": True,
            "source": "duckdb",
            "sql": sql,
            "table": table,
            "table_index": table_index,
            "row_count": len(table.get("rows") or []),
        }


    def _maybe_baggage_window(self, sql: str, table: dict[str, Any]) -> None:
        return

    def _web_search_fetch(self, query: str, max_results: int = 5) -> dict[str, Any]:
        """Network-only web search (no shared-state mutation). Safe for thread pool."""
        if not query.strip():
            return {"ok": False, "error": "query is required"}

        from app.local_react.runtime_facts import build_runtime_facts

        facts = build_runtime_facts()
        q = bias_web_query(query, current_year=str(facts["current_year"]))

        text = ""
        sources: list[dict[str, str]] = []
        backend = "rest"
        attempts: list[dict[str, Any]] = []
        mcp_url = (os.getenv("TAVILY_MCP_URL") or "").strip()
        mcp_enabled = (os.getenv("TAVILY_MCP_ENABLED") or "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if mcp_url and mcp_enabled:
            try:
                from app.local_react.duckdb_client import DuckDBSQLClient

                client = WrenMCPClient(url=mcp_url)
                try:
                    raw = client.call_tool(
                        "tavily_search", {"query": q, "max_results": max_results}
                    )
                finally:
                    attempts.append(client.last_observability)
                    client.close()
                if isinstance(raw, dict) and raw.get("ok"):
                    backend = "mcp"
                    text = str(raw.get("text") or "")
                    sources = list(raw.get("sources") or [])
                elif isinstance(raw, dict) and raw.get("error"):
                    raise RuntimeError(str(raw.get("error")))
            except Exception:  # noqa: BLE001
                backend = "rest_after_mcp_error"

        if not text:
            from dualsource.tools.tavily import TavilyClient

            rest_attempt: dict[str, Any] = {"transport": "tavily_rest", "stages": []}
            attempts.append(rest_attempt)
            try:
                with timed_stage(rest_attempt["stages"], "tavily.rest_search") as stage:
                    result = TavilyClient().search(q, max_results=max_results)
                    stage["ok"] = result.status == "ok"
            except Exception as exc:
                return {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "query": q,
                    "backend": backend,
                    "_ci_observability": {"attempts": attempts},
                }
            if result.status != "ok":
                return {
                    "ok": False,
                    "error": result.text,
                    "status": result.status,
                    "query": q,
                    "backend": backend,
                    "_ci_observability": {"attempts": attempts},
                }
            text = result.text
            sources = list(result.sources or [])
            backend = "rest" if backend == "rest" else backend

        return {
            "ok": True,
            "text": text,
            "sources": sources,
            "query": q,
            "backend": backend,
            "sources_n": len(sources),
            "_ci_observability": {"attempts": attempts},
        }

    def _web_search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        """Serial web path (caller already holds lock). Prefer execute()/parallel path."""
        cap = _tavily_max_searches(8)
        if self._web_runs >= cap:
            return {"ok": False, "error": f"web search cap reached for this turn ({cap})"}
        self._web_runs += 1
        fetched = self._web_search_fetch(query, max_results=max_results)
        if fetched.get("ok"):
            self.status["web"] = "ok"
            for src in list(fetched.get("sources") or []):
                if not isinstance(src, dict):
                    continue
                url = str(src.get("url") or "") or None
                label = _safe_source_label(
                    str(src.get("label") or "public source"), url
                )
                self.sources.append(
                    {
                        "type": "public",
                        "label": label,
                        "url": url,
                        "cite_tag": _cite_tag_for(label, url),
                    }
                )
        else:
            self.status["web"] = "error" if fetched.get("error") else "skipped"
        return fetched

    def _chart_plan(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self.tables:
            return {"ok": False, "error": "No tables available; run SQL first"}
        idx = int(args.get("table_index") or 0)
        if idx < 0 or idx >= len(self.tables):
            idx = len(self.tables) - 1
        chart = build_chart_from_plan(args, self.tables[idx])
        if not chart:
            # fallback heuristic from existing charts helper when possible
            try:
                from dualsource.tools.charts import maybe_chart_from_table

                chart = maybe_chart_from_table(
                    self.tables[idx], prefer=str(args.get("type") or "auto")
                )
            except Exception:
                chart = None
        if not chart:
            return {"ok": False, "error": "Could not build chart from plan/table"}
        chart["index"] = len(self.charts)
        self.charts.append(chart)
        return {"ok": True, "chart_index": chart["index"], "chart": {"title": chart.get("title"), "type": chart.get("type")}}

    def _genie_ask(self, question: str) -> dict[str, Any]:
        if not self.enable_genie():
            return {"ok": False, "error": "NL SQL fallback disabled in DualSource portfolio build"}
        if not question.strip():
            return {"ok": False, "error": "question is required"}
        if self._genie_runs >= 1:
            return {"ok": False, "error": "genie_ask already used this turn"}
        self._genie_runs += 1
        try:
            from dualsource.tools.genie import GenieClient
        except Exception as exc:  # noqa: BLE001
            # TODO: wire richer Genie reuse (conversation_id) when needed.
            return {"ok": False, "error": f"Genie client unavailable: {exc}"}
        result = GenieClient().ask(question)
        self.status["genie"] = result.status if result.status in {"ok", "insufficient", "error"} else "error"
        if result.tables:
            for t in result.tables:
                self.tables.append(
                    {
                        "title": str(t.get("title") or "Query result"),
                        "columns": list(t.get("columns") or []),
                        "rows": list(t.get("rows") or [])[:CHAT_TABLE_MAX_ROWS],
                    }
                )
        if result.charts:
            self.charts.extend(list(result.charts))
        if result.status == "ok" and not any(s.get("label") == "Internal" for s in self.sources):
            self.sources.append(
                {
                    "type": "internal",
                    "label": "Internal",
                    "url": None,
                    "cite_tag": "[Internal]",
                }
            )
        return {
            "ok": result.status == "ok",
            "status": result.status,
            "text": result.text,
            "tables": len(result.tables or []),
            "conversation_id": result.conversation_id,
        }
