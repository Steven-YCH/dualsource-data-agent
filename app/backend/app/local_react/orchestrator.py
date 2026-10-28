"""Local Streaming ReAct orchestrator (Foundry + Wren/SQL + Tavily + chart)."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterator

from app.local_react.chart_plan import (
    build_bar_chart_from_pairs,
    build_chart_from_plan,
    normalize_airport_label,
)
from app.local_react.foundry_client import FoundryClient, FoundryError
from app.local_react.progress import stage_event, tool_event, tool_label
from app.local_react.prompts import SYSTEM_REACT, SYNTH_SYSTEM
from app.local_react.runtime_facts import build_runtime_facts, runtime_facts_text
from app.local_react.tools import ToolRuntime, openai_tools
from app.local_react.langsmith_trace import (
    configure_langsmith,
    current_run_ids,
    wrap_generator,
)

MAX_TOOL_ROUNDS = int(os.getenv("CI_LOCAL_REACT_MAX_ROUNDS") or "6")

_KPI_HINT = re.compile(
    r"\b(ytd|cargo|passenger|pax|atm|baggage|tonne|traffic|flight|monthly|trend|compare)\b",
    re.IGNORECASE,
)
_PEER_HINT = re.compile(
    r"\b(changi|taipei|narita|incheon|shanghai|pudong|hongqiao|peer|compar(?:e|ed|ison|isons)|"
    r"versus|vs\.?|airports including)\b",
    re.IGNORECASE,
)
_H1_HINT = re.compile(r"\b(first half|\bh1\b)\b", re.IGNORECASE)
_CARGO_HINT = re.compile(r"\b(cargo|freight|tonnes?)\b", re.IGNORECASE)
_PAX_HINT = re.compile(r"\b(passenger|pax)\b", re.IGNORECASE)
_YTD_HINT = re.compile(r"\b(ytd|year[- ]to[- ]date)\b", re.IGNORECASE)
_CHART_HINT = re.compile(
    r"\b(chart|bar\s*chart|bart\s*chart|graph|plot|visuali[sz]e|visualisation|visualization)\b",
    re.IGNORECASE,
)
_MONTHLY_BREAKDOWN_HINT = re.compile(
    r"\b((?:break|split|show|give).{0,40}by\s+month|by\s+month|"
    r"month[- ]by[- ]month|monthly\s+(?:break(?:down)?|trend|series|chart|view))\b",
    re.IGNORECASE,
)
_NEW_DATA_HINT = re.compile(
    r"\b(weekly|daily|monthly|by\s+week|by\s+day|by\s+month|ytd|year[- ]to[- ]date|"
    r"first\s+half|second\s+half|\bh[12]\b|\bq[1-4]\b|quarter|"
    r"breakdown|split\s+by|compar(?:e|ed|ison|isons)|versus|\bvs\.?\b|another|instead)\b",
    re.IGNORECASE,
)
_NUMBERISH = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:m|million)?", re.I)


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        data = json.loads(str(raw))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _history_messages(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for turn in history or []:
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or "")
        if role not in {"user", "assistant"} or not content:
            continue
        out.append({"role": role, "content": content})
    return out[-12:]


def _prior_artifacts(
    history: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Latest assistant structured tables/sources from client_history."""
    tables: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for turn in reversed(history or []):
        if str(turn.get("role") or "").strip().lower() != "assistant":
            continue
        for table in turn.get("tables") or []:
            if isinstance(table, dict) and (table.get("columns") or table.get("rows")):
                tables.append(table)
        for src in turn.get("sources") or []:
            if isinstance(src, dict):
                sources.append(src)
        if tables or sources:
            break
    return tables, sources


def _inherit_kpi_ask(question: str, history: list[dict[str, Any]] | None) -> bool:
    if _looks_like_kpi_question(question):
        return True
    q = (question or "").strip()
    if not history or not q:
        return False
    # Short / chart follow-ups inherit prior KPI intent from recent user turns.
    if len(q) > 140 and not _looks_like_chart_question(q):
        return False
    for turn in reversed((history or [])[-8:]):
        if str(turn.get("role") or "").strip().lower() != "user":
            continue
        if _looks_like_kpi_question(str(turn.get("content") or "")):
            return True
    return False


def _looks_like_monthly_breakdown(question: str) -> bool:
    """the hub monthly series follow-up — not another peer ranking."""
    return bool(_MONTHLY_BREAKDOWN_HINT.search(question or ""))


def _inherit_peer_ask(question: str, history: list[dict[str, Any]] | None) -> bool:
    """Short chart/period follow-ups keep the prior peer-compare intent."""
    if _looks_like_peer_question(question):
        return True
    # "Break this down by month" after a peer ask means the hub monthly Internal series,
    # not a rebuilt peer bar with reused/changed public totals.
    if _looks_like_monthly_breakdown(question):
        return False
    q = (question or "").strip()
    if not history or not q:
        return False
    if len(q) > 160 and not (
        _looks_like_chart_question(q) or _H1_HINT.search(q) or _YTD_HINT.search(q)
    ):
        return False
    for turn in reversed((history or [])[-8:]):
        if str(turn.get("role") or "").strip().lower() != "user":
            continue
        if _looks_like_peer_question(str(turn.get("content") or "")):
            return True
    return False


def _can_reuse_prior_for_chart(
    question: str, prior_tables: list[dict[str, Any]]
) -> bool:
    if not prior_tables or not _looks_like_chart_question(question):
        return False
    # New grain/period/compare needs fresh SQL/web even when chart words are present.
    # "first half comparison by bar chart" must NOT reuse a prior YTD peer table.
    q = question or ""
    if _NEW_DATA_HINT.search(q) or _H1_HINT.search(q) or _YTD_HINT.search(q):
        return False
    return True


def _preferred_chart_type(question: str, *, peer_ask: bool = False) -> str:
    q = (question or "").lower()
    if "pie" in q:
        return "pie"
    if "line" in q:
        return "line"
    if "bar" in q:
        return "bar"
    if _looks_like_monthly_breakdown(question):
        return "line"
    return "bar" if peer_ask else "line"


def _ensure_inline_citation_tags(answer: str) -> str:
    """FE chips only fire on [Internal] / [Label](url) — not bare Internal text."""
    if not answer:
        return answer
    # Bare trailing/prose Internal -> bracket tag the renderer understands.
    fixed = re.sub(r"(?<!\[)\bInternal\b(?!\])", "[Internal]", answer)
    return fixed


def _dedupe_chart_markers(answer: str) -> str:
    """Keep the first [[CHART:n]] per index — repeats paint the same chart twice."""
    if not answer or "[[CHART:" not in answer.upper():
        return answer
    seen: set[str] = set()

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in seen:
            return ""
        seen.add(key)
        return match.group(0)

    return re.sub(r"\[\[CHART:(\d+)\]\]", repl, answer, flags=re.I)


def _strip_unsolicited_peer_filler(answer: str, *, peer_ask: bool) -> str:
    """Drop invented peer-absence lines on pure the hub KPI/chart asks."""
    if peer_ask or not answer:
        return answer
    lines: list[str] = []
    for line in answer.splitlines():
        low = line.lower()
        if re.search(
            r"no peer airport|peer airport data (was |were )?(not |never )?provided|"
            r"no peer comparison|without peer|direct comparison.*(peer|airport)|"
            r"peer airports? (were|are) not (provided|available|in (the )?evidence)",
            low,
        ):
            continue
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _pick_chart_table(
    tables: list[dict[str, Any]], chart_type: str
) -> dict[str, Any] | None:
    if not tables:
        return None
    scored: list[tuple[int, dict[str, Any]]] = []
    for table in tables:
        cols = [str(c).lower() for c in (table.get("columns") or [])]
        score = _table_row_count(table)
        if chart_type == "bar" and any("airport" in c for c in cols):
            score += 100
        if any(k in c for c in cols for k in ("week", "flight_date", "date", "year_month", "month")):
            score += 20
        scored.append((score, table))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _auto_chart_from_table(
    table: dict[str, Any], chart_type: str, title: str | None = None
) -> dict[str, Any] | None:
    cols = [str(c) for c in (table.get("columns") or [])]
    if len(cols) < 2:
        return None
    x = next(
        (
            c
            for c in cols
            if any(
                k in c.lower()
                for k in ("week", "date", "month", "year_month", "airport", "flight_date")
            )
        ),
        cols[0],
    )
    y = next(
        (
            c
            for c in cols
            if c != x
            and any(
                k in c.lower()
                for k in ("atm", "passenger", "cargo", "tonne", "total", "million", "count", "pax")
            )
        ),
        None,
    )
    if not y:
        y = next((c for c in cols if c != x), None)
    if not y:
        return None
    return build_chart_from_plan(
        {
            "type": chart_type,
            "title": title or table.get("title") or "Chart",
            "x": x,
            "y": [y],
        },
        table,
    )


def _followups(question: str, tables: list[dict[str, Any]]) -> list[str]:
    base = [
        "Show the latest monthly trend as a chart",
        "What about cargo for the same period?",
        "Compare with the previous year",
    ]
    q = (question or "").lower()
    if "cargo" in q:
        base[1] = "What about passengers for the same period?"
    if tables:
        base[0] = "Break this down by month"
    return base[:3]


def _ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


def _looks_like_kpi_question(question: str) -> bool:
    return bool(_KPI_HINT.search(question or ""))


def _looks_like_peer_question(question: str) -> bool:
    return bool(_PEER_HINT.search(question or ""))


def _looks_like_ytd_question(question: str) -> bool:
    return bool(_YTD_HINT.search(question or ""))


def _table_row_count(table: dict[str, Any]) -> int:
    return len(list(table.get("rows") or []))


def _prefer_authoritative_tables(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the richest monthly series; drop conflicting partial duplicates."""
    if len(tables) <= 1:
        return tables
    return [max(tables, key=_table_row_count)]


def _ytd_month_coverage(tables: list[dict[str, Any]]) -> int:
    best = 0
    for t in tables:
        best = max(best, _table_row_count(t))
    return best


def _looks_like_chart_question(question: str) -> bool:
    return bool(_CHART_HINT.search(question or ""))


def _peer_compare_spec(
    question: str,
    facts: dict[str, str],
    *,
    tables: list[dict[str, Any]] | None = None,
    web_snippets: list[str] | None = None,
) -> dict[str, Any]:
    """Metric + period for a peer chart. Not hardcoded to H1 passengers."""
    q = question or ""
    cargo = bool(_CARGO_HINT.search(q))
    pax = bool(_PAX_HINT.search(q))
    if cargo and not pax:
        metric = "cargo"
        value_key = "cargo_million_tonnes"
        unit = "million tonnes"
        column_needles = ("cargo", "tonne")
    else:
        metric = "passengers"
        value_key = "passengers_millions"
        unit = "million passengers"
        column_needles = ("passenger", "pax")
    period_key = _infer_period_key(q, facts, tables=tables, web_snippets=web_snippets)
    if period_key == "ytd":
        period = facts.get("ytd_window") or "YTD"
    elif period_key == "h1":
        period = facts.get("h1_window") or "H1"
    else:
        period = "the period named in the question"
    return {
        "metric": metric,
        "value_key": value_key,
        "unit": unit,
        "period": period,
        "period_key": period_key,
        "column_needles": column_needles,
    }


def _infer_period_key(
    question: str,
    facts: dict[str, str],
    *,
    tables: list[dict[str, Any]] | None = None,
    web_snippets: list[str] | None = None,
) -> str:
    """Return ytd|h1|other from the question, else from SQL/web evidence."""
    q = question or ""
    if _YTD_HINT.search(q) and not _H1_HINT.search(q):
        return "ytd"
    if _H1_HINT.search(q):
        return "h1"
    blob = " ".join(
        [
            json.dumps(tables or [], ensure_ascii=False, default=str)[:4000],
            " ".join(web_snippets or [])[:4000],
        ]
    ).lower()
    h1_hits = len(re.findall(r"\bh1\b|first half|january[-– ]june|jan[-– ]jun", blob))
    ytd_hits = len(re.findall(r"\bytd\b|year[- ]to[- ]date|january[-– ]september|jan[-– ]sep", blob))
    if h1_hits > ytd_hits and h1_hits > 0:
        return "h1"
    if ytd_hits > 0:
        return "ytd"
    for table in tables or []:
        for row in table.get("rows") or []:
            label = ""
            if isinstance(row, dict):
                label = str(next(iter(row.values()), ""))
            elif isinstance(row, (list, tuple)) and row:
                label = str(row[0])
            if re.search(r"\bh1\b|first half", label, re.I):
                return "h1"
    return "other"


def _row_cells(row: Any, cols: list[str]) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if isinstance(row, (list, tuple)):
        return {cols[i]: row[i] for i in range(min(len(cols), len(row)))}
    return {}


def _month_token(raw: Any) -> str | None:
    text = str(raw or "").strip()
    m = re.search(r"(20\d{2})[-/](\d{1,2})", text)
    if not m:
        return None
    return f"{m.group(1)}-{int(m.group(2)):02d}"


def _hkia_metric_value(
    tables: list[dict[str, Any]],
    needles: tuple[str, ...],
    *,
    period_key: str = "other",
    facts: dict[str, str] | None = None,
) -> float | None:
    """the hub metric for the asked period only.

    Never sum an H1 row with an H2/YTD remainder. Monthly series are filtered to
    the asked window (H1 = Jan-Jun; YTD = completed months from facts).
    """
    if not tables:
        return None
    facts = facts or {}
    ranked = sorted(
        [t for t in tables if isinstance(t, dict)],
        key=lambda t: (
            0 if str(t.get("source_kind") or "") == "peer_compare" else 1,
            _table_row_count(t),
        ),
        reverse=True,
    )
    for table in ranked:
        cols = [str(c) for c in (table.get("columns") or [])]
        rows = list(table.get("rows") or [])
        if not cols or not rows:
            continue
        y_col = next((c for c in cols if any(n in c.lower() for n in needles)), None)
        if not y_col:
            continue
        period_col = next(
            (
                c
                for c in cols
                if c != y_col
                and any(k in c.lower() for k in ("period", "window", "half", "label"))
            ),
            None,
        )
        month_col = next(
            (
                c
                for c in cols
                if any(k in c.lower() for k in ("year_month", "month", "date"))
            ),
            None,
        )
        values: list[float] = []
        for row in rows:
            cells = _row_cells(row, cols)
            if period_key in {"h1", "ytd"} and period_col:
                label = str(cells.get(period_col) or "")
                if period_key == "h1" and not re.search(
                    r"\bh1\b|first half|jan(?:uary)?\s*[-–/]\s*jun", label, re.I
                ):
                    continue
                if period_key == "ytd" and re.search(
                    r"\bh1\b|\bh2\b|first half|second half", label, re.I
                ):
                    continue
            if period_key == "h1" and month_col:
                ym = _month_token(cells.get(month_col))
                if ym:
                    month = int(ym.split("-")[1])
                    if month > 6:
                        continue
            raw = cells.get(y_col)
            try:
                values.append(float(str(raw).replace(",", "")))
            except Exception:
                continue
        if not values:
            continue
        # Period-labeled rows (H1/H2/YTD) are alternatives, never add them together.
        if period_col and period_key in {"h1", "ytd"}:
            total = values[0]
        elif period_col and len(values) > 1:
            total = values[0]
            for row in rows:
                cells = _row_cells(row, cols)
                label = str(cells.get(period_col) or "")
                if re.search(r"\bh1\b|first half", label, re.I):
                    try:
                        total = float(str(cells.get(y_col)).replace(",", ""))
                        break
                    except Exception:
                        pass
        else:
            total = sum(values)
        if total <= 0:
            continue
        if total >= 1000:
            return round(total / 1_000_000.0, 3)
        return round(total, 3)
    return None


def _hkia_total_millions(tables: list[dict[str, Any]]) -> float | None:
    if not tables:
        return None
    table = max(tables, key=_table_row_count)
    cols = [str(c) for c in (table.get("columns") or [])]
    rows = list(table.get("rows") or [])
    if not cols or not rows:
        return None
    y_col = next(
        (c for c in cols if "passenger" in c.lower() or c.lower() in {"total_passengers", "pax"}),
        None,
    )
    if not y_col:
        return None
    total = 0.0
    n = 0
    for row in rows:
        if isinstance(row, dict):
            raw = row.get(y_col)
        elif isinstance(row, (list, tuple)):
            idx = cols.index(y_col)
            raw = row[idx] if idx < len(row) else None
        else:
            raw = None
        try:
            total += float(str(raw).replace(",", ""))
            n += 1
        except Exception:
            continue
    if n < 2:
        return None
    return round(total / 1_000_000.0, 1)


def _extract_peer_compare_pairs(
    foundry: FoundryClient,
    *,
    question: str,
    facts: dict[str, str],
    tables: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    web_snippets: list[str],
) -> list[tuple[str, float]]:
    spec = _peer_compare_spec(
        question, facts, tables=tables, web_snippets=web_snippets
    )
    needles = spec["column_needles"]
    hkia = _hkia_metric_value(
        tables,
        needles if isinstance(needles, tuple) else ("passenger", "pax"),
        period_key=str(spec.get("period_key") or "other"),
        facts=facts,
    )
    value_key = str(spec["value_key"])
    zeap_preview = []
    for table in tables[:3]:
        if not isinstance(table, dict):
            continue
        if str(table.get("source_kind") or "") == "peer_compare":
            continue
        zeap_preview.append(
            {
                "title": table.get("title"),
                "columns": table.get("columns"),
                "rows": list(table.get("rows") or [])[:12],
            }
        )
    evidence = {
        "runtime_facts": facts,
        "period": spec["period"],
        "period_key": spec.get("period_key"),
        "metric": spec["metric"],
        "unit": spec["unit"],
        "internal_hub_value": hkia,
        "internal_tables": zeap_preview,
        "web_snippets": web_snippets[:8],
        "sources": [
            {"label": s.get("label"), "url": s.get("url")} for s in (sources or [])[:12]
        ],
    }
    example = (
        '[{"airport":"ICN - Seoul Incheon","' + value_key + '":38.4,"coverage":"H1 2026"},'
        '{"airport":"NVA - Nova International","' + value_key + '":32.7,"coverage":"exact"}]'
    )
    prompt = (
        f"Extract {spec['metric']} totals in {spec['unit']} for airports mentioned in the question.\n"
        f"Asked period: {spec['period']} (period_key={spec.get('period_key')}). Keep the metric; do not invent numbers.\n"
        f"Return ONLY JSON array like {example}.\n"
        "Use labels like 'SHA - Shanghai Hongqiao', 'ICN - Seoul Incheon', "
        "'SIN - Singapore Changi', 'NVA - Nova International', 'TPE - Taipei Taoyuan', "
        "'NRT - Tokyo Narita' when possible. SHA means Hongqiao unless the user said Pudong/PVG.\n"
        "the hub must use internal_hub_value when present. "
        "If internal_tables has both H1 and H2 (or YTD) rows, NEVER add them together for an H1 chart. "
        "H1 HKG is the H1 row / Jan-Jun months only; YTD HKG is Jan through last complete month. "
        "A YTD total (~49M) must not be pasted into an H1 answer (~32.7M). "
        "That Internal figure is an the hub system total only if the SQL summed all routes. "
        "Do not treat one airport_code route row as another airport's own total.\n"
        "Include EVERY airport that has a usable numeric figure in the evidence. "
        "Prefer the exact asked period. If only a nearby public window exists "
        "(H1, Q1+Q2 sum when both quarters are stated, Jan-Aug, international-only, "
        "or prior full year), STILL include that airport with the best figure found and "
        "set coverage to that window. Do not omit peers just because coverage differs. "
        "Omit only when no numeric figure exists at all. No markdown.\n\n"
        f"Question: {question}\n\nEvidence:\n"
        + json.dumps(evidence, ensure_ascii=False, default=str)[:9000]
    )
    try:
        text_out = foundry.chat_text(
            model=foundry.fast_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Return only a JSON array of airport {spec['metric']} figures "
                        f"in {spec['unit']}."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )
    except Exception:
        return []
    raw = (text_out or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
    pairs: list[tuple[str, float]] = []
    if not isinstance(data, list):
        return pairs
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("airport") or item.get("name") or "").strip()
        val = item.get(value_key)
        if val is None:
            val = item.get("passengers_millions")
        if val is None:
            val = item.get("cargo_million_tonnes")
        if val is None:
            val = item.get("value")
        try:
            num = float(val)
        except Exception:
            continue
        if name and num > 0:
            pairs.append((name, num))
    # Normalize labels + force Internal the hub figure when available.
    normalized: list[tuple[str, float]] = []
    seen: set[str] = set()
    for name, num in pairs:
        label = normalize_airport_label(name)
        if label in seen:
            continue
        seen.add(label)
        if re.search(r"hong\s*kong|hkia|\bhkg\b", label, re.I) and hkia is not None:
            normalized.append(("NVA - Nova International", hkia))
        else:
            normalized.append((label, num))
    if hkia is not None and "NVA - Nova International" not in {n for n, _ in normalized}:
        normalized.insert(0, ("NVA - Nova International", hkia))
    # Rank high → low for chart + table readability.
    normalized.sort(key=lambda x: x[1], reverse=True)
    return normalized



def _has_web_evidence(sources: list[dict[str, Any]] | None) -> bool:
    """True when public/web tool results exist (not just the Internal label stub)."""
    for src in sources or []:
        if not isinstance(src, dict):
            continue
        label = str(src.get("label") or src.get("title") or "").strip().lower()
        url = str(src.get("url") or "").strip()
        # Internal is the internal citation stub; web evidence needs a real URL/source.
        if label == "zeap":
            continue
        if url or label:
            return True
    return False


def _enforce_accuracy(
    answer: str,
    tables: list[dict[str, Any]],
    question: str,
    sources: list[dict[str, Any]] | None = None,
    *,
    kpi_ask: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """Block invented numbers only when neither SQL rows nor web evidence exist.

    Acceptable evidence for numeric claims:
    - Internal/SQL structured tables (internal the hub KPIs)
    - web_search sources (peer airports / public external stats)
    """
    has_rows = any(list(t.get("rows") or []) for t in tables)
    has_web = _has_web_evidence(sources)
    meta = {
        "tables_authoritative": True,
        "invented_kpi_blocked": False,
        "had_sql_rows": has_rows,
        "had_web_sources": has_web,
    }
    text = (answer or "").strip()
    is_kpi = bool(kpi_ask) if kpi_ask is not None else _looks_like_kpi_question(question)
    if not is_kpi:
        return text, meta
    # SQL rows and/or web sources are both valid grounding. They do not make
    # a mismatched headline true: the window count and the row sums win.
    if has_rows or has_web:
        from app.local_react.answer_guard import align_answer

        aligned, notes = align_answer(text, tables, question)
        meta.update(notes)
        return aligned, meta
    # No SQL and no web: do not allow numeric KPI claims (model hallucination).
    if _NUMBERISH.search(text):
        meta["invented_kpi_blocked"] = True
        honest = (
            "This looks **out of scope** for DualSource Agent.\n\n"
            "I cover the hub Internal analytics (passengers, cargo, ATM, flights, baggage) "
            "and public peer-airport / industry context. I do not have verified Internal "
            "or public sources for this ask, so I will not invent a number.\n\n"
            "Please rephrase toward an airport / traffic KPI, or ask the relevant "
            "commercial / F&B operations team if this is a concession metric."
        )
        return honest, meta
    if not text:
        meta["invented_kpi_blocked"] = True
        return (
            "This looks **out of scope** for DualSource Agent - I could not "
            "retrieve verified Internal or public sources, so I cannot provide a figure."
        ), meta
    return text, meta


class LocalReactOrchestrator:
    def __init__(self, foundry: FoundryClient | None = None) -> None:
        self.foundry = foundry or FoundryClient()

    def handle(
        self,
        *,
        question: str,
        history: list[dict[str, Any]] | None = None,
        user_email: str = "",
        session_id: str = "",
        files_dir: str = "",
    ) -> dict[str, Any]:
        final = None
        for event in self.handle_stream(
            question=question,
            history=history,
            user_email=user_email,
            session_id=session_id,
            files_dir=files_dir,
        ):
            if event.get("type") == "final":
                final = event.get("response")
            elif event.get("type") == "error":
                raise RuntimeError(event.get("detail") or "local_react failed")
        if not isinstance(final, dict):
            raise RuntimeError("local_react produced no final response")
        return final

    def handle_stream(
        self,
        *,
        question: str,
        history: list[dict[str, Any]] | None = None,
        user_email: str = "",
        session_id: str = "",
        files_dir: str = "",
    ) -> Iterator[dict[str, Any]]:
        configure_langsmith()
        gen = self._handle_stream_inner(
            question=question,
            history=history,
            user_email=user_email,
            session_id=session_id,
            files_dir=files_dir,
        )
        yield from wrap_generator(
            "local_react.turn",
            gen,
            run_type="chain",
            inputs={
                "question": (question or "")[:500],
                "history_turns": len(history or []),
                "session_id": session_id,
            },
            metadata={
                "agent_mode": "local_react",
                "harness": (os.getenv("CI_LOCAL_REACT_HARNESS") or "langchain").strip().lower(),
                "fast_model": os.getenv("CI_LLM_MODEL_FAST") or "",
                "strong_model": os.getenv("CI_LLM_MODEL_STRONG") or "",
            },
            tags=["dualsource", "local_react"],
        )

    def _handle_stream_inner(
        self,
        *,
        question: str,
        history: list[dict[str, Any]] | None = None,
        user_email: str = "",
        session_id: str = "",
        files_dir: str = "",
    ) -> Iterator[dict[str, Any]]:
        _ = (user_email, files_dir)
        trace_id = uuid.uuid4().hex
        progress: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        ttfp_ms: float | None = None
        ttft_ms: float | None = None
        synth_ms = 0.0

        def emit_stage(stage: str, status: str) -> dict[str, Any]:
            nonlocal ttfp_ms
            ev = stage_event(stage, status)
            progress.append(ev)
            if ttfp_ms is None:
                ttfp_ms = _ms_since(t0)
                ev = dict(ev)
                ev["ttfp_ms"] = ttfp_ms
            return ev

        # Emit first progress immediately (<2s target for UAT).
        first = emit_stage("route", "start")
        yield first
        yield {
            "type": "meta",
            "trace_id": trace_id,
            "ttfp_ms": ttfp_ms,
            "label": first.get("label"),
        }

        if not self.foundry.ready():
            yield emit_stage("route", "end")
            detail = "AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY not configured for local_react"
            yield {"type": "error", "detail": detail, "trace_id": trace_id}
            return

        runtime = ToolRuntime()
        tools = openai_tools(enable_genie=runtime.enable_genie())
        facts = build_runtime_facts()
        facts_text = runtime_facts_text()
        prior_tables, prior_sources = _prior_artifacts(history)
        kpi_ask = _inherit_kpi_ask(question, history)
        monthly_ask = _looks_like_monthly_breakdown(question)
        chart_ask = _looks_like_chart_question(question) or monthly_ask
        peer_ask = _inherit_peer_ask(question, history) or (
            any(
                "airport" in str(c).lower()
                for t in prior_tables
                for c in (t.get("columns") or [])
            )
            and chart_ask
            and not monthly_ask
        )
        ytd_ask = _looks_like_ytd_question(question)
        reuse_prior_chart = _can_reuse_prior_for_chart(question, prior_tables)
        period_changed = bool(
            _H1_HINT.search(question or "")
            or _YTD_HINT.search(question or "")
            or _NEW_DATA_HINT.search(question or "")
        )
        continuity_note = ""
        if history:
            if reuse_prior_chart:
                continuity_note = (
                    "\n\nConversation continuity: keep the prior metric/period. "
                    "Short chart/visualize follow-ups should reuse prior structured tables."
                )
            elif period_changed:
                continuity_note = (
                    "\n\nConversation continuity: the user changed period, grain, or compare "
                    "scope. Re-query Internal SQL for the hub and web_search each peer for the NEW "
                    "period. Do not reuse a prior YTD total for an H1/first-half ask (or "
                    "vice versa). Still chart and table the best available figures; a "
                    "comparability note does not mean omit the chart."
                )
            else:
                continuity_note = (
                    "\n\nConversation continuity: keep the prior metric unless the user "
                    "clearly changed topic."
                )
            if prior_tables:
                titles = ", ".join(
                    str(t.get("title") or "table") for t in prior_tables[:3]
                )
                continuity_note += f" Prior tables available: {titles}."
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_REACT + "\n\n" + facts_text}
        ]
        messages.extend(_history_messages(history))
        messages.append(
            {
                "role": "user",
                "content": (
                    facts_text
                    + "\nQuestion:\n"
                    + question
                    + continuity_note
                    + (
                        "\n\nRouting reminder: the hub/Hong Kong figures from Internal SQL; "
                        "other airports from web_search with the current year."
                        if peer_ask
                        else ""
                    )
                ),
            }
        )

        yield emit_stage("route", "end")

        sql_open = False
        web_open = False

        try:
            forced_tool_rounds = 0
            explore_calls = 0
            ytd_nudge_sent = False
            web_snippets: list[str] = []
            retrieval_stats: dict[str, Any] = {"model_calls": 0}
            if reuse_prior_chart:
                runtime.tables = list(prior_tables)
                if prior_sources and not runtime.sources:
                    runtime.sources = list(prior_sources)
            harness = (os.getenv("CI_LOCAL_REACT_HARNESS") or "langchain").strip().lower()
            if harness not in {"legacy", "langchain"}:
                raise ValueError("CI_LOCAL_REACT_HARNESS must be legacy or langchain")
            if harness == "langchain" and not reuse_prior_chart:
                try:
                    from app.local_react.langchain_harness import stream_retrieval
                except ImportError as exc:
                    raise RuntimeError(
                        "LangChain harness dependencies missing; install app/backend/requirements-langchain.txt"
                    ) from exc
                retrieval_stream = stream_retrieval(
                    foundry=self.foundry, runtime=runtime, tools=tools, messages=messages,
                    kpi_ask=kpi_ask, peer_ask=peer_ask, ytd_ask=ytd_ask,
                    facts=facts, max_rounds=MAX_TOOL_ROUNDS, coverage=_ytd_month_coverage,
                    web_snippets=web_snippets, session_id=session_id, stats=retrieval_stats,
                )
                try:
                    for ev in retrieval_stream:
                        if ev.get("status") == "start":
                            name = ev.get("tool_name") or ""
                            if name == "web_search" and not web_open:
                                yield emit_stage("web", "start")
                                web_open = True
                            elif name in {
                                "wren_get_instructions",
                                "wren_describe_schema",
                                "wren_dry_run",
                                "wren_run_sql",
                                "genie_ask",
                            } and not sql_open:
                                yield emit_stage("genie", "start")
                                sql_open = True
                        progress.append(ev)
                        yield ev
                finally:
                    retrieval_stream.close()
            for _round in range(0 if reuse_prior_chart or harness == "langchain" else MAX_TOOL_ROUNDS):
                tool_choice: str | dict[str, Any] | None = "auto"
                # Grok/FAST sometimes refuses before tool use; force tools for KPI asks.
                # Round 1: any tool. Round 2 still without SQL: force wren_run_sql.
                # Peer asks: after the hub SQL exists, force web_search if no web yet.
                if peer_ask and runtime.tables and not _has_web_evidence(runtime.sources):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": "web_search"},
                    }
                elif kpi_ask and not runtime.tables and forced_tool_rounds < 2:
                    if forced_tool_rounds == 0:
                        tool_choice = "required"
                    else:
                        tool_choice = {
                            "type": "function",
                            "function": {"name": "wren_run_sql"},
                        }
                    forced_tool_rounds += 1
                retrieval_stats["model_calls"] += 1
                msg = self.foundry.chat(
                    model=self.foundry.fast_model,
                    messages=messages,
                    tools=tools,
                    temperature=0.1,
                    max_tokens=1400,
                    tool_choice=tool_choice,
                )
                tool_calls = msg.get("tool_calls") or []
                content = msg.get("content")
                assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)

                if not tool_calls:
                    # One retry with required if KPI and no tables yet.
                    if kpi_ask and not runtime.tables and forced_tool_rounds < 2:
                        continue
                    break

                # Prepare call list; soft-skip spare explore calls; parallelize web_search.
                prepared: list[dict[str, Any]] = []
                for call in tool_calls:
                    fn = (call.get("function") or {}) if isinstance(call, dict) else {}
                    name = str(fn.get("name") or "")
                    args = _parse_args(fn.get("arguments"))
                    call_id = str(call.get("id") or uuid.uuid4().hex)
                    if (
                        kpi_ask
                        and name
                        in {
                            "wren_get_instructions",
                            "wren_describe_schema",
                            "wren_dry_run",
                        }
                        and explore_calls >= 1
                        and not runtime.tables
                    ):
                        result = {
                            "ok": True,
                            "skipped": True,
                            "note": (
                                "Explore step skipped for latency. "
                                "Call wren_run_sql next on the known gold table."
                            ),
                        }
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": json.dumps(result, ensure_ascii=False)[:12000],
                            }
                        )
                        continue
                    if name in {
                        "wren_get_instructions",
                        "wren_describe_schema",
                        "wren_dry_run",
                    }:
                        explore_calls += 1
                    prepared.append({"id": call_id, "name": name, "args": args})

                web_calls_raw = [c for c in prepared if c["name"] == "web_search"]
                # Drop near-duplicate peer queries (e.g. Shanghai + PVG) so one airport
                # does not burn two search slots / queue behind the worker pool.
                web_calls: list[dict[str, Any]] = []
                web_seen: set[str] = set()
                for item in web_calls_raw:
                    q = str((item.get("args") or {}).get("query") or "").lower()
                    q = re.sub(r"[^a-z0-9\s]+", " ", q)
                    q = re.sub(r"\b(shanghai pudong|pudong|pvg)\b", "pudong", q)
                    q = re.sub(r"\b(shanghai hongqiao|hongqiao|sha)\b", "hongqiao", q)
                    q = re.sub(r"\b(taipei taoyuan|taoyuan|tpe)\b", "taipei", q)
                    q = re.sub(r"\b(changi|sin)\b", "singapore", q)
                    q = re.sub(r"\b(narita|nrt)\b", "narita", q)
                    q = re.sub(r"\b(incheon|icn)\b", "incheon", q)
                    key = " ".join(q.split())
                    if key and key in web_seen:
                        skip = {
                            "ok": True,
                            "skipped": True,
                            "query": (item.get("args") or {}).get("query"),
                            "note": "Duplicate peer web_search skipped",
                        }
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": item["id"],
                                "content": json.dumps(skip, ensure_ascii=False)[:12000],
                            }
                        )
                        continue
                    if key:
                        web_seen.add(key)
                    web_calls.append(item)
                # Explore tools can run in parallel; SQL/chart/genie stay serial.
                explore_names = {
                    "wren_get_instructions",
                    "wren_describe_schema",
                }
                explore_calls_batch = [c for c in prepared if c["name"] in explore_names]
                serial_calls = [
                    c
                    for c in prepared
                    if c["name"] not in explore_names and c["name"] != "web_search"
                ]

                def _tool_start_ev(item: dict[str, Any]) -> dict[str, Any]:
                    name = item["name"]
                    args = item.get("args") or {}
                    detail = None
                    sql = None
                    query = None
                    if name == "wren_run_sql":
                        sql = str(args.get("sql") or "")
                        detail = str(args.get("title") or "Running Internal SQL…")[:160]
                    elif name == "web_search":
                        query = str(args.get("query") or "")
                        detail = (query[:120] + ("…" if len(query) > 120 else "")) or "Searching…"
                    elif name == "chart_plan":
                        detail = str(args.get("title") or args.get("type") or "Building chart…")
                    elif name == "wren_dry_run":
                        sql = str(args.get("sql") or "")
                        detail = "Validating query…"
                    else:
                        detail = tool_label(name)
                    ev = tool_event(
                        tool_id=item["id"],
                        name=name,
                        status="start",
                        detail=detail,
                        sql=sql,
                        query=query,
                    )
                    progress.append(ev)
                    return ev

                def _tool_end_ev(item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
                    name = item["name"]
                    ok = bool(isinstance(result, dict) and result.get("ok", True))
                    rows = None
                    sources_n = None
                    detail = "Done" if ok else str(result.get("error") or result.get("wren_error") or "Failed")[:320]
                    if name == "wren_run_sql" and isinstance(result, dict):
                        table = result.get("table") or {}
                        rows = len(list(table.get("rows") or []))
                        if ok:
                            detail = f"{rows} row{'s' if rows != 1 else ''} returned"
                    if name == "web_search" and isinstance(result, dict):
                        sources_n = int(result.get("sources_n") or len(result.get("sources") or []))
                        if ok:
                            detail = f"{sources_n} source{'s' if sources_n != 1 else ''} found"
                    ev = tool_event(
                        tool_id=item["id"],
                        name=name,
                        status="end",
                        detail=detail,
                        sql=str(result.get("sql") or item.get("args", {}).get("sql") or "") or None,
                        query=str(result.get("query") or item.get("args", {}).get("query") or "")
                        or None,
                        ok=ok,
                        elapsed_ms=float(result.get("elapsed_ms") or 0.0),
                        rows=rows,
                        sources_n=sources_n,
                    )
                    progress.append(ev)
                    return ev

                # Parallel explore (schema/instructions) when model asks several at once.
                if explore_calls_batch:
                    if not sql_open:
                        yield emit_stage("genie", "start")
                        sql_open = True
                    for item in explore_calls_batch:
                        yield _tool_start_ev(item)

                    def _run_explore(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
                        return item, runtime.execute(item["name"], item["args"])

                    with ThreadPoolExecutor(
                        max_workers=min(3, len(explore_calls_batch))
                    ) as pool:
                        futs = [pool.submit(_run_explore, c) for c in explore_calls_batch]
                        for fut in as_completed(futs):
                            item, result = fut.result()
                            yield _tool_end_ev(item, result if isinstance(result, dict) else {})
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": item["id"],
                                    "content": json.dumps(
                                        result, ensure_ascii=False, default=str
                                    )[:12000],
                                }
                            )

                for c in serial_calls:
                    name = c["name"]
                    if name in {
                        "wren_dry_run",
                        "wren_run_sql",
                        "genie_ask",
                    }:
                        if not sql_open:
                            yield emit_stage("genie", "start")
                            sql_open = True
                    elif name == "chart_plan":
                        # Peer compares: orchestrator rebuilds a ranked bar from Internal+web.
                        # Soft-skip LLM chart_plan so Working steps do not show a false Failed.
                        if peer_ask:
                            skip = {
                                "ok": True,
                                "skipped": True,
                                "note": (
                                    "Peer chart will be built from Internal + online sources; "
                                    "skipped intermediate chart_plan."
                                ),
                            }
                            yield _tool_start_ev(c)
                            yield _tool_end_ev(c, skip)
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": c["id"],
                                    "content": json.dumps(
                                        skip, ensure_ascii=False, default=str
                                    )[:12000],
                                }
                            )
                            continue
                        yield {
                            "type": "stage",
                            "stage": "synthesize",
                            "status": "start",
                            "label": "Preparing chart",
                            "ts": time.time(),
                        }
                    yield _tool_start_ev(c)
                    result = runtime.execute(name, c["args"])
                    yield _tool_end_ev(c, result if isinstance(result, dict) else {})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": c["id"],
                            "content": json.dumps(result, ensure_ascii=False, default=str)[:12000],
                        }
                    )

                if web_calls:
                    if sql_open:
                        yield emit_stage("genie", "end")
                        sql_open = False
                    if not web_open:
                        yield emit_stage("web", "start")
                        web_open = True
                    yield {
                        "type": "stage",
                        "stage": "web",
                        "status": "start",
                        "label": "Checking online sources",
                        "detail": f"{len(web_calls)} searches in parallel",
                        "ts": time.time(),
                    }
                    # Emit one card per search BEFORE waiting — LibreChat mid-wait feel.
                    for item in web_calls:
                        yield _tool_start_ev(item)

                    def _run_web(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
                        return item, runtime.execute("web_search", item["args"])

                    # Match per-turn search cap so 5–8 peer queries run truly in parallel.
                    # copy_context keeps LangSmith turn parent across ThreadPool workers.
                    from contextvars import copy_context
                    from app.local_react.tools import _tavily_max_workers

                    workers = min(_tavily_max_workers(), max(1, len(web_calls)))
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futs = [pool.submit(copy_context().run, _run_web, c) for c in web_calls]
                        for fut in as_completed(futs):
                            item, result = fut.result()
                            if isinstance(result, dict) and result.get("text"):
                                web_snippets.append(str(result.get("text") or "")[:600])
                            yield _tool_end_ev(item, result if isinstance(result, dict) else {})
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": item["id"],
                                    "content": json.dumps(
                                        result, ensure_ascii=False, default=str
                                    )[:12000],
                                }
                            )

                # After a round that only explored, nudge FAST to run SQL next.
                if kpi_ask and not runtime.tables and explore_calls and _round + 1 < MAX_TOOL_ROUNDS:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Now run wren_run_sql for this KPI on the known gold table. "
                                "Skip further schema/dry_run unless SQL fails."
                            ),
                        }
                    )

                # YTD coverage repair: if only Q1-ish rows returned, force a full-window SQL.
                expected_months = int(facts.get("expected_ytd_months") or "9")
                if (
                    ytd_ask
                    and runtime.tables
                    and not ytd_nudge_sent
                    and _ytd_month_coverage(runtime.tables) < max(6, expected_months - 1)
                    and _round + 1 < MAX_TOOL_ROUNDS
                ):
                    ytd_nudge_sent = True
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your SQL window is incomplete for YTD. Re-run wren_run_sql for ALL completed "
                                f"months ({facts['ytd_window']}) using year_month >= TIMESTAMP '{facts['ytd_sql_start']}' "
                                f"AND year_month < TIMESTAMP '{facts['ytd_sql_end_exclusive']}'. "
                                f"Expect about {expected_months} monthly rows, then SUM the total."
                            ),
                        }
                    )

                # Peer compares need both the hub SQL and web evidence.
                if (
                    peer_ask
                    and runtime.tables
                    and not _has_web_evidence(runtime.sources)
                    and _round + 1 < MAX_TOOL_ROUNDS
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Now web_search peer airports for the CURRENT year "
                                f"({facts['current_year']}) / first half {facts['current_year']}. "
                                "Do not use 2024 unless no newer public figure exists, and say so if fallback."
                            ),
                        }
                    )

            if sql_open:
                yield emit_stage("genie", "end")
            if web_open:
                yield emit_stage("web", "end")

            # Prefer one authoritative monthly series for YTD conflicts; keep multi-table
            # compare/chart evidence intact for continuity and peer charts.
            if ytd_ask and not chart_ask and not peer_ask:
                runtime.tables = _prefer_authoritative_tables(runtime.tables)
            elif (
                not runtime.tables
                and prior_tables
                and (chart_ask or kpi_ask)
                and not period_changed
            ):
                # Period/compare changes must not revive a prior YTD Internal card as H1.
                runtime.tables = list(prior_tables)

            chart_type = _preferred_chart_type(question, peer_ask=peer_ask)

            # Peer bar/compare: prefer Databricks-style horizontal ranking and replace
            # weak/incomplete LLM chart_plan bars when we extract >=2 airport figures.
            # Monthly breakdown follow-ups must keep the Internal month series, not rebuild
            # a peer ranking (that also rewrote peer totals and dropped the month chart).
            peer_pairs: list[tuple[str, float]] = []
            if (
                peer_ask
                and not monthly_ask
                and _has_web_evidence(runtime.sources)
                and not reuse_prior_chart
            ):
                yield {
                    "type": "stage",
                    "stage": "synthesize",
                    "status": "start",
                    "label": "Preparing chart",
                    "ts": time.time(),
                }
                peer_spec = _peer_compare_spec(
                    question,
                    facts,
                    tables=runtime.tables,
                    web_snippets=web_snippets,
                )
                peer_pairs = _extract_peer_compare_pairs(
                    self.foundry,
                    question=question,
                    facts=facts,
                    tables=runtime.tables,
                    sources=runtime.sources,
                    web_snippets=web_snippets,
                )
                if peer_pairs:
                    value_key = str(peer_spec["value_key"])
                    compare_table = {
                        "title": (
                            f"{peer_spec['period']} {peer_spec['metric']} — "
                            f"ranked ({peer_spec['unit']})"
                        ),
                        "columns": ["airport", value_key],
                        "rows": [[a, v] for a, v in peer_pairs],
                        "source_kind": "peer_compare",
                        "display_source": "Internal + public",
                    }
                    # Keep Internal monthly as continuity evidence, but hide under chart so FE
                    # does not paint two identical "Show data table · 6 rows" cards.
                    prior = []
                    for t in list(runtime.tables):
                        if not isinstance(t, dict):
                            continue
                        tt = dict(t)
                        tt.setdefault("display_source", "Internal")
                        tt["ui_hidden"] = True
                        prior.append(tt)
                    runtime.tables = [compare_table] + prior
                    chart = build_bar_chart_from_pairs(
                        title=compare_table["title"],
                        pairs=peer_pairs,
                        y_name=value_key,
                    )
                    if chart:
                        chart["index"] = 0
                        runtime.charts = [chart]
                    elif not runtime.charts:
                        chart = _auto_chart_from_table(
                            compare_table,
                            "bar",
                            title=compare_table["title"],
                        )
                        if chart:
                            chart["index"] = 0
                            runtime.charts = [chart]

            # Chart from current or prior structured tables (line/bar/pie).
            if chart_ask and runtime.tables and not runtime.charts:
                yield {
                    "type": "stage",
                    "stage": "synthesize",
                    "status": "start",
                    "label": "Preparing chart",
                    "ts": time.time(),
                }
                # Unhide Internal monthly rows that a prior peer turn may have marked ui_hidden.
                if monthly_ask:
                    for t in runtime.tables:
                        if isinstance(t, dict) and t.get("ui_hidden"):
                            t["ui_hidden"] = False
                t0_table = _pick_chart_table(runtime.tables, chart_type) or runtime.tables[0]
                if monthly_ask:
                    # Prefer a year_month series over any leftover peer_compare card.
                    for t in runtime.tables:
                        if not isinstance(t, dict):
                            continue
                        if str(t.get("source_kind") or "") == "peer_compare":
                            continue
                        cols = [str(c).lower() for c in (t.get("columns") or [])]
                        if any("month" in c or c == "date" for c in cols):
                            t0_table = t
                            break
                chart = _auto_chart_from_table(
                    t0_table,
                    chart_type,
                    title=str(
                        t0_table.get("title")
                        or ("the hub monthly passengers" if monthly_ask else "Trend")
                    ),
                )
                if chart:
                    chart["index"] = len(runtime.charts)
                    runtime.charts.append(chart)

            # Progressive chart events: charts are fully built before synthesis.
            # Stream them now so the FE can render at [[CHART:n]] marker positions
            # as answer tokens arrive, instead of waiting for the final payload.
            for i, chart_ev in enumerate(runtime.charts):
                yield {
                    "type": "chart",
                    "index": i,
                    "chart": chart_ev,
                    "ts": time.time(),
                    "trace_id": trace_id,
                }

            yield emit_stage("synthesize", "start")
            synth_t0 = time.perf_counter()

            evidence = {
                "runtime_facts": facts,
                "tables": runtime.tables,
                "charts": [
                    {
                        "index": i,
                        "title": c.get("title"),
                        "type": c.get("type"),
                        "orientation": c.get("orientation"),
                    }
                    for i, c in enumerate(runtime.charts)
                ],
                "peer_compare": [
                    {"airport": a, "value": v, "unit": str(peer_spec["unit"]) if peer_pairs else ""}
                    for a, v in peer_pairs
                ]
                if peer_pairs
                else None,
                "web_snippets": web_snippets[:10],
                "sources": runtime.sources,
                "status": runtime.status,
                "last_sql": runtime.last_sql,
                "session_id": session_id,
                "accuracy_rules": [
                    "Never invent KPI numbers without SQL/tool rows",
                    "Structured tables are authoritative over prose",
                    "Cargo units: tonnes or million tonnes",
                    f"YTD = {facts['ytd_window']} (completed months only)",
                    "the hub figures from Internal tables; peer airport own stats via web only",
                    "Do not answer current-year asks with 2024 unless explicitly disclosed as fallback",
                    "Do not invent YoY / drivers unless present in web_snippets/sources",
                ],
            }
            peer_layout_nudge = (
                "\nThis is a peer/multi-airport compare. Follow the Peer layout in system prompt: "
                "headline → [[CHART:0]] when a chart exists → compact ranked markdown table of "
                "the BEST available figures (include partial-period peers; do not blank them) → "
                "## Key Highlights (one bullet per airport with IATA + figure + coverage) → "
                "⚠️ Comparability note when scopes differ. A comparability note is NOT a reason "
                "to omit the chart or the ranked results."
                if peer_ask
                else ""
            )
            synth_messages = [
                {"role": "system", "content": SYNTH_SYSTEM + "\n\n" + facts_text},
                {
                    "role": "user",
                    "content": (
                        facts_text
                        + chr(10)
                        + "Question: "
                        + question
                        + chr(10)
                        + chr(10)
                        + "Evidence JSON:"
                        + chr(10)
                        + json.dumps(evidence, ensure_ascii=False, default=str)[:16000]
                        + chr(10)
                        + chr(10)
                        + "Write the final markdown answer. Period stated in prose MUST match table months. "
                        + "If the hub table exists, cite [Internal] immediately after each the hub figure — not bare Internal text, "
                        + "and not a public web link for the hub. "
                        + (
                            "For each peer figure cite [domain](url) from sources immediately after that figure "
                            '(short host only, e.g. [changiairport.com](https://...); never page titles with "|"). '
                            if peer_ask
                            else (
                                "This is an the hub-only ask — do NOT mention peer airports, peer comparison, "
                                "or that peer data was missing. "
                            )
                        )
                        + (
                            " This is a monthly breakdown follow-up: lead with the the hub monthly Internal series "
                            "and [[CHART:0]] once; do NOT rebuild a peer ranking or change prior peer totals."
                            if monthly_ask
                            else ""
                        )
                        + (
                            " Chart follow-up: put [[CHART:0]] once after the headline; do not repeat the marker."
                            if chart_ask and not peer_ask
                            else ""
                        )
                        + peer_layout_nudge
                    ),
                },
            ]

            synth_max_tokens = 2400 if peer_ask else 1600
            answer_parts: list[str] = []
            try:
                for delta in self.foundry.stream_text(
                    model=self.foundry.strong_model,
                    messages=synth_messages,
                    temperature=0.2,
                    max_tokens=synth_max_tokens,
                ):
                    if ttft_ms is None and delta:
                        ttft_ms = _ms_since(t0)
                    answer_parts.append(delta)
                    token_ev: dict[str, Any] = {"type": "token", "delta": delta, "trace_id": trace_id}
                    if ttft_ms is not None:
                        token_ev["ttft_ms"] = ttft_ms
                    yield token_ev
            except FoundryError:
                text_out = self.foundry.chat_text(
                    model=self.foundry.strong_model,
                    messages=synth_messages,
                    temperature=0.2,
                    max_tokens=synth_max_tokens,
                )
                if ttft_ms is None and text_out:
                    ttft_ms = _ms_since(t0)
                answer_parts.append(text_out)
                if text_out:
                    yield {
                        "type": "token",
                        "delta": text_out,
                        "trace_id": trace_id,
                        "ttft_ms": ttft_ms,
                    }

            synth_ms = round((time.perf_counter() - synth_t0) * 1000.0, 1)
            answer = "".join(answer_parts).strip()
            if not answer:
                answer = (
                    "I could not produce an answer from the available tools. "
                    "Please retry or rephrase the question."
                )

            answer, accuracy_meta = _enforce_accuracy(
                answer,
                runtime.tables,
                question,
                sources=runtime.sources,
                kpi_ask=kpi_ask,
            )
            answer = _ensure_inline_citation_tags(answer)
            answer = _strip_unsolicited_peer_filler(answer, peer_ask=peer_ask)

            if runtime.charts and "[[CHART:" not in answer:
                markers = " ".join(f"[[CHART:{i}]]" for i in range(len(runtime.charts)))
                answer = answer + chr(10) + chr(10) + markers
            elif not runtime.charts and "[[CHART:" in answer:
                # Synth sometimes invents chart markers when chart_plan failed.
                answer = re.sub(r"\s*\[\[CHART:\d+\]\]", "", answer).strip()
            answer = _dedupe_chart_markers(answer)
            # One chart object per index — duplicate payloads also paint twice in FE.
            if runtime.charts:
                deduped: list[dict[str, Any]] = []
                seen_idx: set[int] = set()
                for i, ch in enumerate(runtime.charts):
                    if not isinstance(ch, dict):
                        continue
                    idx = int(ch.get("index") if ch.get("index") is not None else i)
                    if idx in seen_idx:
                        continue
                    seen_idx.add(idx)
                    ch = dict(ch)
                    ch["index"] = idx
                    deduped.append(ch)
                runtime.charts = deduped

            yield emit_stage("synthesize", "end")
            yield emit_stage("complete", "end")

            total_ms = _ms_since(t0)
            timing = {
                "ttfp_ms": ttfp_ms,
                "ttft_ms": ttft_ms,
                "total_ms": total_ms,
                "harness": harness,
                "retrieval": retrieval_stats,
                "tool_wall_ms": getattr(runtime, "tool_wall_ms", 0.0),
                "tool_timeline_complete": getattr(runtime, "tool_timeline_complete", False),
                "tools_ms": {
                    "sql": runtime.timings_ms.get("sql", 0.0),
                    "wren": runtime.timings_ms.get("wren", 0.0),
                    "web": runtime.timings_ms.get("web", 0.0),
                    "genie": runtime.timings_ms.get("genie", 0.0),
                    "chart": runtime.timings_ms.get("chart", 0.0),
                    "synth": synth_ms,
                },
                "tool_calls": runtime.tool_calls,
            }

            payload = {
                "answer_markdown": answer,
                "sources": runtime.sources,
                "tables": runtime.tables,
                "export_tables": [],
                "charts": runtime.charts,
                "files": [],
                "followups": _followups(question, runtime.tables),
                "status": runtime.status,
                "trace_id": trace_id,
                "progress_events": progress,
                "timing": timing,
                "accuracy": accuracy_meta,
                # Debug payload for local FE debug mode (SQL / tool timings).
                "debug": {
                    "harness": harness,
                    "tool_wall_ms": getattr(runtime, "tool_wall_ms", 0.0),
                    "tool_timeline_complete": getattr(runtime, "tool_timeline_complete", False),
                    "last_sql": runtime.last_sql,
                    "tool_calls": runtime.tool_calls,
                    "timings_ms": {
                        "sql": runtime.timings_ms.get("sql", 0.0),
                        "wren": runtime.timings_ms.get("wren", 0.0),
                        "web": runtime.timings_ms.get("web", 0.0),
                        "genie": runtime.timings_ms.get("genie", 0.0),
                        "chart": runtime.timings_ms.get("chart", 0.0),
                        "synth": synth_ms,
                    },
                    "status": runtime.status,
                    **current_run_ids(),
                },
            }
            yield {
                "type": "final",
                "response": payload,
                "trace_id": trace_id,
                "timing": timing,
            }
        except FoundryError as exc:
            yield {"type": "error", "detail": str(exc), "trace_id": trace_id}
        except Exception as exc:  # noqa: BLE001
            yield {
                "type": "error",
                "detail": f"{type(exc).__name__}: {exc}",
                "trace_id": trace_id,
            }


def run_local_react(**kwargs: Any) -> dict[str, Any]:
    return LocalReactOrchestrator().handle(**kwargs)


def run_local_react_stream(**kwargs: Any) -> Iterator[dict[str, Any]]:
    return LocalReactOrchestrator().handle_stream(**kwargs)
