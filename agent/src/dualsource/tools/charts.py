"""Chart generation from Genie table data.

Produces ECharts-compatible option specs. Supports:
- Line / bar charts (date-like x-axis, one or more numeric series)
- Category bar charts (non-date x-axis like airports / airlines, grouped series)
- Long-format pivot (date + category + single measure -> one series per category)
- Pie / doughnut charts (category x-axis, single numeric column, ≤12 rows)
- Multi-series (all numeric columns become separate series)
"""
from __future__ import annotations

import re
import uuid
from typing import Any

# DualSource palette
_PALETTE = [
    "#003366",  # primary dark
    "#0066A1",  # primary
    "#E68A00",  # accent amber
    "#339966",  # green
    "#CC3333",  # red
    "#6633CC",  # purple
    "#009999",  # teal
    "#CC6699",  # pink
]

_DATE_NAME_RE = re.compile(
    r"(date|time|month|year|week|day|period|ym|year_month|dt)",
    re.IGNORECASE,
)
_DATE_VALUE_RE = re.compile(
    r"^("
    r"\d{4}([-|/]\d{1,2}){1,2}"  # YYYY-MM / YYYY-MM-DD
    r"|\d{4}[-|/]\d{1,2}"  # YYYY-MM
    r"|\d{4}-W\d{1,2}"  # ISO week
    r"|\d{4}"  # year
    r")([ T]\d{2}:\d{2}(:\d{2})?)?$"
)


# ---- ECharts builders ----

def _echarts_time_series(
    title: str,
    chart_type: str,
    labels: list[str],
    series_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build an ECharts line/bar option with one or more series."""
    series = []
    for i, sd in enumerate(series_data):
        color = _PALETTE[i % len(_PALETTE)]
        s: dict[str, Any] = {
            "name": sd["name"],
            "type": chart_type,
            "data": sd["values"],
        }
        if chart_type == "line":
            s["smooth"] = True
            s["lineStyle"] = {"width": 2}
            s["itemStyle"] = {"color": color}
            s["areaStyle"] = {"opacity": 0.08} if len(series_data) == 1 else None
        else:
            s["itemStyle"] = {"color": color}
        series.append(s)

    return {
        "id": uuid.uuid4().hex,
        "title": title,
        "type": chart_type,
        "echarts": {
            "title": {"text": title, "left": "center", "textStyle": {"fontSize": 14}},
            "tooltip": {"trigger": "axis"},
            "legend": {"show": len(series) > 1, "bottom": 0},
            "grid": {"left": "3%", "right": "4%", "bottom": "12%", "containLabel": True},
            "xAxis": {"type": "category", "data": labels, "axisLabel": {"rotate": 30 if len(labels) > 8 else 0}},
            "yAxis": {"type": "value"},
            "series": series,
        },
    }


def _echarts_pie(
    title: str,
    data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build an ECharts pie/doughnut option."""
    for i, d in enumerate(data):
        d["itemStyle"] = {"color": _PALETTE[i % len(_PALETTE)]}
    return {
        "id": uuid.uuid4().hex,
        "title": title,
        "type": "pie",
        "echarts": {
            "title": {"text": title, "left": "center", "textStyle": {"fontSize": 14}},
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            "legend": {"show": True, "orient": "vertical", "right": 10, "top": "middle"},
            "series": [{
                "type": "pie",
                "radius": ["40%", "70%"],
                "avoidLabelOverlap": True,
                "label": {"show": True, "formatter": "{b}: {d}%"},
                "data": data,
            }],
        },
    }


# ---- helpers ----

def _is_numeric(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    try:
        float(str(value).replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def _to_number(value: Any) -> float | int:
    if isinstance(value, bool):
        raise ValueError("bool is not numeric for charts")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value
    text = str(value).replace(",", "").strip()
    num = float(text)
    if num.is_integer():
        return int(num)
    return num


def _is_date_like(column_name: str, values: list[Any]) -> bool:
    if _DATE_NAME_RE.search(column_name or ""):
        return True
    sample = [v for v in values if v is not None and str(v).strip() != ""][:8]
    if not sample:
        return False
    hits = 0
    for value in sample:
        text = str(value).strip()
        if _DATE_VALUE_RE.match(text):
            hits += 1
            continue
        if re.match(r"^\d{4}-\d{2}-\d{2}T", text):
            hits += 1
    return hits >= max(1, len(sample) // 2)


def _humanize_col(name: str) -> str:
    """total_cargo_tonnes -> Total cargo tonnes."""
    return name.replace("_", " ").strip().capitalize()


def _numeric_col_idxs(columns: list[str], rows: list[list[Any]]) -> list[int]:
    """Column indexes (1+) where at least half the values are numeric."""
    numeric_idxs: list[int] = []
    for idx in range(1, len(columns)):
        col_vals = [row[idx] if idx < len(row) else None for row in rows]
        numeric_count = sum(1 for v in col_vals if _is_numeric(v))
        if numeric_count >= max(1, len(col_vals) // 2):
            numeric_idxs.append(idx)
    return numeric_idxs


def _build_label_series(
    rows: list[list[Any]], numeric_idxs: list[int]
) -> tuple[list[str], dict[int, list[float | int]]]:
    """Keep only rows where ALL numeric columns are valid; return labels + values."""
    labels: list[str] = []
    series_values: dict[int, list[float | int]] = {idx: [] for idx in numeric_idxs}
    for row in rows:
        label = str(row[0]) if row else ""
        row_ok = True
        vals: dict[int, float | int] = {}
        for idx in numeric_idxs:
            if idx >= len(row) or not _is_numeric(row[idx]):
                row_ok = False
                break
            vals[idx] = _to_number(row[idx])
        if not row_ok:
            continue
        labels.append(label)
        for idx in numeric_idxs:
            series_values[idx].append(vals[idx])
    return labels, series_values


def _try_pivot_long_format(
    table: dict[str, Any], numeric_idxs: list[int]
) -> tuple[list[str], list[dict[str, Any]]] | None:
    """Pivot a 3-column long-format table (x, series, measure) into multi-series.

    Recognizes Genie outputs such as:
      year_month | airport_code | total_pax   (date first)
      airport_code | year_month | total_pax   (date second)
    x-axis = date column, one series per category value.
    Returns (labels, series_data) or None when the shape does not fit.
    """
    columns = list(table.get("columns") or [])
    rows = list(table.get("rows") or [])
    if len(columns) != 3 or len(numeric_idxs) != 1 or not rows:
        return None

    measure_idx = numeric_idxs[0]
    if measure_idx != 2:
        return None

    x0_date = _is_date_like(str(columns[0]), [r[0] if r else None for r in rows])
    x1_date = _is_date_like(str(columns[1]), [r[1] if len(r) > 1 else None for r in rows])
    if x0_date and not x1_date:
        date_idx, cat_idx = 0, 1
    elif x1_date and not x0_date:
        date_idx, cat_idx = 1, 0
    else:
        return None

    # Category column must be text-like (numeric category columns are not pivotable here).
    cat_values: list[str] = []
    for row in rows:
        v = row[cat_idx] if cat_idx < len(row) else None
        if v is None or str(v).strip() == "":
            cat_values.append("")
            continue
        if _is_numeric(v):
            return None
        cat_values.append(str(v))

    cats = list(dict.fromkeys(c for c in cat_values if c))
    if len(cats) < 2 or len(cats) > 12:
        return None

    labels = list(dict.fromkeys(
        str(r[date_idx]) for r in rows if len(r) > date_idx and r[date_idx] is not None
    ))
    try:
        labels.sort()  # date-like values (YYYY-MM etc.) sort chronologically
    except TypeError:
        pass
    if len(labels) < 2:
        return None

    value_map: dict[tuple[str, str], float | int] = {}
    label_set = set(labels)
    cat_set = set(cats)
    for row in rows:
        if len(row) <= measure_idx:
            continue
        label = str(row[date_idx]) if len(row) > date_idx else ""
        cat = str(row[cat_idx]) if len(row) > cat_idx else ""
        if label not in label_set or cat not in cat_set:
            continue
        v = row[measure_idx]
        if not _is_numeric(v):
            continue
        value_map[(label, cat)] = _to_number(v)

    series_data: list[dict[str, Any]] = []
    for cat in cats:
        values = [value_map.get((label, cat)) for label in labels]
        if any(v is not None for v in values):
            series_data.append({"name": cat, "values": values})
    if len(series_data) < 2:
        return None
    return labels, series_data


def _category_chart(
    table: dict[str, Any],
    numeric_idxs: list[int],
    chart_type: str,
) -> dict[str, Any] | None:
    """Bar/line chart over a non-date category x-axis (airports, airlines, ...)."""
    columns = list(table.get("columns") or [])
    rows = list(table.get("rows") or [])
    built = _build_label_series(rows, numeric_idxs)
    labels, series_values = built
    if len(labels) < 2:
        return None
    series_data = [
        {"name": _humanize_col(str(columns[idx])), "values": series_values[idx]}
        for idx in numeric_idxs
    ]
    return _echarts_time_series(
        title=str(table.get("title") or "Query result"),
        chart_type=chart_type,
        labels=labels,
        series_data=series_data,
    )


def _pie_chart(
    table: dict[str, Any], numeric_idxs: list[int]
) -> dict[str, Any] | None:
    """Pie/doughnut over a category x-axis with a single measure column."""
    rows = list(table.get("rows") or [])
    measure_idx = numeric_idxs[0]
    pie_data = []
    for row in rows:
        if measure_idx >= len(row) or not _is_numeric(row[measure_idx]):
            continue
        pie_data.append({"name": str(row[0]), "value": _to_number(row[measure_idx])})
    if len(pie_data) < 2:
        return None
    return _echarts_pie(title=str(table.get("title") or "Query result"), data=pie_data)


# ---- public entry point ----

def maybe_chart_from_table(table: dict[str, Any], *, prefer: str = "auto") -> dict[str, Any] | None:
    """Build an ECharts option spec from a Genie table.

    `prefer` chart type hint: "auto" (default), "bar", "line", or "pie".
    "auto" keeps built-in heuristics: short date series -> bar, long trends ->
    line, small single-measure category tables -> pie, other category shapes
    -> grouped bars. An explicit "bar"/"line"/"pie" (from the user's request)
    is honored whenever the table shape supports it.
    Returns None when the shape is not chartable.
    """
    columns = list(table.get("columns") or [])
    rows = list(table.get("rows") or [])
    if len(columns) < 2 or not rows:
        return None

    prefer = (prefer or "auto").lower()
    if prefer not in {"auto", "bar", "line", "pie"}:
        prefer = "auto"

    first_values = [row[0] if row else None for row in rows]
    is_date_x = _is_date_like(str(columns[0]), first_values)

    numeric_idxs = _numeric_col_idxs(columns, rows)
    if not numeric_idxs:
        return None

    title = str(table.get("title") or "Query result")

    # ---- Long-format pivot: [date, category, measure] -> one series per category ----
    pivot = _try_pivot_long_format(table, numeric_idxs)
    if pivot is not None:
        labels, series_data = pivot
        chart_type = prefer if prefer in {"bar", "line"} else ("line" if len(labels) > 8 else "bar")
        return _echarts_time_series(title=title, chart_type=chart_type, labels=labels, series_data=series_data)

    # ---- Category x-axis (airports, airlines, regions, ...): bar / pie ----
    if not is_date_x:
        pie_eligible = len(numeric_idxs) == 1 and len(rows) <= 12
        if prefer == "pie":
            if pie_eligible:
                pie = _pie_chart(table, numeric_idxs)
                if pie is not None:
                    return pie
            return _category_chart(table, numeric_idxs, "bar")
        if prefer in {"bar", "line"}:
            # Explicit user request wins (e.g. "show in bar chart").
            return _category_chart(table, numeric_idxs, prefer)
        # auto: preserve the historical pie default for small single-measure tables
        if pie_eligible:
            pie = _pie_chart(table, numeric_idxs)
            if pie is not None:
                return pie
        # Multi-measure or >12-row category tables previously produced NO chart —
        # render them as grouped bars instead.
        return _category_chart(table, numeric_idxs, "bar")

    # ---- Date-like x-axis: line/bar multi-series ----
    labels, series_values = _build_label_series(rows, numeric_idxs)
    if len(labels) < 2:
        return None

    series_data = [
        {"name": _humanize_col(str(columns[idx])), "values": series_values[idx]}
        for idx in numeric_idxs
    ]

    if prefer == "line" or (prefer == "auto" and len(labels) > 8):
        chart_type = "line"
    else:
        chart_type = "bar"
    return _echarts_time_series(title=title, chart_type=chart_type, labels=labels, series_data=series_data)
