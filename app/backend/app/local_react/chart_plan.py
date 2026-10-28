"""Build ECharts payloads from LLM chart plans + table data."""

from __future__ import annotations

import re
import uuid
from typing import Any

# Canvas-height contract shared with the FE
# (app/frontend/src/components/ChartCard.tsx -> canvasHeight). The FE reserves
# placeholder slots at exactly these heights so charts swap in with zero layout
# shift; keep both sides in sync if these ever change.
_CHART_HEIGHT_DEFAULT = 400
_CHART_HEIGHT_H_MIN = 320
_CHART_HEIGHT_H_BASE = 56
_CHART_HEIGHT_H_PER_ROW = 42

_PALETTE = [
    "#003366",
    "#0066A1",
    "#E68A00",
    "#339966",
    "#CC3333",
    "#6633CC",
    "#009999",
    "#CC6699",
]

# Databricks-like sequential blues (dark = highest).
_PEER_BAR_GRADIENT = [
    "#0B4F6C",
    "#0E6B8A",
    "#1282A6",
    "#1F96B8",
    "#4AADC7",
    "#7BC4D6",
    "#A6D7E4",
    "#C9E6EF",
]

_AIRPORT_LABELS = [
    ("hong kong", "NVA - Nova International"),
    ("nva", "NVA - Nova International"),
    ("hkg", "NVA - Nova International"),
    ("shanghai pudong", "PVG - Shanghai Pudong"),
    ("pudong", "PVG - Shanghai Pudong"),
    ("pvg", "PVG - Shanghai Pudong"),
    ("shanghai", "PVG - Shanghai Pudong"),
    ("incheon", "ICN - Seoul Incheon"),
    ("seoul", "ICN - Seoul Incheon"),
    ("icn", "ICN - Seoul Incheon"),
    ("changi", "SIN - Singapore Changi"),
    ("singapore", "SIN - Singapore Changi"),
    ("sin", "SIN - Singapore Changi"),
    ("taipei", "TPE - Taipei Taoyuan"),
    ("taoyuan", "TPE - Taipei Taoyuan"),
    ("tpe", "TPE - Taipei Taoyuan"),
    ("narita", "NRT - Tokyo Narita"),
    ("tokyo", "NRT - Tokyo Narita"),
    ("nrt", "NRT - Tokyo Narita"),
]


def _format_millions_label(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return f"{int(round(value))}M"
    return f"{value:.1f}M"


def normalize_airport_label(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return raw
    low = raw.lower()
    # SHA in a peer list is Hongqiao. Pudong only when the user said Pudong or PVG.
    # "sha" is a substring of "shanghai", so short codes are matched as tokens.
    if re.search(r"pudong|\bpvg\b", low):
        return "PVG - Shanghai Pudong"
    if re.search(r"hongqiao|\bsha\b", low) or low in {"shanghai", "shanghai airport"}:
        return "SHA - Shanghai Hongqiao"
    for needle, label in _AIRPORT_LABELS:
        if needle in {"sha", "shanghai", "pvg", "pudong", "shanghai pudong"}:
            continue
        if needle in low:
            return label
    return raw


def _normalize_rows(columns: list[str], rows: list[Any]) -> list[list[Any]]:
    """Accept array rows or object rows from Wren/SQL."""
    out: list[list[Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append([row.get(c) for c in columns])
        elif isinstance(row, (list, tuple)):
            out.append(list(row))
        else:
            out.append([row])
    return out


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        s = str(value).replace(",", "").replace("—", "").replace("–", "").strip()
        if not s or s == "-":
            return None
        # values like "32.7 million" -> 32.7
        if "million" in s.lower():
            s = s.lower().split("million", 1)[0].strip()
        return float(s)
    except Exception:
        return None


def build_chart_from_plan(plan: dict[str, Any], table: dict[str, Any]) -> dict[str, Any] | None:
    """Convert chart plan + table into charts[].echarts object.

    Supported types:
      - line / bar / pie (baseline)
      - area (filled line)
      - stacked_bar
      - combo (mixed bar+line; optional dual y-axis)

    Dual / multi y-axis knobs (any cartesian type):
      - dual_y: true  -> series after the first y use yAxisIndex=1
      - y_axes: ["left_name", "right_name"]
      - series_types: ["bar", "line", ...] aligned to y columns
      - y_axis_index: [0, 1, ...] aligned to y columns
    """
    if not isinstance(plan, dict) or not isinstance(table, dict):
        return None
    columns = [str(c) for c in (table.get("columns") or [])]
    rows = _normalize_rows(columns, list(table.get("rows") or []))
    if not columns or not rows:
        return None

    chart_type = str(plan.get("type") or "line").lower().strip()
    if chart_type in {"stacked", "stack", "stackedbar"}:
        chart_type = "stacked_bar"
    if chart_type not in {"line", "bar", "pie", "area", "stacked_bar", "combo"}:
        chart_type = "line"
    title = str(plan.get("title") or table.get("title") or "Chart")

    x_col = str(plan.get("x") or plan.get("x_column") or columns[0])
    y_cols = plan.get("y") or plan.get("y_columns") or []
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    y_cols = [str(y) for y in y_cols] if y_cols else [c for c in columns if c != x_col][:3]
    if not y_cols:
        return None

    col_index = {c: i for i, c in enumerate(columns)}
    if x_col not in col_index:
        for c in columns:
            if c.lower() == x_col.lower():
                x_col = c
                break
        else:
            x_col = columns[0]
    y_resolved: list[str] = []
    for y in y_cols:
        if y in col_index:
            y_resolved.append(y)
            continue
        hit = next((c for c in columns if c.lower() == y.lower()), None)
        if hit:
            y_resolved.append(hit)
    if not y_resolved:
        return None

    xi = col_index[x_col]
    labels = [str(r[xi]) if xi < len(r) else "" for r in rows]

    if chart_type == "pie":
        yi = col_index[y_resolved[0]]
        data = []
        for i, r in enumerate(rows):
            val = _to_float(r[yi]) if yi < len(r) else None
            if val is None:
                continue
            data.append(
                {
                    "name": labels[i],
                    "value": val,
                    "itemStyle": {"color": _PALETTE[len(data) % len(_PALETTE)]},
                }
            )
        if len(data) < 2:
            return None
        return {
            "id": uuid.uuid4().hex,
            "title": title,
            "type": "pie",
            "echarts": {
                "title": {"text": title, "left": "center", "textStyle": {"fontSize": 14}},
                "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                "legend": {"show": True, "orient": "vertical", "right": 10, "top": "middle"},
                "series": [
                    {
                        "type": "pie",
                        "radius": ["40%", "70%"],
                        "avoidLabelOverlap": True,
                        "label": {"show": True, "formatter": "{b}: {d}%"},
                        "data": data,
                    }
                ],
            },
            "index": 0,
            "position": "end",
        }

    series_types_raw = plan.get("series_types") or plan.get("seriesTypes") or []
    if isinstance(series_types_raw, str):
        series_types_raw = [series_types_raw]
    series_types = [str(t).lower().strip() for t in series_types_raw] if series_types_raw else []

    y_axis_index_raw = plan.get("y_axis_index") or plan.get("yAxisIndex") or []
    if isinstance(y_axis_index_raw, (int, float, str)):
        y_axis_index_raw = [y_axis_index_raw]
    y_axis_indexes: list[int] = []
    for v in y_axis_index_raw:
        try:
            y_axis_indexes.append(int(v))
        except Exception:
            continue

    dual_y = bool(plan.get("dual_y") or plan.get("dualY") or plan.get("dual_axis"))
    if dual_y and len(y_resolved) >= 2 and not y_axis_indexes:
        y_axis_indexes = [0] + [1] * (len(y_resolved) - 1)

    # Combo default: first series bar, remaining line; enable dual_y when >=2 series.
    if chart_type == "combo":
        if not series_types:
            series_types = ["bar"] + ["line"] * max(0, len(y_resolved) - 1)
        if len(y_resolved) >= 2 and not y_axis_indexes:
            dual_y = True
            y_axis_indexes = [0] + [1] * (len(y_resolved) - 1)

    axis_names_raw = plan.get("y_axes") or plan.get("y_axis") or plan.get("yAxis") or []
    if isinstance(axis_names_raw, str):
        axis_names_raw = [axis_names_raw]
    axis_names = [str(n) for n in axis_names_raw] if axis_names_raw else []

    series: list[dict[str, Any]] = []
    max_axis = 0
    for si, y in enumerate(y_resolved):
        yi = col_index[y]
        values = [_to_float(r[yi]) if yi < len(r) else None for r in rows]
        if not any(v is not None for v in values):
            continue
        color = _PALETTE[si % len(_PALETTE)]

        if chart_type == "combo" and si < len(series_types):
            st = series_types[si]
        elif chart_type == "stacked_bar":
            st = "bar"
        elif chart_type == "area":
            st = "line"
        elif si < len(series_types) and series_types[si] in {"line", "bar", "area"}:
            st = series_types[si]
        else:
            st = (
                "line"
                if chart_type in {"line", "area"}
                else "bar"
                if chart_type in {"bar", "stacked_bar"}
                else chart_type
            )
        if st == "area":
            st = "line"
            area = True
        else:
            area = chart_type == "area"

        axis_i = y_axis_indexes[si] if si < len(y_axis_indexes) else 0
        if axis_i not in (0, 1):
            axis_i = 0
        max_axis = max(max_axis, axis_i)

        s: dict[str, Any] = {
            "name": y.replace("_", " ").strip().capitalize(),
            "type": st if st in {"line", "bar"} else "line",
            "data": values,
            "itemStyle": {"color": color},
            "yAxisIndex": axis_i,
        }
        if s["type"] == "line":
            s["smooth"] = True
            s["lineStyle"] = {"width": 2}
            if area:
                s["areaStyle"] = {"opacity": 0.18}
        if s["type"] == "bar":
            s["barMaxWidth"] = 48
            if chart_type == "stacked_bar":
                s["stack"] = str(plan.get("stack") or "total")
        series.append(s)

    if not series:
        return None

    # Airport / peer compares -> horizontal Databricks-style bars (single series only).
    airportish = any(
        re_search_airport(lbl) or "airport" in str(x_col).lower() for lbl in labels
    )
    if chart_type == "bar" and airportish and len(series) == 1 and max_axis == 0:
        values = [float(v) if v is not None else 0.0 for v in series[0]["data"]]
        paired = list(zip(labels, values))
        return build_bar_chart_from_pairs(title, paired, y_name=y_resolved[0])

    use_dual = max_axis >= 1 or dual_y
    if use_dual:
        left_name = axis_names[0] if len(axis_names) > 0 else (
            series[0]["name"] if series else "Value"
        )
        right_name = axis_names[1] if len(axis_names) > 1 else next(
            (s["name"] for s in series if int(s.get("yAxisIndex") or 0) == 1),
            "Value (right)",
        )
        y_axis: Any = [
            {
                "type": "value",
                "name": left_name,
                "nameTextStyle": {"fontSize": 11},
                "axisLabel": {"color": "#334155"},
            },
            {
                "type": "value",
                "name": right_name,
                "nameTextStyle": {"fontSize": 11},
                "axisLabel": {"color": "#334155"},
                "splitLine": {"show": False},
            },
        ]
        grid_right = "6%"
    else:
        y_axis = {"type": "value", "name": (axis_names[0] if axis_names else "")}
        grid_right = "4%"

    return {
        "id": uuid.uuid4().hex,
        "title": title,
        "type": chart_type,
        "dual_y": bool(use_dual),
        # Explicit height hint so the FE placeholder reserves the exact slot.
        "height": _CHART_HEIGHT_DEFAULT,
        "echarts": {
            "title": {"text": title, "left": "center", "textStyle": {"fontSize": 14}},
            "tooltip": {"trigger": "axis"},
            "legend": {"show": len(series) > 1, "bottom": 0},
            "grid": {
                "left": "3%",
                "right": grid_right,
                "bottom": "12%",
                "containLabel": True,
            },
            "xAxis": {
                "type": "category",
                "data": labels,
                "axisLabel": {"rotate": 30 if len(labels) > 8 else 0},
            },
            "yAxis": y_axis,
            "series": series,
            "color": _PALETTE,
        },
        "index": 0,
        "position": "end",
    }


def re_search_airport(label: str) -> bool:
    low = str(label or "").lower()
    return any(k in low for k, _ in _AIRPORT_LABELS)


def build_bar_chart_from_pairs(
    title: str,
    pairs: list[tuple[str, float]],
    y_name: str = "Value",
) -> dict[str, Any] | None:
    """Build a Databricks-style horizontal peer bar chart from (label, value) pairs."""
    clean: list[tuple[str, float]] = []
    seen: set[str] = set()
    for k, v in pairs:
        if k is None or v is None:
            continue
        try:
            num = float(v)
        except Exception:
            continue
        if num <= 0:
            continue
        label = normalize_airport_label(str(k))
        if label in seen:
            continue
        seen.add(label)
        clean.append((label, num))
    if len(clean) < 2:
        return None

    # Highest at top (Databricks peer ranking look).
    clean.sort(key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in clean]
    values = [round(v, 1) for _, v in clean]
    colors = [
        _PEER_BAR_GRADIENT[i % len(_PEER_BAR_GRADIENT)] for i in range(len(clean))
    ]
    data = [
        {
            "value": values[i],
            "itemStyle": {
                "color": colors[i],
                "borderRadius": [0, 6, 6, 0],
            },
            "label": {
                "show": True,
                "position": "insideRight",
                "formatter": _format_millions_label(values[i]),
                "color": "#ffffff",
                "fontWeight": 700,
                "fontSize": 12,
            },
        }
        for i in range(len(clean))
    ]

    height_hint = max(
        _CHART_HEIGHT_H_MIN,
        _CHART_HEIGHT_H_BASE + len(clean) * _CHART_HEIGHT_H_PER_ROW,
    )
    return {
        "id": uuid.uuid4().hex,
        "title": title,
        "type": "bar",
        "orientation": "horizontal",
        "height": height_hint,
        "echarts": {
            "title": {
                "text": title,
                "left": "center",
                "textStyle": {"fontSize": 14, "fontWeight": 600, "color": "#0a3d73"},
            },
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "shadow"},
                "formatter": "{b}: {c}M",
            },
            "grid": {
                "left": 8,
                "right": 28,
                "top": 42,
                "bottom": 28,
                "containLabel": True,
            },
            "xAxis": {
                "type": "value",
                "name": "Millions",
                "nameLocation": "middle",
                "nameGap": 24,
                "nameTextStyle": {"color": "#7a8799", "fontSize": 11},
                "axisLabel": {"color": "#7a8799"},
                "splitLine": {"lineStyle": {"color": "#eef2f6"}},
            },
            "yAxis": {
                "type": "category",
                "data": labels,
                "inverse": True,
                "axisTick": {"show": False},
                "axisLine": {"show": False},
                "axisLabel": {
                    "color": "#10233d",
                    "fontSize": 12,
                    "fontWeight": 600,
                },
            },
            "series": [
                {
                    "name": y_name.replace("_", " ").strip().capitalize(),
                    "type": "bar",
                    "data": data,
                    "barWidth": 22,
                    "barMaxWidth": 28,
                }
            ],
        },
        "index": 0,
        "position": "end",
    }
