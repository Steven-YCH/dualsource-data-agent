"""Deterministic headline checks the writer is not allowed to override.

`bag_id` is reused across months, so COUNT(DISTINCT bag_id) is not the
piece count. Pieces are SUM(bag_count). A monthly series sums to the
window total.
"""

from __future__ import annotations

import re
from typing import Any

_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+")
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_YM = re.compile(r"^\d{4}-\d{2}")
_BAG_TABLE = re.compile(r"((?:[\w]+\.){0,2}genie_baggage_journey)", re.I)


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def _fmt_int(value: float) -> str:
    return f"{int(round(value)):,}"


def _fmt_num(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return _fmt_int(value)
    return f"{value:,.1f}"


def _columns(table: dict[str, Any]) -> list[str]:
    return [str(c) for c in (table.get("columns") or [])]


def _blob(table: dict[str, Any]) -> str:
    return " ".join(_columns(table) + [str(table.get("title") or "")]).lower()


def baggage_window_count(tables: list[dict[str, Any]]) -> float | None:
    """One-row pieces total. Monthly series are not a window total."""
    for table in tables:
        if "bag" not in _blob(table):
            continue
        rows = list(table.get("rows") or [])
        cols = [c.lower() for c in _columns(table)]
        if len(rows) != 1 or not rows[0]:
            continue
        if any("year" in c or c in {"ym", "month"} for c in cols):
            continue
        try:
            return float(str(rows[0][-1]).replace(",", ""))
        except (TypeError, ValueError):
            continue
    return None


def baggage_monthly_sum(tables: list[dict[str, Any]]) -> float | None:
    total = 0.0
    found = False
    for table in tables:
        if "bag" not in _blob(table):
            continue
        cols = [c.lower() for c in _columns(table)]
        ym = next((i for i, c in enumerate(cols) if "year" in c or c in {"ym", "month"}), None)
        if ym is None:
            continue
        for row in table.get("rows") or []:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            try:
                total += float(str(row[-1]).replace(",", ""))
                found = True
            except (TypeError, ValueError):
                continue
    return total if found else None


def baggage_repair_sql(sql: str, table: dict[str, Any]) -> str | None:
    """Window SUM(bag_count) when the model only returned monthly pieces."""
    compact = re.sub(r"\s+", " ", sql or "").lower()
    if "genie_baggage_journey" not in compact:
        return None
    if "sum(bag_count)" in compact and "group by" not in compact:
        return None
    match = _BAG_TABLE.search(sql or "")
    if not match:
        return None
    cols = [c.lower() for c in _columns(table)]
    ym_at = next((i for i, c in enumerate(cols) if "year" in c or c in {"ym", "month"}), None)
    if ym_at is None:
        return None
    months: list[str] = []
    for row in table.get("rows") or []:
        if not isinstance(row, (list, tuple)) or ym_at >= len(row):
            continue
        text = str(row[ym_at])[:7]
        if _YM.match(text):
            months.append(text)
    if len(months) < 2:
        return None
    years = {month[:4] for month in months}
    if len(years) > 1:
        latest = max(years)
        months = [month for month in months if month.startswith(latest)]
    return (
        "SELECT SUM(bag_count) AS baggage_pieces "
        f"FROM {match.group(1)} "
        f"WHERE year_month >= '{min(months)}' AND year_month <= '{max(months)}'"
    )


def _year_sums(tables: list[dict[str, Any]], *needles: str) -> dict[int, float]:
    totals: dict[int, float] = {}
    for table in tables:
        blob = _blob(table)
        if not any(n in blob for n in needles):
            continue
        cols = [c.lower() for c in _columns(table)]
        ym = next((i for i, c in enumerate(cols) if "year" in c or c in {"ym", "month"}), None)
        if ym is None:
            continue
        for row in table.get("rows") or []:
            if not isinstance(row, (list, tuple)) or len(row) <= ym:
                continue
            year = str(row[ym])[:4]
            if not year.isdigit():
                continue
            try:
                value = float(str(row[-1]).replace(",", ""))
            except (TypeError, ValueError):
                continue
            totals[int(year)] = totals.get(int(year), 0.0) + value
    return totals


def _replace_near(head: str, target: float, *, band: float) -> str:
    def repl(match: re.Match[str]) -> str:
        got = _num(match.group(0))
        if got == 0 or abs(got - target) / max(abs(target), 1) > band:
            return match.group(0)
        if abs(got - target) < 0.05:
            return match.group(0)
        return _fmt_num(target)

    return _NUM.sub(repl, head)


def align_answer(
    answer: str,
    tables: list[dict[str, Any]],
    question: str,
) -> tuple[str, dict[str, Any]]:
    """Rewrite a headline that does not equal the returned rows."""
    text = answer or ""
    notes: dict[str, Any] = {}
    question_l = (question or "").lower()
    head, sep, tail = text.partition("\n\n")

    window = baggage_window_count(tables)
    monthly = baggage_monthly_sum(tables)
    # Monthly SUM(bag_count) rows are the piece count. A one-row distinct count
    # must not replace them.
    pieces = monthly if monthly is not None else window
    if pieces is not None and re.search(r"bag", question_l):
        notes["baggage_pieces"] = pieces

        def bag_repl(match: re.Match[str]) -> str:
            got = _num(match.group(0))
            if pieces <= 0 or abs(got - pieces) <= 1:
                return match.group(0)
            gap = abs(got - pieces) / pieces
            half = abs(got - pieces / 2) / max(pieces / 2, 1)
            # Distinct-bag headlines land near half. Other wrong totals sit close
            # to the row sum but are not equal to it.
            if half < 0.02 or (got > 1000 and gap < 0.15):
                notes["baggage_headline_rewritten"] = True
                return _fmt_int(pieces)
            return match.group(0)

        head = _NUM.sub(bag_repl, head)

    pax = _year_sums(tables, "passenger", "pax")
    # Fire whenever evidence has multi-year passenger rows, including short chart
    # follow-ups ("show in line chart") that would otherwise keep a wrong body total.
    if len(pax) >= 2:
        years = sorted(pax)
        earlier, later = pax[years[0]], pax[years[-1]]
        if earlier > 0:
            pct = (later - earlier) / earlier * 100.0
            notes["passenger_yoy_pct"] = round(pct, 1)
            notes["passenger_yoy_totals"] = {
                str(years[0]): earlier,
                str(years[-1]): later,
            }

            def pct_repl(match: re.Match[str]) -> str:
                got = float(match.group(1))
                if abs(got - pct) > 0.3:
                    notes["yoy_rewritten"] = True
                    return f"{pct:.1f}%"
                return match.group(0)

            # Align headline AND body — models often correct the lead sentence then
            # contradict it in the next paragraph with a different total/pct.
            full = head + sep + tail
            full = _PCT.sub(pct_repl, full)
            for total in (earlier, later):
                rewritten = _replace_near(full, total, band=0.03)
                if rewritten != full:
                    notes["yoy_total_rewritten"] = True
                full = rewritten
            if sep:
                head, sep, tail = full.partition("\n\n")
            else:
                head, sep, tail = full, "", ""

    atm = _year_sums(tables, "atm", "movement")
    wants_atm = bool(re.search(r"\batm\b|air traffic movement", question_l))
    if wants_atm and not atm:
        kept = []
        for line in head.splitlines():
            if re.search(r"\batm\b|movement", line, re.I) and _NUM.search(line):
                notes["atm_sentence_dropped"] = True
                continue
            kept.append(line)
        head = "\n".join(kept).strip()
        if "not in the returned rows" not in head.lower():
            head += (
                "\n\nAir traffic movements were not in the returned rows, "
                "so this answer does not state an ATM total."
            )
            notes["atm_missing"] = True
    elif atm:
        atm_total = sum(atm.values())
        notes["atm_total"] = atm_total
        head = _replace_near(head, atm_total, band=0.08)

    return (head + sep + tail).strip(), notes
