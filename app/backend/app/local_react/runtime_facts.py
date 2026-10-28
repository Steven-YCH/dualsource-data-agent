"""Hong Kong-calendar runtime facts for local ReAct (ported from Serving graph)."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo


def hk_today() -> date:
    try:
        return datetime.now(ZoneInfo("Asia/Hong_Kong")).date()
    except Exception:
        return date.today()


def last_complete_month(today: date | None = None) -> date:
    """First day of the last fully completed calendar month (HK time)."""
    d = today or hk_today()
    first_this = d.replace(day=1)
    prev_last = first_this.fromordinal(first_this.toordinal() - 1)
    return prev_last.replace(day=1)


def month_label(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def month_name(d: date) -> str:
    return d.strftime("%B %Y")


def next_month_start(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def h1_window(year: int | None = None, today: date | None = None) -> tuple[date, date]:
    """Calendar H1 = Jan-Jun of the target year, clamped to last complete month."""
    d = today or hk_today()
    y = year or d.year
    start = date(y, 1, 1)
    natural_end = date(y, 6, 1)
    complete = last_complete_month(d)
    end = min(natural_end, complete) if y == d.year else natural_end
    if end < start:
        end = start
    return start, end


def build_runtime_facts(today: date | None = None) -> dict[str, str]:
    d = today or hk_today()
    ytd = last_complete_month(d)
    h1_start, h1_end = h1_window(d.year, d)
    ytd_end_excl = next_month_start(ytd)
    h1_end_excl = next_month_start(h1_end)
    return {
        "today": d.isoformat(),
        "current_year": str(d.year),
        "last_complete_month": month_label(ytd),
        "last_complete_month_name": month_name(ytd),
        "ytd_window": f"January-{ytd.strftime('%B')} {ytd.year}",
        "ytd_sql_start": f"{d.year:04d}-01-01",
        "ytd_sql_end_exclusive": ytd_end_excl.isoformat(),
        "h1_window": f"January-{h1_end.strftime('%B')} {h1_end.year}",
        "h1_sql_start": h1_start.isoformat(),
        "h1_sql_end_exclusive": h1_end_excl.isoformat(),
        "expected_ytd_months": str(ytd.month if ytd.year == d.year else 12),
    }


def runtime_facts_text(today: date | None = None) -> str:
    f = build_runtime_facts(today)
    return (
        "RUNTIME FACTS (Hong Kong time - must obey):\n"
        f"- Today: {f['today']}\n"
        f"- Current year: {f['current_year']}\n"
        f"- YTD = January through the last complete month "
        f"({f['ytd_window']}; last complete month = {f['last_complete_month']}).\n"
        "- Current incomplete month is excluded from default YTD / H1 aggregates.\n"
        f"- For YTD {f['current_year']} SQL use year_month >= TIMESTAMP '{f['ytd_sql_start']}' "
        f"AND year_month < TIMESTAMP '{f['ytd_sql_end_exclusive']}'. "
        f"Expect about {f['expected_ytd_months']} monthly rows, not Q1-only.\n"
        f"- 'First half' / H1 for the current year means {f['h1_window']} "
        f"(SQL: year_month >= TIMESTAMP '{f['h1_sql_start']}' "
        f"AND year_month < TIMESTAMP '{f['h1_sql_end_exclusive']}').\n"
        "- If the user names no year, use the current year above. Do not answer a different year.\n"
        "- Quarters: Q1 Jan-Mar, Q2 Apr-Jun, Q3 Jul-Sep, Q4 Oct-Dec, "
        "clamped to the last complete month when the quarter is still open.\n"
        "- Data freshness: monthly cargo/passengers available through the last complete month.\n"
        "- Cargo units: tonnes or million tonnes (never bare 't').\n"
    )
