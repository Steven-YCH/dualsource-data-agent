"""Runtime facts for DualSource local ReAct (seed-clamped)."""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

SEED_LAST_COMPLETE_MONTH = date(2025, 12, 1)


def hk_today() -> date:
    try:
        real = datetime.now(ZoneInfo("Asia/Hong_Kong")).date()
    except Exception:
        real = date.today()
    return min(real, date(2026, 1, 15))


def last_complete_month(today: date | None = None) -> date:
    d = today or hk_today()
    first_this = d.replace(day=1)
    prev_last = first_this.fromordinal(first_this.toordinal() - 1)
    complete = prev_last.replace(day=1)
    return min(complete, SEED_LAST_COMPLETE_MONTH)


def month_label(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def month_name(d: date) -> str:
    return d.strftime("%B %Y")


def next_month_start(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def h1_window(year: int | None = None, today: date | None = None) -> tuple[date, date]:
    d = today or hk_today()
    y = year or last_complete_month(d).year
    start = date(y, 1, 1)
    natural_end = date(y, 6, 1)
    complete = last_complete_month(d)
    end = min(natural_end, complete) if y == complete.year else natural_end
    if end < start:
        end = start
    return start, end


def build_runtime_facts(today: date | None = None) -> dict[str, str]:
    d = today or hk_today()
    ytd = last_complete_month(d)
    h1_start, h1_end = h1_window(ytd.year, d)
    ytd_end_excl = next_month_start(ytd)
    h1_end_excl = next_month_start(h1_end)
    return {
        "today": d.isoformat(),
        "current_year": str(ytd.year),
        "last_complete_month": month_label(ytd),
        "last_complete_month_name": month_name(ytd),
        "ytd_window": f"January-{ytd.strftime('%B')} {ytd.year}",
        "ytd_sql_start": f"{ytd.year:04d}-01-01",
        "ytd_sql_end_exclusive": ytd_end_excl.isoformat(),
        "h1_window": f"January-{h1_end.strftime('%B')} {h1_end.year}",
        "h1_sql_start": h1_start.isoformat(),
        "h1_sql_end_exclusive": h1_end_excl.isoformat(),
        "expected_ytd_months": str(ytd.month),
        "seed_horizon": "2024-01-01 .. 2025-12-01",
    }


def runtime_facts_text(today: date | None = None) -> str:
    f = build_runtime_facts(today)
    lines = [
        "RUNTIME FACTS (must obey):",
        f"- Demo warehouse seed horizon: {f['seed_horizon']}",
        f"- Effective today for planning: {f['today']}",
        f"- Current analysis year: {f['current_year']}",
        (
            f"- YTD = January through the last complete seeded month "
            f"({f['ytd_window']}; last complete month = {f['last_complete_month']})."
        ),
        "- Do not query months after the seed horizon; there is no 2026 internal data.",
        (
            f"- For YTD {f['current_year']} SQL use year_month >= TIMESTAMP '{f['ytd_sql_start']}' "
            f"AND year_month < TIMESTAMP '{f['ytd_sql_end_exclusive']}'. "
            f"Expect about {f['expected_ytd_months']} monthly rows."
        ),
        (
            f"- First half / H1 means {f['h1_window']} "
            f"(SQL: year_month >= TIMESTAMP '{f['h1_sql_start']}' "
            f"AND year_month < TIMESTAMP '{f['h1_sql_end_exclusive']}')."
        ),
        "- If the user names no year, use the current analysis year above.",
        "- Cargo units: tonnes or million tonnes (never bare 't').",
    ]
    return "\n".join(lines) + "\n"
