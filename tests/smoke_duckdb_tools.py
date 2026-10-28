from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app" / "backend"), str(ROOT / "agent" / "src")]

os.environ.setdefault("CI_AGENT_MODE", "local_react")
os.environ.setdefault("CI_ENABLE_GENIE_FALLBACK", "false")
os.environ.setdefault("WREN_MCP_ENABLED", "false")


def test_duckdb_ytd_passengers():
    from app.local_react.tools import ToolRuntime

    rt = ToolRuntime()
    res = rt._run_sql(
        """
        SELECT hub_code, SUM(passengers) AS pax
        FROM hub_kpi_monthly
        WHERE year_month >= DATE '2025-01-01'
          AND year_month < DATE '2026-01-01'
        GROUP BY 1
        ORDER BY pax DESC
        """,
        title="YTD",
    )
    assert res["ok"], res
    rows = {r[0]: r[1] for r in res["table"]["rows"]}
    assert rows["NVA"] > rows["SKY"]


def test_alias_rewrite():
    from app.local_react.tools import _rewrite_common_column_aliases

    out = _rewrite_common_column_aliases(
        "SELECT total_passengers, atm FROM foo.bar.hub_kpi_monthly"
    )
    assert "passengers" in out
    assert "aircraft_movements" in out
    assert "hub_kpi_monthly" in out
    assert "foo.bar" not in out


def test_app_imports():
    from app.main import app

    assert "DualSource" in app.title
