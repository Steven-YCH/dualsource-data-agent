"""Local DuckDB SQL client for DualSource Agent (replaces Databricks/Wren)."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import duckdb

_WRITE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|COPY|EXPORT|CALL|INSTALL|LOAD)\b",
    re.I,
)


def default_db_path() -> Path:
    env = (os.getenv("DUALSOURCE_DUCKDB_PATH") or "").strip()
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[4]
    return root / "data" / "warehouse" / "warehouse.duckdb"


class DuckDBSQLClient:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or default_db_path())
        self.last_observability: dict[str, Any] = {"transport": "duckdb", "stages": []}
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"DuckDB warehouse missing at {self.db_path}. Run: python data/load_duckdb.py"
            )

    def run_sql(self, sql: str, *, limit: int = 200, max_rows: int | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        row_cap = int(max_rows or limit or 200)
        text = str(sql or "").strip().rstrip(";")
        if not text:
            return {"ok": False, "error": "Empty SQL", "source": "duckdb"}
        if _WRITE.search(text):
            return {
                "ok": False,
                "error": "Only read-only SELECT/WITH/EXPLAIN queries are allowed",
                "source": "duckdb",
                "sql": text,
            }
        compact = " ".join(text.lower().split())
        if (not compact.startswith("explain")) and (" limit " not in f" {compact} "):
            text = f"SELECT * FROM ({text}) AS _q LIMIT {row_cap}"
        try:
            con = duckdb.connect(str(self.db_path), read_only=True)
            try:
                rel = con.execute(text)
                cols = [d[0] for d in rel.description] if rel.description else []
                rows = rel.fetchall()
            finally:
                con.close()
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "error": str(exc),
                "source": "duckdb",
                "sql": text,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            }
        table = {
            "title": "query_result",
            "columns": cols,
            "rows": [list(r) for r in rows],
        }
        return {
            "ok": True,
            "source": "duckdb",
            "sql": text,
            "table": table,
            "columns": cols,
            "rows": [list(r) for r in rows],
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }

    def list_tables(self) -> list[str]:
        con = duckdb.connect(str(self.db_path), read_only=True)
        try:
            return [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        finally:
            con.close()
