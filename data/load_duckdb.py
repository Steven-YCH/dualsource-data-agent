"""Build local DuckDB warehouse from seed CSVs."""
from __future__ import annotations

import os
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed"
DB_PATH = Path(os.getenv("DUALSOURCE_DUCKDB_PATH") or (ROOT / "data" / "warehouse" / "warehouse.duckdb"))


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = duckdb.connect(str(DB_PATH))
    mapping = {
        "hub_kpi_monthly": SEED / "hub_kpi_monthly.csv",
        "route_kpi_monthly": SEED / "route_kpi_monthly.csv",
        "hub_retail_monthly": SEED / "hub_retail_monthly.csv",
        "hub_otp_monthly": SEED / "hub_otp_monthly.csv",
    }
    for table, csv_path in mapping.items():
        con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?, header=true);",
            [str(csv_path)],
        )
    print("DuckDB ready:", DB_PATH)
    for (name,) in con.execute("SHOW TABLES").fetchall():
        n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  {name}: {n} rows")
    con.close()


if __name__ == "__main__":
    main()
