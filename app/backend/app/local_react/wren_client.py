"""Compatibility stub — Databricks/Wren removed in DualSource portfolio build."""
from __future__ import annotations

from app.local_react.duckdb_client import DuckDBSQLClient

# Historical names used by older imports/tests.
DatabricksSQLClient = DuckDBSQLClient
WrenMCPClient = DuckDBSQLClient
