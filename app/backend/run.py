from __future__ import annotations

import os
import sys
from pathlib import Path

# app/backend/run.py -> repo root is parents[2]
ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parent
AGENT_SRC = ROOT / "agent" / "src"
for p in (str(BACKEND), str(AGENT_SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Default static dir relative to backend package location
os.environ.setdefault("CI_STATIC_DIR", str(BACKEND / "static"))
os.environ.setdefault("CI_FILES_DIR", str((Path(__file__).resolve().parents[2] / "tmp" / "ci_files")))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("CI_AGENT_MODE", "local_react")
os.environ.setdefault("CI_HISTORY_BACKEND", "local")
os.environ.setdefault("CI_ENABLE_GENIE_FALLBACK", "false")
os.environ.setdefault("WREN_MCP_ENABLED", "false")

# Local PAT convenience + scheme normalization for Apps/SDK hosts
if not os.getenv("DATABRICKS_HOST") and os.getenv("TARGET_HOST"):
    os.environ["DATABRICKS_HOST"] = os.environ["TARGET_HOST"]
if not os.getenv("DATABRICKS_TOKEN") and os.getenv("TARGET_TOKEN"):
    os.environ["DATABRICKS_TOKEN"] = os.environ["TARGET_TOKEN"]


def _normalize_host_env(key: str) -> None:
    val = (os.getenv(key) or "").strip().strip("\"'").rstrip("/")
    if not val:
        return
    if not (val.startswith("https://") or val.startswith("http://")):
        val = f"https://{val}"
    os.environ[key] = val


_normalize_host_env("DATABRICKS_HOST")
_normalize_host_env("TARGET_HOST")

import uvicorn


if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT") or os.environ.get("PORT") or 8000)
    # Import check before serving
    from app.main import app  # noqa: F401

    uvicorn.run("app.main:app", host="0.0.0.0", port=port, factory=False)
