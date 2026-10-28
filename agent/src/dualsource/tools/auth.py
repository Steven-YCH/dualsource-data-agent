from __future__ import annotations

import os
import re
from contextvars import ContextVar
from functools import lru_cache
from typing import Any


_obo_token: ContextVar[str | None] = ContextVar("ci_obo_token", default=None)

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_NUMERIC_ID_RE = re.compile(r"^\d+@\d+$")


def set_obo_token(token: str | None) -> None:
    _obo_token.set(token or None)


def get_obo_token() -> str | None:
    return _obo_token.get() or os.getenv("CI_OBO_TOKEN") or None


def looks_like_email(value: str | None) -> bool:
    if not value or not isinstance(value, str):
        return False
    v = value.strip()
    if not v or " " in v:
        return False
    if _NUMERIC_ID_RE.match(v):
        return False
    if v.endswith("@databricks.com") and v.split("@", 1)[0].isdigit():
        return False
    return bool(_EMAIL_RE.match(v))


def display_name_from_email(email: str) -> str:
    local = email.split("@", 1)[0]
    parts = [p for p in local.replace(".", " ").replace("_", " ").split() if p]
    if not parts:
        return email
    return " ".join(p.capitalize() for p in parts)


def initials_from(name: str | None, email: str) -> str:
    base = (name or email or "?").strip()
    if looks_like_email(base):
        base = base.split("@", 1)[0]
    # never derive initials from pure numeric workspace ids
    if base.isdigit():
        base = email.split("@", 1)[0] if looks_like_email(email) else "?"
    parts = [p for p in base.replace(".", " ").replace("_", " ").replace("@", " ").split() if p and not p.isdigit()]
    if not parts and looks_like_email(email):
        parts = [p for p in email.split("@", 1)[0].replace(".", " ").replace("_", " ").split() if p and not p.isdigit()]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()


def normalize_workspace_host(host: str | None) -> str:
    """Ensure Databricks host has https:// scheme.

    Apps / SDK sometimes inject hostname only, which breaks requests URLs.
    """
    h = (host or "").strip().strip("\"'").rstrip("/")
    if not h:
        return ""
    if h.startswith("https://") or h.startswith("http://"):
        return h.rstrip("/")
    return f"https://{h}"


@lru_cache(maxsize=1)
def workspace_host() -> str:
    host = normalize_workspace_host(os.getenv("DATABRICKS_HOST") or os.getenv("TARGET_HOST") or "")
    if host:
        return host
    try:
        from databricks.sdk import WorkspaceClient

        return normalize_workspace_host(WorkspaceClient().config.host or "")
    except Exception:
        return ""


def auth_headers() -> dict[str, str]:
    """Return Authorization headers for Databricks REST calls.

    Order:
    1) request-scoped OBO token
    2) PAT (local/dev)
    3) Databricks SDK default auth (Apps service principal)
    """
    token = get_obo_token() or os.getenv("DATABRICKS_TOKEN") or os.getenv("TARGET_TOKEN") or ""
    if token:
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    from databricks.sdk import WorkspaceClient

    headers = WorkspaceClient().config.authenticate()
    if "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"
    return headers


def resolve_user_from_obo(token: str | None = None) -> dict[str, str] | None:
    """Resolve real email/display name via SCIM Me using OBO token."""
    tok = token or get_obo_token()
    host = workspace_host()
    if not tok or not host:
        return None
    try:
        import requests

        resp = requests.get(
            f"{host}/api/2.0/preview/scim/v2/Me",
            headers={"Authorization": f"Bearer {tok}"},
            timeout=20,
        )
        if resp.status_code >= 400:
            return None
        data = resp.json()
        emails = data.get("emails") or []
        email = None
        for e in emails:
            val = e.get("value") if isinstance(e, dict) else None
            if looks_like_email(val):
                email = val.lower()
                if e.get("primary"):
                    break
        user_name = data.get("userName")
        if not email and looks_like_email(user_name):
            email = str(user_name).lower()
        if not email:
            return None
        display = data.get("displayName") or display_name_from_email(email)
        return {"email": email, "display_name": str(display), "source": "scim_me"}
    except Exception:
        return None


def sdk_available() -> bool:
    try:
        import databricks.sdk  # noqa: F401

        return True
    except Exception:
        return False
