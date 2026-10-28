from __future__ import annotations

import logging
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Allow local imports of agent package during development / deploy layout
ROOT = Path(__file__).resolve().parents[3]
AGENT_SRC = ROOT / "agent" / "src"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from dotenv import load_dotenv

    # Prefer repo .env over ambient shell DATABRICKS_HOST (often another workspace).
    load_dotenv(ROOT / ".env", override=True)
except Exception:
    pass

from app.agent_client import agent_mode, invoke_agent, invoke_agent_stream  # noqa: E402
from app.history_store import HistoryStore  # noqa: E402
from app.local_react.langsmith_trace import ContextIterator  # noqa: E402
from app.request_observability import (  # noqa: E402
    RequestObservabilityMiddleware,
    attach_request_timing,
    observed_call,
    observation_for,
)
from dualsource.tools.auth import (  # noqa: E402
    display_name_from_email,
    initials_from,
    looks_like_email,
    resolve_user_from_obo,
    set_obo_token,
)

class ClosingStreamingResponse(StreamingResponse):
    """Close synchronous body iterators explicitly on completion/disconnect."""

    def __init__(self, content, **kwargs):
        self._source_iterator = content
        super().__init__(content, **kwargs)

    async def __call__(self, scope, receive, send):
        import anyio
        try:
            await super().__call__(scope, receive, send)
        finally:
            close = getattr(self._source_iterator, "close", None)
            if close is not None:
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(close)


APP_TITLE = "DualSource Agent"
FILES_DIR = Path(os.getenv("CI_FILES_DIR") or (ROOT / "tmp_files"))
FILES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(os.getenv("CI_STATIC_DIR") or (BACKEND_DIR / "static"))

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

history_store = HistoryStore()
log = logging.getLogger("ci.chat")

CLIENT_HISTORY_MAX_TURNS = 12
CLIENT_HISTORY_MAX_CHARS_EACH = 8000
CLIENT_HISTORY_MAX_CHARS_TOTAL = 40000


def _normalize_claim_citations(payload: dict[str, Any]) -> dict[str, Any]:
    """Re-attach URLs when the model emits [Label] without (url).

    GPT-OSS sometimes strips URLs from claim-level citation tags. The App can
    safely reinject them from the structured sources list so FE chips stay clickable.
    """
    import re

    answer = str(payload.get("answer_markdown") or "")
    sources = list(payload.get("sources") or [])
    if not answer:
        return payload
    # FE chips need [Internal], not bare Internal prose.
    if re.search(r"(?<!\[)\bInternal\b(?!\])", answer):
        answer = re.sub(r"(?<!\[)\bInternal\b(?!\])", "[Internal]", answer)
        out = dict(payload)
        out["answer_markdown"] = answer
        payload = out
    if not sources:
        return payload

    # Map label / domain-ish keys -> first source URL.
    by_label: dict[str, str] = {}
    for src in sources:
        if not isinstance(src, dict):
            continue
        url = str(src.get("url") or "").strip()
        if not url:
            continue
        label = str(src.get("label") or "").strip()
        if label:
            by_label[label.lower()] = url
        # Also index bare domain for [cna.asia]-style tags.
        try:
            host = url.split("//", 1)[-1].split("/", 1)[0].lower()
            if host.startswith("www."):
                host = host[4:]
            if host:
                by_label[host] = url
        except Exception:
            pass

    if not by_label:
        return payload

    def repl(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        # Skip already-linked markdown and Internal internal marker.
        if label.upper() == "Internal":
            return match.group(0)
        url = by_label.get(label.lower())
        if not url:
            # Fuzzy: label contained in source label or vice versa.
            low = label.lower()
            for k, v in by_label.items():
                if low in k or k in low:
                    url = v
                    break
        if not url:
            return match.group(0)
        return f"[{label}]({url})"

    # Replace [Label] that is NOT already [Label](url)
    fixed = re.sub(r"\[([^\]\n]{1,120})\](?!\()", repl, answer)

    # Long page titles with "|" break GFM tables when used as cite labels inside cells.
    # Rewrite those (and other oversized labels) to short [domain](url) chips.
    def rewrite_pipe_cite(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        url = match.group(2).strip()
        if "|" not in label and len(label) <= 40:
            return match.group(0)
        try:
            host = url.split("//", 1)[-1].split("/", 1)[0].lower()
            if host.startswith("www."):
                host = host[4:]
        except Exception:
            host = ""
        return f"[{host or 'source'}]({url})"

    fixed = re.sub(
        r"\[([^\]\n]{1,200})\]\((https?://[^)\s]+)\)",
        rewrite_pipe_cite,
        fixed,
    )
    if fixed == answer:
        return payload
    out = dict(payload)
    out["answer_markdown"] = fixed
    return out


class ClientHistoryTurn(BaseModel):
    role: str
    content: str = ""
    tables: list[dict[str, Any]] | None = None
    sources: list[dict[str, Any]] | None = None


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)
    client_history: list[ClientHistoryTurn] | None = None


def _truncate_history_tables(tables: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for table in (tables or [])[:3]:
        if not isinstance(table, dict):
            continue
        cols = [str(c) for c in list(table.get("columns") or [])[:20]]
        rows = list(table.get("rows") or [])[:40]
        out.append(
            {
                "title": str(table.get("title") or "")[:200],
                "columns": cols,
                "rows": rows,
            }
        )
    return out


def _sanitize_client_history(raw: list[ClientHistoryTurn] | None) -> list[dict[str, Any]] | None:
    """Use FE last-12 as orchestrator context only. Never persist. None = fall back to UC."""
    if not raw:
        return None
    out: list[dict[str, Any]] = []
    total = 0
    for turn in raw[-CLIENT_HISTORY_MAX_TURNS:]:
        role = str(getattr(turn, "role", "") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(getattr(turn, "content", "") or "")
        if len(content) > CLIENT_HISTORY_MAX_CHARS_EACH:
            content = content[:CLIENT_HISTORY_MAX_CHARS_EACH]
        total += len(content)
        if total > CLIENT_HISTORY_MAX_CHARS_TOTAL:
            break
        item: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant":
            tables = _truncate_history_tables(getattr(turn, "tables", None))
            if tables:
                item["tables"] = tables
            sources = list(getattr(turn, "sources", None) or [])[:8]
            clean_sources = []
            for src in sources:
                if not isinstance(src, dict):
                    continue
                clean_sources.append(
                    {
                        "label": str(src.get("label") or "")[:120],
                        "url": str(src.get("url") or "")[:500],
                        "snippet": str(src.get("snippet") or "")[:240],
                    }
                )
            if clean_sources:
                item["sources"] = clean_sources
        out.append(item)
    return out or None


class SessionPatch(BaseModel):
    title: str | None = None
    status: str | None = None  # active|archived|deleted


def _wants_xlsx(question: str, payload: dict[str, Any] | None = None) -> bool:
    """True only for clear Excel-export intent.

    Must NOT trigger on meta/complaint turns like:
    - "did i tell u to give me an excel????"
    - "why excel...."
    - "did i ask u for an excel????"
    """
    import re

    q = (question or "").lower().strip()
    if not q:
        return False

    # Meta / negation / complaint about Excel → never export.
    meta_markers = (
        "did i tell",
        "did i ask",
        "did you give",
        "do i need",
        "why excel",
        "why an excel",
        "why the excel",
        "why export",
        "why as excel",
        "don't want excel",
        "dont want excel",
        "do not want excel",
        "no excel",
        "not excel",
        "without excel",
        "stop excel",
        "don't export",
        "dont export",
        "do not export",
        "didn't ask",
        "didnt ask",
        "did not ask",
        "never asked",
        "didn't tell",
        "didnt tell",
    )
    if any(m in q for m in meta_markers):
        return False
    # Any Excel mention that is a question/complaint, not an export request.
    if re.search(r"\b(excel|xlsx|spreadsheet|workbook)\b", q):
        if re.search(
            r"\b(why|did i|didn't|didnt|don't|dont|do not|not ask|without|stop)\b",
            q,
        ):
            return False
    if q in {"excel", "excel?", "excel!", "xlsx", "xlsx?"}:
        return False

    # Explicit export / spreadsheet intents.
    explicit = (
        "export this as excel",
        "export as excel",
        "export to excel",
        "export in excel",
        "download as excel",
        "download excel",
        "save as excel",
        "export excel",
        "as an excel",
        "as excel",
        "in excel",
        "to excel",
        ".xlsx",
        "spreadsheet",
        "workbook",
    )
    if any(p in q for p in explicit):
        return True
    if re.search(r"\bxlsx\b", q):
        return True

    # Bare "export this" / "export the data" / "download this" — but not PPT.
    if re.search(r"\b(export|download)\b", q):
        if any(k in q for k in ("ppt", "powerpoint", "slide")):
            return False
        if any(k in q for k in ("why", "did i", "don't", "dont", "not ")):
            return False
        return True

    return False


def _asked_ppt(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in ("ppt", "powerpoint", "slide deck")) or bool(
        __import__("re").search(r"\bslides?\b", q)
    )


def _fast_path_response(question: str) -> dict[str, Any] | None:
    """Short-circuit greetings / identity / help without Genie/Tavily.

    Modern chat apps answer these in <2s; do not pay full agent latency.
    PPT/Excel meta Q&A is intentionally NOT hardcoded here — agent/Genie owns it.
    """
    import re

    q = (question or "").strip()
    qn = re.sub(r"\s+", " ", q.lower()).strip(" !?.,")
    if not qn:
        return None

    def _pack(answer: str, followups: list[str]) -> dict[str, Any]:
        return {
            "answer_markdown": answer,
            "sources": [],
            "tables": [],
            "export_tables": [],
            "charts": [],
            "files": [],
            "followups": followups,
            "status": {"genie": "skipped", "web": "skipped", "files": "skipped"},
            "trace_id": None,
        }

    is_greeting = bool(
        re.fullmatch(
            r"(hi|hello|hey|hiya|yo|good morning|good afternoon|good evening)( there)?",
            qn,
        )
    )
    is_thanks = qn in {"thanks", "thank you", "thx", "ty"}
    is_identity = qn in {
        "explain yourself",
        "who are you",
        "what are you",
        "what can you do",
        "help",
        "how can you help",
        "introduce yourself",
        "about yourself",
    } or qn.startswith("explain yourself")

    if not (is_greeting or is_thanks or is_identity):
        return None

    if is_thanks:
        answer = "You're welcome — ask whenever you need hub traffic, cargo, flights, or baggage insights."
        followups = [
            "How many passengers has hub handled YTD?",
            "Show me the monthly passenger trend",
        ]
    elif is_greeting:
        answer = (
            "Hi — I'm **DualSource Agent**, your Internal assistant for hub analytics.\n\n"
            "I can help with passenger traffic, cargo, flight operations, baggage performance, "
            "and peer-airport comparisons with sources."
        )
        followups = [
            "How many passengers has hub handled YTD?",
            "Show hub cargo trend for the last 6 months as a chart",
            "Compare hub and Shanghai cargo volumes with sources",
        ]
    else:
        answer = (
            "I'm **DualSource Agent**, an hub analytics assistant powered by Internal internal data "
            "and public sources when needed.\n\n"
            "I can help with:\n"
            "- **Passenger traffic** — YTD totals, monthly trends, airline/region share\n"
            "- **Cargo volumes** — tonnage, growth, carrier/airport breakdowns\n"
            "- **Flight operations** — ATM / flight activity trends\n"
            "- **Baggage handling** — volumes and dwell performance\n"
            "- **Peer comparisons** — hub vs other airports, with source grounding\n\n"
            "Ask a business question, or request a chart / Excel export when useful."
        )
        followups = [
            "How many passengers has hub handled YTD?",
            "Show me the monthly trend",
            "What is hub YTD cargo volume?",
        ]

    return _pack(answer, followups)


def _export_tables_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    if message.get("export_tables_json"):
        return list(message.get("export_tables_json") or [])
    status = message.get("tool_status_json") or {}
    if isinstance(status, dict) and status.get("_export_tables"):
        return list(status.get("_export_tables") or [])
    return []


def _pick_export_context(
    payload: dict[str, Any],
    prior: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    answer = str(payload.get("answer_markdown") or "")
    tables = list(payload.get("tables") or [])
    export_tables = list(payload.get("export_tables") or [])
    sources = list(payload.get("sources") or [])
    low = answer.lower()
    weak = (
        "i don't have data" in low
        or "cannot generate actual" in low
        or "i cannot generate" in low
        or "your export is ready" in low
        or not answer.strip()
    )
    if weak or not tables or not export_tables:
        for m in reversed(prior):
            if m.get("role") != "assistant":
                continue
            if weak and (m.get("content_markdown") or "").strip():
                # Prefer prior analytical answer over export boilerplate.
                prior_text = str(m.get("content_markdown") or "")
                if prior_text and "your export is ready" not in prior_text.lower():
                    answer = prior_text
            if not tables and m.get("tables_json"):
                tables = list(m.get("tables_json") or [])
            if not export_tables:
                export_tables = _export_tables_from_message(m)
            if not sources and m.get("sources_json"):
                sources = list(m.get("sources_json") or [])
            if answer and (export_tables or tables):
                break
    return answer, tables, export_tables, sources


def materialize_exports_locally(
    *,
    question: str,
    payload: dict[str, Any],
    prior: list[dict[str, Any]],
    user_email: str,
    session_id: str,
) -> dict[str, Any]:
    """Regenerate XLSX on App local disk so /api/files can serve them.

    Model Serving writes files inside the serving container; the App process cannot
    read that filesystem. Generating on the App side makes downloads work.

    Prefer `export_tables` (full rows) over chat `tables` (display-truncated).
    PPT is disabled in Phase 1.
    """
    asked_ppt = _asked_ppt(question)
    need_xlsx = _wants_xlsx(question, payload)
    # PPT alone must never materialize as Excel. Do not overwrite agent wording.
    if asked_ppt and not need_xlsx:
        out = dict(payload)
        out["files"] = []
        status = dict(out.get("status") or {})
        status["files"] = "skipped"
        out["status"] = status
        return out

    if not need_xlsx:
        # Strip serving-side file artifacts when user did not ask to export.
        # Keep agent answer text — App no longer hardcodes Excel-meta replies.
        out = dict(payload)
        out["files"] = []
        status = dict(out.get("status") or {})
        status["files"] = "skipped"
        out["status"] = status
        qlow = (question or "").lower()
        excel_meta = any(
            marker in qlow
            for marker in (
                "did i ask",
                "did i tell",
                "why excel",
                "why export",
                "no excel",
                "not excel",
                "don't want excel",
                "dont want excel",
            )
        )
        if excel_meta:
            cleaned = []
            for fu in list(out.get("followups") or []):
                t = str(fu or "").lower()
                if any(k in t for k in ("excel", "xlsx", "spreadsheet", "workbook", "worksheet")):
                    continue
                cleaned.append(fu)
            out["followups"] = cleaned
        return out

    from dualsource.tools.documents import generate_xlsx

    answer, tables, export_tables, _sources = _pick_export_context(payload, prior)
    export_table = (export_tables[0] if export_tables else None) or (tables[0] if tables else None)
    out = dict(payload)
    # Keep full export payload available to history even on export-only turns.
    if export_tables:
        out["export_tables"] = export_tables
    elif export_table:
        out["export_tables"] = [export_table]

    if not export_table or not (export_table.get("rows") or []):
        status = dict(out.get("status") or {})
        status["files"] = "error"
        out["status"] = status
        out["files"] = []
        out["answer_markdown"] = (
            "I couldn't find tabular data to export.\n\n"
            "Ask a data question first, then try **Export this as Excel**."
        )
        out["followups"] = ["How many passengers has hub handled YTD?"]
        return out

    xls = generate_xlsx(
        title="DualSource Agent Export",
        summary=answer[:2000],
        table=export_table,
        output_dir=str(FILES_DIR),
        user_email=user_email,
        session_id=session_id,
    )
    if xls.status != "ok" or not xls.files:
        status = dict(out.get("status") or {})
        status["files"] = "error"
        out["status"] = status
        out["files"] = []
        out["answer_markdown"] = "Excel export failed. Please try again."
        return out

    out["files"] = xls.files
    status = dict(out.get("status") or {})
    status["files"] = "ok"
    out["status"] = status
    row_n = len(export_table.get("rows") or [])
    out["answer_markdown"] = (
        f"Excel export ready (**{row_n}** data rows).\n\n"
        "Use the download button below to save your file."
    )
    out["charts"] = []
    out["followups"] = ["Show me the latest passenger trends"]
    return out


def _pick_real_email(*candidates: str | None) -> str | None:
    for c in candidates:
        if looks_like_email(c):
            return str(c).strip().lower()
    return None


def _resolve_current_user(
    request: Request,
    x_forwarded_email: str | None = Header(default=None),
    x_forwarded_user: str | None = Header(default=None),
    x_forwarded_preferred_username: str | None = Header(default=None),
    x_user_email: str | None = Header(default=None),
    x_user_name: str | None = Header(default=None),
) -> dict[str, str]:
    # Request-scoped OBO token (never stash into process-global env).
    obo = request.headers.get("x-forwarded-access-token")
    set_obo_token(obo)

    email = _pick_real_email(
        x_forwarded_email,
        x_forwarded_preferred_username,
        x_user_email,
        os.getenv("CI_LOCAL_USER_EMAIL"),
        os.getenv("DEV_USER_EMAIL"),
    )
    source = "header"

    # If Apps sent numeric id@workspace as "email", resolve via SCIM Me.
    if email is None and obo:
        resolved = resolve_user_from_obo(obo)
        if resolved:
            email = resolved["email"]
            display = resolved["display_name"]
            source = resolved.get("source") or "scim_me"
            return {
                "email": email,
                "display_name": display,
                "initials": initials_from(display, email),
                "identity_source": source,
            }

    if email is None:
        # Local/dev fallback only. Never use x-forwarded-user numeric id as email.
        email = (os.getenv("CI_LOCAL_USER_EMAIL") or "steven.yung@example.com").strip().lower()
        source = "fallback"

    # Display name: never show workspace numeric user id.
    display = None
    for c in (x_user_name, x_forwarded_preferred_username):
        if not c:
            continue
        c = str(c).strip()
        if c.isdigit():
            continue
        if "@" in c and c.split("@", 1)[0].isdigit():
            continue
        if looks_like_email(c):
            display = display_name_from_email(c)
        else:
            display = c
        break
    if not display:
        local_name = (os.getenv("CI_LOCAL_USER_NAME") or "").strip()
        display = local_name or display_name_from_email(email)

    # Keep lint quiet: x_forwarded_user is a stable id, not display identity.
    _ = x_forwarded_user

    return {
        "email": email,
        "display_name": display,
        "initials": initials_from(display, email),
        "identity_source": source,
    }


def current_user(
    request: Request,
    x_forwarded_email: str | None = Header(default=None),
    x_forwarded_user: str | None = Header(default=None),
    x_forwarded_preferred_username: str | None = Header(default=None),
    x_user_email: str | None = Header(default=None),
    x_user_name: str | None = Header(default=None),
) -> dict[str, str]:
    return observed_call(
        request,
        "identity",
        _resolve_current_user,
        request,
        x_forwarded_email=x_forwarded_email,
        x_forwarded_user=x_forwarded_user,
        x_forwarded_preferred_username=x_forwarded_preferred_username,
        x_user_email=x_user_email,
        x_user_name=x_user_name,
    )


@app.middleware("http")
async def clear_obo_after_request(request: Request, call_next):
    try:
        return await call_next(request)
    finally:
        set_obo_token(None)


# Outer ASGI middleware (register after @app.middleware so it stays first).
app.add_middleware(RequestObservabilityMiddleware)


@app.get("/api/health")
def health() -> dict[str, Any]:
    reasoning = None
    if agent_mode() == "local_react":
        from app.local_react.foundry_client import FoundryClient
        client = FoundryClient()
        reasoning = {
            "configured": client.reasoning_effort,
            "fast": client.reasoning_settings(client.fast_model),
            "strong": client.reasoning_settings(client.strong_model),
        }
    return {
        "status": "ok",
        "service": APP_TITLE,
        "history_backend": history_store.mode,
        "history_backend_requested": getattr(history_store, "requested_mode", None) or history_store.mode,
        "history_demote_reason": getattr(history_store, "demote_reason", None),
        "history_warehouse_id": getattr(history_store, "warehouse_id", None) or None,
        "agent_mode": agent_mode(),
        "local_react_harness": os.getenv("CI_LOCAL_REACT_HARNESS", "langchain") if agent_mode() == "local_react" else None,
        "ai_endpoint": os.getenv("AI_ENDPOINT_NAME") or None,
        "llm_provider": os.getenv("CI_LLM_PROVIDER") or None,
        "llm_model_fast": os.getenv("CI_LLM_MODEL_FAST") or None,
        "llm_model_fast_backup": (
            os.getenv("CI_LLM_MODEL_FAST_BACKUP") or "gpt-5.4-nano"
        ) if agent_mode() == "local_react" else None,
        "llm_model_strong": os.getenv("CI_LLM_MODEL_STRONG") or None,
        "llm_reasoning": reasoning,
        "wren_mcp_url": os.getenv("WREN_MCP_URL") or None,
        "stream_enabled": (os.getenv("CI_STREAM_ENABLED") or "true").lower() in {"1", "true", "yes", "on"},
    }


@app.get("/api/me")
def me(user: dict[str, str] = Depends(current_user)) -> dict[str, str]:
    return user


@app.get("/api/sessions")
def list_sessions(user: dict[str, str] = Depends(current_user)) -> list[dict[str, Any]]:
    try:
        return history_store.list_sessions(user["email"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"list sessions failed: {exc}") from exc


@app.post("/api/sessions")
def create_session(user: dict[str, str] = Depends(current_user)) -> dict[str, Any]:
    try:
        return history_store.create_session(user["email"], user["display_name"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"create session failed: {exc}") from exc


@app.patch("/api/sessions/{session_id}")
def patch_session(
    session_id: str,
    body: SessionPatch,
    user: dict[str, str] = Depends(current_user),
) -> dict[str, Any]:
    try:
        return history_store.patch_session(
            user["email"], session_id, title=body.title, status=body.status
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str, user: dict[str, str] = Depends(current_user)) -> list[dict[str, Any]]:
    try:
        return history_store.get_messages(user["email"], session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found") from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"get messages failed: {exc}") from exc


@app.post("/api/chat")
def chat(body: ChatRequest, request: Request, user: dict[str, str] = Depends(current_user)) -> dict[str, Any]:
    try:
        t0 = time.time()
        # Detect greeting/help early so session+history can skip sync UC latency.
        fast_payload = observed_call(request, "fast_path", _fast_path_response, body.message)
        session_id = body.session_id
        created_new = False
        if not session_id:
            created = observed_call(
                request,
                "history.create_session",
                history_store.create_session,
                user["email"],
                user["display_name"],
                defer_uc=True,
            )
            session_id = created["session_id"]
            created_new = True

        session = observed_call(request, "history.get_session", history_store.get_session, user["email"], session_id)
        if not session:
            raise HTTPException(status_code=404, detail="session not found for this user")
        t_session = time.time()

        # Fast path for greetings / identity / help — skip history read + agent.
        if fast_payload is not None:
            payload = fast_payload
            assistant_msg = observed_call(
                request,
                "history.enqueue_write",
                history_store.append_turn,
                user_email=user["email"],
                session_id=session_id,
                user_message=body.message,
                assistant_payload=payload,
                title_if_new=body.message,
                defer_uc=True,
            )
            log.info(
                "chat fast-path session_ms=%.0f append_ms=%.0f total_ms=%.0f",
                (t_session - t0) * 1000,
                (time.time() - t_session) * 1000,
                (time.time() - t0) * 1000,
            )
            return {
                "session_id": session_id,
                "message": assistant_msg,
                "response": attach_request_timing(request, payload),
                "history_source": "none",
            }

        history = observed_call(request, "history.client_context", _sanitize_client_history, body.client_history)
        if history is not None:
            history_source = "client"
        elif created_new:
            history = []
            history_source = "empty"
        else:
            history_source = "uc"
            try:
                history = observed_call(
                    request,
                    "history.read",
                    history_store.get_recent_text_turns,
                    user["email"],
                    session_id,
                    limit=12,
                    ensure_session=False,
                )
            except KeyError:
                history = []
        t_hist = time.time()

        payload = observed_call(
            request,
            "agent.invoke",
            invoke_agent,
            question=body.message,
            history=history,
            user_email=user["email"],
            session_id=session_id,
            files_dir=str(FILES_DIR),
        )
        payload = observed_call(request, "citations", _normalize_claim_citations, payload)
        t_agent = time.time()

        prior: list[dict[str, Any]] = []
        if _wants_xlsx(body.message, payload):
            try:
                latest = observed_call(
                    request,
                    "history.export_context",
                    history_store.get_latest_assistant_export,
                    user["email"],
                    session_id,
                    ensure_session=False,
                )
            except KeyError:
                latest = None
            if latest:
                prior = [latest]
        payload = observed_call(
            request,
            "exports",
            materialize_exports_locally,
            question=body.message,
            payload=payload,
            prior=prior,
            user_email=user["email"],
            session_id=session_id,
        )
        t_xlsx = time.time()
        assistant_msg = observed_call(
            request,
            "history.enqueue_write",
            history_store.append_turn,
            user_email=user["email"],
            session_id=session_id,
            user_message=body.message,
            assistant_payload=payload,
            title_if_new=body.message,
            defer_uc=True,
        )
        log.info(
            "chat history_source=%s session_ms=%.0f hist_ms=%.0f agent_ms=%.0f xlsx_ms=%.0f append_ms=%.0f total_ms=%.0f",
            history_source,
            (t_session - t0) * 1000,
            (t_hist - t_session) * 1000,
            (t_agent - t_hist) * 1000,
            (t_xlsx - t_agent) * 1000,
            (time.time() - t_xlsx) * 1000,
            (time.time() - t0) * 1000,
        )
        return {
            "session_id": session_id,
            "message": assistant_msg,
            "response": attach_request_timing(request, payload),
            "history_source": history_source,
        }
    except HTTPException:
        raise
    except Exception as exc:
        _ = traceback.format_exc(limit=4)
        detail = str(exc).strip() or type(exc).__name__
        # Keep user-facing errors short; full traceback stays server-side.
        if len(detail) > 240:
            detail = detail[:240] + "…"
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed — please retry. ({detail})",
        ) from exc



@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest, request: Request, user: dict[str, str] = Depends(current_user)):
    """SSE chat: stage progress, token deltas, final payload."""
    import json as _json

    def _sse(obj: dict) -> str:
        observation = observation_for(request)
        if observation:
            observation.mark_event(obj)
            if obj.get("type") == "final":
                obj = dict(obj)
                obj["response"] = attach_request_timing(request, obj.get("response") or {})
        return "data: " + _json.dumps(obj, ensure_ascii=False, default=str) + chr(10) + chr(10)

    # Fast-path greetings still stream as stage + final for FE uniformity.
    fast_payload = observed_call(request, "fast_path", _fast_path_response, body.message)

    session_id = body.session_id
    created_new = False
    if not session_id:
        created = observed_call(
            request,
            "history.create_session",
            history_store.create_session,
            user["email"],
            user["display_name"],
            defer_uc=True,
        )
        session_id = created["session_id"]
        created_new = True

    session = observed_call(request, "history.get_session", history_store.get_session, user["email"], session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found for this user")

    history = observed_call(request, "history.client_context", _sanitize_client_history, body.client_history)
    if history is not None:
        history_source = "client"
    elif created_new:
        history = []
        history_source = "empty"
    else:
        history_source = "uc"
        try:
            history = observed_call(
                request,
                "history.read",
                history_store.get_recent_text_turns,
                user["email"],
                session_id,
                limit=12,
                ensure_session=False,
            )
        except KeyError:
            history = []

    def event_gen():
        agent_stream = None
        try:
            if fast_payload is not None:
                yield _sse({"type": "stage", "stage": "route", "status": "start", "label": "Understanding question", "ts": time.time()})
                yield _sse({"type": "stage", "stage": "route", "status": "end", "label": "Understanding question", "ts": time.time()})
                yield _sse({"type": "stage", "stage": "complete", "status": "end", "label": "Complete", "ts": time.time()})
                if fast_payload.get("answer_markdown"):
                    yield _sse({"type": "token", "delta": str(fast_payload.get("answer_markdown"))})
                assistant_msg = observed_call(
                    request,
                    "history.enqueue_write",
                    history_store.append_turn,
                    user_email=user["email"],
                    session_id=session_id,
                    user_message=body.message,
                    assistant_payload=fast_payload,
                    title_if_new=body.message,
                    defer_uc=True,
                )
                yield _sse({
                    "type": "final",
                    "session_id": session_id,
                    "message": assistant_msg,
                    "response": fast_payload,
                    "history_source": "none",
                })
                return

            payload = None
            agent_stream = invoke_agent_stream(
                question=body.message,
                history=history,
                user_email=user["email"],
                session_id=session_id,
                files_dir=str(FILES_DIR),
            )
            for event in agent_stream:
                et = event.get("type")
                if et == "final":
                    payload = event.get("response") or {}
                    payload = observed_call(request, "citations", _normalize_claim_citations, payload)
                    # Excel optional / disabled for spike; keep strip behavior.
                    payload = observed_call(
                        request,
                        "exports",
                        materialize_exports_locally,
                        question=body.message,
                        payload=payload,
                        prior=[],
                        user_email=user["email"],
                        session_id=session_id,
                    )
                    # History persistence must never kill the final frame:
                    # an append failure used to surface as an SSE error and the
                    # FE would repaint the answer WITHOUT charts/tables. Log it
                    # and deliver the payload with a stub message instead.
                    try:
                        assistant_msg = observed_call(
                            request,
                            "history.enqueue_write",
                            history_store.append_turn,
                            user_email=user["email"],
                            session_id=session_id,
                            user_message=body.message,
                            assistant_payload=payload,
                            title_if_new=body.message,
                            defer_uc=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "chat_stream history append failed (session=%s): %s",
                            session_id,
                            exc,
                        )
                        assistant_msg = {
                            "message_id": uuid.uuid4().hex,
                            "session_id": session_id,
                            "user_email": user["email"].lower(),
                            "role": "assistant",
                            "content_markdown": payload.get("answer_markdown"),
                            "sources_json": payload.get("sources") or [],
                            "tables_json": payload.get("tables") or [],
                            "export_tables_json": payload.get("export_tables") or [],
                            "charts_json": payload.get("charts") or [],
                            "files_meta_json": payload.get("files") or [],
                            "tool_status_json": dict(payload.get("status") or {}),
                            "trace_id": payload.get("trace_id"),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "history_write_failed": True,
                        }
                    yield _sse({
                        "type": "final",
                        "session_id": session_id,
                        "message": assistant_msg,
                        "response": payload,
                        "history_source": history_source,
                    })
                elif et == "error":
                    yield _sse({"type": "error", "detail": event.get("detail") or "stream failed"})
                else:
                    yield _sse(event)
            if payload is None:
                yield _sse({"type": "error", "detail": "stream ended without final payload"})
        except Exception as exc:  # noqa: BLE001
            detail = str(exc).strip() or type(exc).__name__
            if len(detail) > 240:
                detail = detail[:240] + "…"
            yield _sse({"type": "error", "detail": f"Chat stream failed — please retry. ({detail})"})
        finally:
            if agent_stream is not None:
                agent_stream.close()

    return ClosingStreamingResponse(
        ContextIterator(event_gen()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/files/{file_id}")
def download_file(file_id: str, user: dict[str, str] = Depends(current_user)):
    user_root = FILES_DIR / user["email"]
    if not user_root.exists():
        raise HTTPException(status_code=404, detail="file not found")
    matches = list(user_root.rglob(f"{file_id}_*"))
    if not matches:
        raise HTTPException(status_code=404, detail="file not found")
    path = matches[0]
    return FileResponse(path, filename=path.name)


# Serve built React frontend when present (Databricks Apps single-process model).
# Mount hashed assets first so missing SPA fallback never returns HTML for JS/CSS.
assets_dir = STATIC_DIR / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


@app.get("/")
def root_page():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"service": APP_TITLE, "hint": "Frontend not built yet. Use /api/health"}


@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    # Never hijack API routes.
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="not found")
    # Serve known static files (logo etc.) directly.
    candidate = STATIC_DIR / full_path
    if candidate.exists() and candidate.is_file():
        return FileResponse(candidate)
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="not found")
