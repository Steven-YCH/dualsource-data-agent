from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


class HistoryStore:
    """Conversation history store.

    Modes:
    - local: JSON files under CI_HISTORY_DIR (dev fallback)
    - uc: Databricks SQL + UC tables with RLS
    """

    def __init__(
        self,
        mode: str | None = None,
        host: str | None = None,
        token: str | None = None,
        warehouse_id: str | None = None,
        history_dir: str | Path | None = None,
    ) -> None:
        self.requested_mode = (mode or os.getenv("CI_HISTORY_BACKEND", "local") or "local").lower()
        self.mode = self.requested_mode
        from dualsource.tools.auth import normalize_workspace_host

        self.host = normalize_workspace_host(
            host or os.getenv("DATABRICKS_HOST") or os.getenv("TARGET_HOST") or ""
        )
        self._explicit_token = token or ""
        # Databricks Apps value_from:sql-warehouse should inject these; keep prod-002
        # warehouse as last-resort default so UC history does not silently demote.
        self.warehouse_id = (
            warehouse_id
            or os.getenv("DATABRICKS_WAREHOUSE_ID")
            or os.getenv("SQL_WAREHOUSE_ID")
            or os.getenv("TARGET_WAREHOUSE_ID")
            or os.getenv("DATABRICKS_SQL_WAREHOUSE_ID")
            or "db96019f77087a03"
        )
        self.catalog = os.getenv("CI_UC_CATALOG", "local_duckdb")
        self.schema = os.getenv("CI_UC_SCHEMA", "dualsource")
        self.history_dir = Path(history_dir or os.getenv("CI_HISTORY_DIR") or "./tmp_history")
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.demote_reason: str | None = None
        # UC mode can use SDK auth in Apps even without PAT; only require warehouse_id.
        if self.mode == "uc" and not self.warehouse_id:
            self.mode = "local"
            self.demote_reason = "missing_warehouse_id"
        # preferred_mode reflects operator intent; actual mode may fall back per-call.
        self.preferred_mode = self.requested_mode

    def list_sessions(self, user_email: str) -> list[dict[str, Any]]:
        local = self._list_sessions_local(user_email)
        if self.mode == "uc":
            try:
                rows = self._sql(
                    f"""
                    SELECT session_id, user_email, user_display_name, title, status,
                           created_at, updated_at, expires_at, last_trace_id
                    FROM {self.catalog}.{self.schema}.ci_chat_sessions
                    WHERE user_email = :email AND status != 'deleted'
                    ORDER BY updated_at DESC
                    """,
                    params={"email": user_email.lower()},
                )
                uc_sessions = [self._session_row(r) for r in rows]
                # Merge deferred local-only sessions (greeting fast-path create).
                by_id = {s["session_id"]: s for s in uc_sessions if s.get("session_id")}
                for s in local:
                    sid = s.get("session_id")
                    if sid and sid not in by_id:
                        by_id[sid] = s
                merged = list(by_id.values())
                merged.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
                return merged
            except Exception:
                # Keep chat usable if UC/SQL is temporarily unavailable.
                return local
        return local

    def create_session(
        self,
        user_email: str,
        display_name: str,
        *,
        defer_uc: bool = False,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        session = {
            "session_id": uuid.uuid4().hex,
            "user_email": user_email.lower(),
            "user_display_name": display_name,
            "title": "New chat",
            "status": "active",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "expires_at": (now + timedelta(days=90)).isoformat(),
            "last_trace_id": None,
        }
        # Greeting/help fast-path: return immediately on local disk; mirror to UC async.
        if defer_uc:
            local = self._create_session_local(session)
            if self.mode == "uc":
                self._spawn_uc_session_insert(local)
            return local
        if self.mode == "uc":
            try:
                self._sql(
                    f"""
                    INSERT INTO {self.catalog}.{self.schema}.ci_chat_sessions
                    (session_id, user_email, user_display_name, title, status, created_at, updated_at, expires_at, last_trace_id)
                    VALUES (:session_id, :user_email, :user_display_name, :title, :status,
                            TIMESTAMP(:created_at), TIMESTAMP(:updated_at), TIMESTAMP(:expires_at), NULL)
                    """,
                    params={
                        "session_id": session["session_id"],
                        "user_email": session["user_email"],
                        "user_display_name": session["user_display_name"],
                        "title": session["title"],
                        "status": session["status"],
                        "created_at": session["created_at"],
                        "updated_at": session["updated_at"],
                        "expires_at": session["expires_at"],
                    },
                )
                # Keep local mirror for Excel follow-ups / offline reads.
                return self._create_session_local(session)
            except Exception:
                return self._create_session_local(session)
        return self._create_session_local(session)

    def patch_session(self, user_email: str, session_id: str, title: str | None = None, status: str | None = None) -> dict[str, Any]:
        sessions = self.list_sessions(user_email)
        # include deleted for patch? no - only active/archived from list. fetch directly
        current = self.get_session(user_email, session_id)
        if not current:
            raise KeyError("session not found")
        if title is not None:
            current["title"] = title[:120]
        if status is not None:
            if status not in {"active", "archived", "deleted"}:
                raise ValueError("invalid status")
            current["status"] = status
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        if self.mode == "uc":
            self._sql(
                f"""
                UPDATE {self.catalog}.{self.schema}.ci_chat_sessions
                SET title = :title, status = :status, updated_at = TIMESTAMP(:updated_at)
                WHERE session_id = :session_id AND user_email = :email
                """,
                params={
                    "title": current["title"],
                    "status": current["status"],
                    "updated_at": current["updated_at"],
                    "session_id": session_id,
                    "email": user_email.lower(),
                },
            )
            return current
        store = self._load_local(user_email)
        store["sessions"][session_id] = current
        self._save_local(user_email, store)
        return current

    def get_session(self, user_email: str, session_id: str) -> dict[str, Any] | None:
        if self.mode == "uc":
            try:
                rows = self._sql(
                    f"""
                    SELECT session_id, user_email, user_display_name, title, status,
                           created_at, updated_at, expires_at, last_trace_id
                    FROM {self.catalog}.{self.schema}.ci_chat_sessions
                    WHERE session_id = :session_id AND user_email = :email
                    LIMIT 1
                    """,
                    params={"session_id": session_id, "email": user_email.lower()},
                )
                if rows:
                    return self._session_row(rows[0])
            except Exception as exc:
                self.demote_reason = f"get_session_uc_failed:{type(exc).__name__}"
        store = self._load_local(user_email)
        s = store["sessions"].get(session_id)
        if not s or s.get("user_email") != user_email.lower():
            return None
        return s

    def get_messages(self, user_email: str, session_id: str) -> list[dict[str, Any]]:
        if not self.get_session(user_email, session_id):
            raise KeyError("session not found")
        local_msgs = list(
            self._load_local(user_email).get("messages", {}).get(session_id, []) or []
        )
        if self.mode == "uc":
            try:
                rows = self._sql(
                    f"""
                    SELECT message_id, session_id, user_email, role, content_markdown,
                           sources_json, tables_json, charts_json, files_meta_json,
                           tool_status_json, trace_id, created_at
                    FROM {self.catalog}.{self.schema}.ci_chat_messages
                    WHERE session_id = :session_id AND user_email = :email
                    ORDER BY created_at ASC
                    """,
                    params={"session_id": session_id, "email": user_email.lower()},
                )
                messages = [self._message_row(r) for r in rows]
                # Fast-path / deferred UC writes land locally first. Prefer local when
                # UC is empty or behind so FE refresh after greeting still shows turns.
                if not messages and local_msgs:
                    return local_msgs
                if local_msgs and len(local_msgs) > len(messages):
                    return local_msgs
                # Merge locally mirrored export_tables (kept out of UC status JSON).
                by_id = {
                    m.get("message_id"): m
                    for m in local_msgs
                    if isinstance(m, dict) and m.get("message_id")
                }
                for m in messages:
                    loc = by_id.get(m.get("message_id")) or {}
                    export_tables = list(loc.get("export_tables_json") or [])
                    if export_tables:
                        m["export_tables_json"] = export_tables
                        status = dict(m.get("tool_status_json") or {})
                        status["_export_tables"] = export_tables
                        m["tool_status_json"] = status
                return messages
            except Exception as exc:
                self.demote_reason = f"get_messages_uc_failed:{type(exc).__name__}"
        store = self._load_local(user_email)
        return store["messages"].get(session_id, [])

    def get_recent_text_turns(
        self,
        user_email: str,
        session_id: str,
        limit: int = 12,
        *,
        ensure_session: bool = True,
    ) -> list[dict[str, str]]:
        """Last N turns as {role, content} for Serving — no tables/charts blobs."""
        if ensure_session and not self.get_session(user_email, session_id):
            raise KeyError("session not found")
        lim = max(1, min(int(limit), 24))

        def as_text(msgs: list[dict[str, Any]]) -> list[dict[str, str]]:
            out: list[dict[str, str]] = []
            for m in msgs or []:
                role = str(m.get("role") or "").lower()
                if role not in {"user", "assistant"}:
                    continue
                out.append(
                    {
                        "role": role,
                        "content": str(m.get("content_markdown") or m.get("content") or ""),
                        "message_id": str(m.get("message_id") or ""),
                        "created_at": str(m.get("created_at") or ""),
                    }
                )
            return out

        local = as_text(
            self._load_local(user_email).get("messages", {}).get(session_id, []) or []
        )
        uc_text: list[dict[str, str]] = []
        if self.mode == "uc":
            try:
                rows = self._sql(
                    f"""
                    SELECT role, content_markdown, created_at, message_id
                    FROM {self.catalog}.{self.schema}.ci_chat_messages
                    WHERE session_id = :session_id AND user_email = :email
                    ORDER BY created_at DESC
                    LIMIT {lim}
                    """,
                    params={"session_id": session_id, "email": user_email.lower()},
                )
                uc_text = as_text(
                    [
                        {
                            "role": r[0],
                            "content_markdown": r[1],
                            "created_at": r[2],
                            "message_id": r[3],
                        }
                        for r in reversed(rows or [])
                    ]
                )
            except Exception as exc:
                self.demote_reason = f"get_recent_text_uc_failed:{type(exc).__name__}"

        merged: dict[str, dict[str, str]] = {}
        for m in uc_text + local:
            key = m.get("message_id") or f"{m.get('created_at')}|{m.get('role')}|{(m.get('content') or '')[:48]}"
            merged[key] = m
        ordered = list(merged.values())
        ordered.sort(key=lambda x: x.get("created_at") or "")
        return [{"role": m["role"], "content": m.get("content") or ""} for m in ordered[-lim:]]

    def get_latest_assistant_export(
        self,
        user_email: str,
        session_id: str,
        *,
        ensure_session: bool = True,
    ) -> dict[str, Any] | None:
        """Latest assistant row that still has tables/export payloads (Excel follow-up)."""
        if ensure_session and not self.get_session(user_email, session_id):
            raise KeyError("session not found")

        def has_export(m: dict[str, Any]) -> bool:
            if not isinstance(m, dict) or m.get("role") != "assistant":
                return False
            if m.get("tables_json"):
                return True
            if m.get("export_tables_json"):
                return True
            status = m.get("tool_status_json") or {}
            return isinstance(status, dict) and bool(status.get("_export_tables"))

        def pick(msgs: list[dict[str, Any]]) -> dict[str, Any] | None:
            for m in reversed(msgs or []):
                if has_export(m):
                    return m
            return None

        local_msgs = list(
            self._load_local(user_email).get("messages", {}).get(session_id, []) or []
        )
        local_hit = pick(local_msgs)
        if self.mode == "uc":
            try:
                rows = self._sql(
                    f"""
                    SELECT message_id, session_id, user_email, role, content_markdown,
                           sources_json, tables_json, charts_json, files_meta_json,
                           tool_status_json, trace_id, created_at
                    FROM {self.catalog}.{self.schema}.ci_chat_messages
                    WHERE session_id = :session_id AND user_email = :email AND role = 'assistant'
                    ORDER BY created_at DESC
                    LIMIT 8
                    """,
                    params={"session_id": session_id, "email": user_email.lower()},
                )
                messages = [self._message_row(r) for r in rows]
                by_id = {
                    m.get("message_id"): m
                    for m in local_msgs
                    if isinstance(m, dict) and m.get("message_id")
                }
                for m in messages:
                    loc = by_id.get(m.get("message_id")) or {}
                    export_tables = list(loc.get("export_tables_json") or [])
                    if export_tables:
                        m["export_tables_json"] = export_tables
                        status = dict(m.get("tool_status_json") or {})
                        status["_export_tables"] = export_tables
                        m["tool_status_json"] = status
                    if has_export(m):
                        return m
            except Exception as exc:
                self.demote_reason = f"get_latest_export_uc_failed:{type(exc).__name__}"
        return local_hit

    def append_turn(
        self,
        user_email: str,
        session_id: str,
        user_message: str,
        assistant_payload: dict[str, Any],
        title_if_new: str | None = None,
        *,
        defer_uc: bool = False,
    ) -> dict[str, Any]:
        session = self.get_session(user_email, session_id)
        if not session:
            raise KeyError("session not found")
        now = datetime.now(timezone.utc).isoformat()
        user_msg = {
            "message_id": uuid.uuid4().hex,
            "session_id": session_id,
            "user_email": user_email.lower(),
            "role": "user",
            "content_markdown": user_message,
            "created_at": now,
        }
        export_tables = list(assistant_payload.get("export_tables") or [])
        status_blob = dict(assistant_payload.get("status") or {})
        # Persist full export tables without a UC schema migration by nesting under status.
        if export_tables:
            status_blob["_export_tables"] = export_tables
        assistant_msg = {
            "message_id": uuid.uuid4().hex,
            "session_id": session_id,
            "user_email": user_email.lower(),
            "role": "assistant",
            "content_markdown": assistant_payload.get("answer_markdown"),
            "sources_json": assistant_payload.get("sources") or [],
            "tables_json": assistant_payload.get("tables") or [],
            "export_tables_json": export_tables,
            "charts_json": assistant_payload.get("charts") or [],
            "files_meta_json": [
                f for f in (assistant_payload.get("files") or [])
                if str((f or {}).get("type") or "").lower() != "pptx"
            ],
            "tool_status_json": status_blob,
            "trace_id": assistant_payload.get("trace_id"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        new_title = session.get("title")
        if new_title in {None, "", "New chat"} and title_if_new:
            new_title = title_if_new[:80]
        updated_at = datetime.now(timezone.utc).isoformat()

        # Always persist local first so the App can answer greetings/help quickly.
        local_msg = self._append_turn_local(
            user_email=user_email,
            session=session,
            session_id=session_id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            new_title=new_title,
            updated_at=updated_at,
        )

        if self.mode != "uc":
            return local_msg

        if defer_uc:
            self._spawn_uc_append_turn(
                user_email=user_email,
                session_id=session_id,
                user_msg=user_msg,
                assistant_msg=assistant_msg,
                status_blob=status_blob,
                new_title=new_title,
                updated_at=updated_at,
            )
            return local_msg

        try:
            self._persist_uc_append_turn(
                user_email=user_email,
                session_id=session_id,
                user_msg=user_msg,
                assistant_msg=assistant_msg,
                status_blob=status_blob,
                new_title=new_title,
                updated_at=updated_at,
            )
            return local_msg
        except Exception as exc:
            self.demote_reason = f"append_turn_uc_failed:{type(exc).__name__}:{exc}"[:300]
            return local_msg

    def _spawn_uc_session_insert(self, session: dict[str, Any]) -> None:
        import threading

        def _run() -> None:
            try:
                self._sql(
                    f"""
                    INSERT INTO {self.catalog}.{self.schema}.ci_chat_sessions
                    (session_id, user_email, user_display_name, title, status, created_at, updated_at, expires_at, last_trace_id)
                    VALUES (:session_id, :user_email, :user_display_name, :title, :status,
                            TIMESTAMP(:created_at), TIMESTAMP(:updated_at), TIMESTAMP(:expires_at), NULL)
                    """,
                    params={
                        "session_id": session["session_id"],
                        "user_email": session["user_email"],
                        "user_display_name": session["user_display_name"],
                        "title": session["title"],
                        "status": session["status"],
                        "created_at": session["created_at"],
                        "updated_at": session["updated_at"],
                        "expires_at": session["expires_at"],
                    },
                )
            except Exception as exc:  # noqa: BLE001
                self.demote_reason = f"create_session_uc_async_failed:{type(exc).__name__}:{exc}"[:300]

        threading.Thread(target=_run, name="ci-uc-session", daemon=True).start()

    def _persist_uc_append_turn(
        self,
        *,
        user_email: str,
        session_id: str,
        user_msg: dict[str, Any],
        assistant_msg: dict[str, Any],
        status_blob: dict[str, Any],
        new_title: str | None,
        updated_at: str,
    ) -> None:
        # Keep UC row lean: do not persist full export_tables inside tool_status_json.
        uc_status = {k: v for k, v in status_blob.items() if k != "_export_tables"}
        self._sql(
            f"""
            INSERT INTO {self.catalog}.{self.schema}.ci_chat_messages
            (message_id, session_id, user_email, role, content_markdown, sources_json, tables_json,
             charts_json, files_meta_json, tool_status_json, trace_id, created_at)
            VALUES
            (:m1, :sid, :email, 'user', :ucontent, NULL, NULL, NULL, NULL, NULL, NULL, TIMESTAMP(:t1))
            """,
            params={
                "m1": user_msg["message_id"],
                "sid": session_id,
                "email": user_email.lower(),
                "ucontent": user_msg["content_markdown"],
                "t1": user_msg["created_at"],
            },
        )
        self._sql(
            f"""
            INSERT INTO {self.catalog}.{self.schema}.ci_chat_messages
            (message_id, session_id, user_email, role, content_markdown, sources_json, tables_json,
             charts_json, files_meta_json, tool_status_json, trace_id, created_at)
            VALUES
            (:m2, :sid, :email, 'assistant', :acontent, :sources, :tables, :charts, :files, :status, :trace, TIMESTAMP(:t2))
            """,
            params={
                "m2": assistant_msg["message_id"],
                "sid": session_id,
                "email": user_email.lower(),
                "acontent": assistant_msg["content_markdown"],
                "sources": json.dumps(assistant_msg["sources_json"], ensure_ascii=False),
                "tables": json.dumps(assistant_msg["tables_json"], ensure_ascii=False),
                "charts": json.dumps(assistant_msg["charts_json"], ensure_ascii=False),
                "files": json.dumps(assistant_msg["files_meta_json"], ensure_ascii=False),
                "status": json.dumps(uc_status, ensure_ascii=False),
                "trace": assistant_msg["trace_id"],
                "t2": assistant_msg["created_at"],
            },
        )
        self._sql(
            f"""
            UPDATE {self.catalog}.{self.schema}.ci_chat_sessions
            SET title = :title, updated_at = TIMESTAMP(:updated_at), last_trace_id = :trace
            WHERE session_id = :sid AND user_email = :email
            """,
            params={
                "title": new_title,
                "updated_at": updated_at,
                "trace": assistant_msg["trace_id"],
                "sid": session_id,
                "email": user_email.lower(),
            },
        )

    def _spawn_uc_append_turn(
        self,
        *,
        user_email: str,
        session_id: str,
        user_msg: dict[str, Any],
        assistant_msg: dict[str, Any],
        status_blob: dict[str, Any],
        new_title: str | None,
        updated_at: str,
    ) -> None:
        import threading

        def _run() -> None:
            try:
                self._persist_uc_append_turn(
                    user_email=user_email,
                    session_id=session_id,
                    user_msg=user_msg,
                    assistant_msg=assistant_msg,
                    status_blob=status_blob,
                    new_title=new_title,
                    updated_at=updated_at,
                )
            except Exception as exc:  # noqa: BLE001
                self.demote_reason = f"append_turn_uc_async_failed:{type(exc).__name__}:{exc}"[:300]

        threading.Thread(target=_run, name="ci-uc-append", daemon=True).start()

    # ---- local helpers ----
    def _list_sessions_local(self, user_email: str) -> list[dict[str, Any]]:
        store = self._load_local(user_email)
        sessions = [
            s for s in store["sessions"].values()
            if s.get("user_email") == user_email and s.get("status") != "deleted"
        ]
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    def _create_session_local(self, session: dict[str, Any]) -> dict[str, Any]:
        store = self._load_local(session["user_email"])
        store["sessions"][session["session_id"]] = session
        store["messages"][session["session_id"]] = []
        self._save_local(session["user_email"], store)
        return session

    def _append_turn_local(
        self,
        *,
        user_email: str,
        session: dict[str, Any],
        session_id: str,
        user_msg: dict[str, Any],
        assistant_msg: dict[str, Any],
        new_title: str | None,
        updated_at: str,
    ) -> dict[str, Any]:
        store = self._load_local(user_email)
        # Mirror session if it only existed in UC so local export follow-ups still work.
        local_session = dict(session)
        local_session["title"] = new_title
        local_session["updated_at"] = updated_at
        local_session["last_trace_id"] = assistant_msg["trace_id"]
        store["sessions"][session_id] = local_session
        msgs = store["messages"].setdefault(session_id, [])
        # Avoid duplicating when UC path mirrors after a successful write.
        existing_ids = {m.get("message_id") for m in msgs}
        if user_msg["message_id"] not in existing_ids:
            msgs.append(user_msg)
        if assistant_msg["message_id"] not in existing_ids:
            msgs.append(assistant_msg)
        self._save_local(user_email, store)
        return assistant_msg

    def _local_path(self, email: str) -> Path:
        p = self.history_dir / email.lower()
        p.mkdir(parents=True, exist_ok=True)
        return p / "sessions.json"

    def _load_local(self, email: str) -> dict[str, Any]:
        fp = self._local_path(email)
        if not fp.exists():
            return {"sessions": {}, "messages": {}}
        return json.loads(fp.read_text(encoding="utf-8"))

    def _save_local(self, email: str, store: dict[str, Any]) -> None:
        # default=str: SQL/Genie cells can arrive as Decimal/datetime — a
        # serialization error here used to abort the whole final SSE frame.
        self._local_path(email).write_text(
            json.dumps(store, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # ---- UC SQL helpers ----
    def _headers(self) -> dict[str, str]:
        from dualsource.tools.auth import auth_headers, get_obo_token

        token = self._explicit_token or get_obo_token() or os.getenv("DATABRICKS_TOKEN") or os.getenv("TARGET_TOKEN") or ""
        if token:
            return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        return auth_headers()

    def _sql(self, statement: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        from dualsource.tools.auth import workspace_host

        host = self.host or workspace_host()
        body: dict[str, Any] = {
            "warehouse_id": self.warehouse_id,
            "catalog": self.catalog,
            "schema": self.schema,
            "statement": statement,
            "wait_timeout": "50s",
        }
        if params:
            body["parameters"] = [{"name": k, "value": "" if v is None else str(v)} for k, v in params.items()]
        req = requests.post(
            f"{host}/api/2.0/sql/statements",
            headers=self._headers(),
            json=body,
            timeout=120,
        )
        req.raise_for_status()
        data = req.json()
        state = (data.get("status") or {}).get("state")
        statement_id = data.get("statement_id")
        # poll if needed
        deadline = time.time() + 60
        while state in {"PENDING", "RUNNING"} and time.time() < deadline:
            time.sleep(1.0)
            poll = requests.get(
                f"{host}/api/2.0/sql/statements/{statement_id}",
                headers=self._headers(),
                timeout=60,
            )
            poll.raise_for_status()
            data = poll.json()
            state = (data.get("status") or {}).get("state")
        if state not in {"SUCCEEDED", "SUCCESS"}:
            raise RuntimeError(f"SQL failed: {state} {json.dumps(data)[:800]}")
        return ((data.get("result") or {}).get("data_array")) or []

    def _session_row(self, row: list[Any]) -> dict[str, Any]:
        return {
            "session_id": row[0],
            "user_email": row[1],
            "user_display_name": row[2],
            "title": row[3],
            "status": row[4],
            "created_at": row[5],
            "updated_at": row[6],
            "expires_at": row[7],
            "last_trace_id": row[8],
        }

    def _message_row(self, row: list[Any]) -> dict[str, Any]:
        def loads(v: Any):
            if v is None or v == "":
                return None
            if isinstance(v, (list, dict)):
                return v
            try:
                return json.loads(v)
            except Exception:
                return v

        status = loads(row[9]) or {}
        export_tables = []
        if isinstance(status, dict):
            export_tables = list(status.get("_export_tables") or [])
        files_meta = loads(row[8]) or []
        if isinstance(files_meta, list):
            files_meta = [
                f for f in files_meta
                if str((f or {}).get("type") or "").lower() != "pptx"
            ]
        return {
            "message_id": row[0],
            "session_id": row[1],
            "user_email": row[2],
            "role": row[3],
            "content_markdown": row[4],
            "sources_json": loads(row[5]) or [],
            "tables_json": loads(row[6]) or [],
            "export_tables_json": export_tables,
            "charts_json": loads(row[7]) or [],
            "files_meta_json": files_meta,
            "tool_status_json": status if isinstance(status, dict) else {},
            "trace_id": row[10],
            "created_at": row[11],
        }
