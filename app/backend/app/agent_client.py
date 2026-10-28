"""Call DualSource Agent agent: Serving, in-app, or local_react.

Mode is selected by deploy-time env (``CI_AGENT_MODE``), never by reading the
git branch at runtime. Databricks App images usually have no ``.git``, and ADO
deploys committed ``app.yaml`` / App env — that is the channel switch:

- local / ``dev`` tip: ``CI_AGENT_MODE=local_react`` (``.env``)
- ``master`` App target: ``CI_AGENT_MODE=serving`` + ``AI_ENDPOINT_NAME``
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterator

import requests

from dualsource.tools.auth import auth_headers, get_obo_token, workspace_host

log = logging.getLogger("ci.agent_client")

_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: str = "true") -> bool:
    return (os.getenv(name) or default).strip().lower() in _TRUTHY


def _endpoint_name() -> str:
    return (
        os.getenv("AI_ENDPOINT_NAME")
        or os.getenv("CI_AGENT_ENDPOINT_NAME")
        or ""
    ).strip()


def agent_mode() -> str:
    """Resolve orchestrator mode.

    Priority:
    1. Explicit ``CI_AGENT_MODE`` = local_react | serving | in_app
    2. Legacy fallback: ``AI_ENDPOINT_NAME`` present => serving, else in_app

    Do **not** infer from git branch name. Channel intent belongs in deploy
    config (local ``.env`` vs ``master`` ``app.yaml`` / App env UI).
    """
    explicit = (os.getenv("CI_AGENT_MODE") or "").strip().lower()
    if explicit in {"local_react", "serving", "in_app"}:
        return explicit
    return "serving" if _endpoint_name() else "in_app"


# ---------------------------------------------------------------------------
# Serving request plumbing (shared by blocking + streaming paths)
# ---------------------------------------------------------------------------


def _serving_target() -> tuple[str, dict[str, str]]:
    """Serving invocation URL + auth headers (OBO > PAT > ambient)."""
    name = _endpoint_name()
    host = workspace_host()
    token = (
        get_obo_token() or os.getenv("DATABRICKS_TOKEN") or os.getenv("TARGET_TOKEN") or ""
    )
    headers = (
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        if token
        else auth_headers()
    )
    return f"{host}/serving-endpoints/{name}/invocations", headers


def _serving_payload(
    *,
    question: str,
    history: list[dict[str, str]] | None,
    user_email: str,
    session_id: str,
    files_dir: str,
) -> dict[str, Any]:
    """Build the ResponsesAgent invocation body (same wire shape blocking/stream)."""
    messages = list(history or [])
    messages.append({"role": "user", "content": question})
    return {
        "input": [
            {
                "role": m["role"],
                "content": [{"type": "input_text", "text": m["content"]}]
                if m["role"] == "user"
                else [{"type": "output_text", "text": m["content"]}],
            }
            for m in messages
            if m.get("content")
        ],
        "custom_inputs": {
            "user_email": user_email,
            "session_id": session_id,
            "files_dir": files_dir,
        },
    }


# ---------------------------------------------------------------------------
# Serving SSE relay (PROGRESS_EVENTS_CONTRACT v1.1, App side)
#
# The App POSTs the endpoint with `Accept: text/event-stream` and relays
# contract-shaped frames (stage/chart/token/final) to its own SSE clients
# verbatim; ResponsesAgent/OpenAI-compat frames are translated.
#
# Until the endpoint adopts mid-stream contract frames (tracked open item in
# AGENTS.md), its predict_stream stub yields one text-only
# `response.output_item.done` at the end. That still reaches the FE as a
# token delta; the payload (charts/tables/citations) is then fetched with a
# blocking call so nothing is lost. Kill-switch: CI_SERVING_STREAM=0.
# ---------------------------------------------------------------------------


class ServingStreamUnavailable(RuntimeError):
    """Serving endpoint did not offer a usable SSE stream.

    Only raised BEFORE any event is yielded, so the caller can fall back to a
    clean blocking invocation without duplicate frames on the wire.
    """


def _iter_sse_data(resp: requests.Response) -> Iterator[str]:
    """Yield one payload per SSE frame (blank-line-delimited `data:` lines)."""
    data_lines: list[str] = []
    for raw in resp.iter_lines(decode_unicode=True):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        line = raw or ""
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue  # SSE comment / keep-alive
        if line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
        # other SSE fields (event/id/retry) are ignored
    if data_lines:
        yield "\n".join(data_lines)


def _extract_item_text(item: Any) -> str:
    """Best-effort text from a ResponsesAgent output item or content part."""
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return ""
    for key in ("text", "delta"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    content = item.get("content")
    if isinstance(content, list):
        return "".join(
            (part.get("text") or "") if isinstance(part, dict) else ""
            for part in content
        )
    return ""


def _frame_delta_text(frame: dict[str, Any]) -> str | None:
    """ResponsesAgent / OpenAI-compat delta frames -> token text, else None."""
    ftype = frame.get("type")
    if ftype == "response.output_text.delta":
        for key in ("delta", "text", "content"):
            value = frame.get(key)
            if isinstance(value, str) and value:
                return value
        return _extract_item_text(frame.get("item")) or None
    if ftype == "response.output_item.done":
        # Current Serving stub: one completed text item per frame.
        return _extract_item_text(frame.get("item")) or None
    if not ftype and isinstance(frame.get("delta"), str):
        return frame["delta"]  # chat-completions style chunk without `type`
    return None


def _frame_payload(frame: dict[str, Any]) -> dict[str, Any] | None:
    """Detect a payload-bearing final frame (authoritative CI response)."""
    custom = frame.get("custom_outputs")
    if (
        isinstance(custom, dict)
        and isinstance(custom.get("dualsource"), dict)
    ):
        return custom["dualsource"]
    response = frame.get("response")
    if isinstance(response, dict) and isinstance(response.get("answer_markdown"), str):
        return response
    if isinstance(frame.get("answer_markdown"), str):
        return frame
    return None


_CONTRACT_EVENTS = {"stage", "chart", "token", "error", "tool_start", "tool_end"}


def _translate_serving_frame(
    frame: Any, state: dict[str, Any]
) -> list[dict[str, Any]]:
    """Translate one Serving SSE frame into App contract events.

    Contract-shaped frames pass through untouched; foreign frames are mapped
    or ignored. `state` is mutated: final payload / trace / stage bookkeeping.
    """
    if not isinstance(frame, dict):
        return []
    custom = frame.get("custom_outputs")
    if (
        isinstance(custom, dict)
        and custom.get("type") in _CONTRACT_EVENTS
        and "dualsource" not in custom
    ):
        frame = custom
    ftype = frame.get("type")

    # Pass-through: contract v1.1 shapes the FE already renders.
    if ftype in {"stage", "tool_start", "tool_end"} and (
        isinstance(frame.get("stage"), str) or isinstance(frame.get("tool_id"), str)
    ):
        state["stages_yielded"] = True
        return [frame]
    if ftype == "chart" and isinstance(frame.get("chart"), dict):
        return [frame]
    if ftype == "token" and frame.get("delta") is not None:
        return [{"type": "token", "delta": str(frame["delta"])}]
    if ftype == "error":
        return [frame]

    # Foreign delta frames -> token deltas (progressive answer text).
    delta = _frame_delta_text(frame)
    if delta:
        return [{"type": "token", "delta": delta}]

    # Trace may ride any frame (databricks_output on the last one).
    db = frame.get("databricks_output")
    if isinstance(db, dict) and db.get("trace"):
        state["trace_id"] = db.get("trace")

    # Payload final: authoritative structured response. Must run AFTER trace
    # extraction — the payload frame often carries both keys.
    payload = _frame_payload(frame)
    if payload is not None:
        state["final"] = payload
        return []
    return []


def _serving_stream_events(
    *,
    question: str,
    history: list[dict[str, str]] | None,
    user_email: str,
    session_id: str,
    files_dir: str,
) -> Iterator[dict[str, Any]]:
    """Stream one Serving invocation, emitting App contract events.

    Raises ServingStreamUnavailable only before the first event (so the caller
    can fall back cleanly). Mid-stream network errors propagate: main.py turns
    them into an SSE error frame, same as a failed blocking call today.
    """
    url, headers = _serving_target()
    headers = {**headers, "Accept": "text/event-stream"}
    connect_timeout = float(os.getenv("CI_SERVING_STREAM_CONNECT_TIMEOUT") or 15)
    read_timeout = float(os.getenv("CI_SERVING_STREAM_READ_TIMEOUT") or 600)
    body = _serving_payload(
        question=question,
        history=history,
        user_email=user_email,
        session_id=session_id,
        files_dir=files_dir,
    )
    state: dict[str, Any] = {"final": None, "trace_id": None, "stages_yielded": False}

    resp = requests.post(
        url,
        headers=headers,
        json=body,
        stream=True,
        timeout=(connect_timeout, read_timeout),
    )
    try:
        if resp.status_code >= 400:
            raise ServingStreamUnavailable(
                f"HTTP {resp.status_code}: {resp.text[:200]}"
            )
        content_type = (resp.headers.get("content-type") or "").lower()
        if "event-stream" not in content_type:
            # Endpoint answered JSON: streaming not supported on this path.
            raise ServingStreamUnavailable(f"content-type {content_type!r}")
        for data in _iter_sse_data(resp):
            if data.strip() == "[DONE]":
                break
            try:
                frame = json.loads(data)
            except ValueError:
                log.warning("unparseable SSE frame: %.120s", data)
                continue
            for event in _translate_serving_frame(frame, state):
                yield event
            if state["final"] is not None:
                break  # payload final received — stream complete
    finally:
        resp.close()

    payload = state["final"]
    if payload is None:
        # Stream carried no payload (today's text-only predict_stream stub).
        # Fetch the authoritative payload blocking so charts/tables/citations
        # are never lost. Transitional ~2x latency until Serving adopts
        # payload-bearing final frames (PROGRESS_EVENTS_CONTRACT v1.1).
        log.warning(
            "Serving stream ended without payload final — blocking invocation"
        )
        payload = invoke_agent(
            question=question,
            history=history,
            user_email=user_email,
            session_id=session_id,
            files_dir=files_dir,
        )
        if not state["stages_yielded"]:
            # Stub emits no stage frames; replay them for the FE timeline.
            for event in list(payload.get("progress_events") or []):
                if isinstance(event, dict):
                    yield event
    yield {
        "type": "final",
        "response": payload,
        "trace_id": state["trace_id"] or payload.get("trace_id"),
    }


# ---------------------------------------------------------------------------
# Agent invocation entry points
# ---------------------------------------------------------------------------


def invoke_agent(
    *,
    question: str,
    history: list[dict[str, str]] | None,
    user_email: str,
    session_id: str,
    files_dir: str,
) -> dict[str, Any]:
    """Invoke configured agent backend and return dualsource payload."""
    mode = agent_mode()
    if mode == "local_react":
        from app.local_react.orchestrator import LocalReactOrchestrator

        return LocalReactOrchestrator().handle(
            question=question,
            history=history or [],
            user_email=user_email,
            session_id=session_id,
            files_dir=files_dir,
        )

    if mode == "in_app" or not _endpoint_name():
        from dualsource.orchestrator import Orchestrator

        return Orchestrator().handle(
            question=question,
            history=history or [],
            user_email=user_email,
            session_id=session_id,
            files_dir=files_dir,
        ).model_dump()

    return _invoke_serving(
        question=question,
        history=history,
        user_email=user_email,
        session_id=session_id,
        files_dir=files_dir,
    )


def invoke_agent_stream(
    *,
    question: str,
    history: list[dict[str, str]] | None,
    user_email: str,
    session_id: str,
    files_dir: str,
) -> Iterator[dict[str, Any]]:
    """Stream progress/token/chart/final events.

    local_react streams natively. Serving modes relay the endpoint's SSE
    (contract v1.1) when CI_SERVING_STREAM is on (default); any other mode —
    or an endpoint that does not offer SSE — replays the blocking payload's
    progress_events and yields a single final.
    """
    mode = agent_mode()
    if mode == "local_react":
        from app.local_react.orchestrator import LocalReactOrchestrator

        yield from LocalReactOrchestrator().handle_stream(
            question=question,
            history=history or [],
            user_email=user_email,
            session_id=session_id,
            files_dir=files_dir,
        )
        return

    if mode == "serving" and _env_flag("CI_SERVING_STREAM"):
        try:
            yield from _serving_stream_events(
                question=question,
                history=history,
                user_email=user_email,
                session_id=session_id,
                files_dir=files_dir,
            )
            return
        except ServingStreamUnavailable as exc:
            log.warning(
                "Serving SSE unavailable (%s) — falling back to blocking call",
                exc,
            )
    # in_app, streaming disabled, or endpoint without SSE.
    payload = invoke_agent(
        question=question,
        history=history,
        user_email=user_email,
        session_id=session_id,
        files_dir=files_dir,
    )
    for event in list(payload.get("progress_events") or []):
        if isinstance(event, dict):
            yield event
    yield {"type": "final", "response": payload, "trace_id": payload.get("trace_id")}


def _invoke_serving(
    *,
    question: str,
    history: list[dict[str, str]] | None,
    user_email: str,
    session_id: str,
    files_dir: str,
) -> dict[str, Any]:
    url, headers = _serving_target()
    body = _serving_payload(
        question=question,
        history=history,
        user_email=user_email,
        session_id=session_id,
        files_dir=files_dir,
    )
    resp = requests.post(url, headers=headers, json=body, timeout=180)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"agent endpoint {_endpoint_name()} failed: {resp.status_code} {resp.text[:500]}"
        )
    data = resp.json()

    custom = data.get("custom_outputs") or {}
    if isinstance(custom, dict) and custom.get("dualsource"):
        return custom["dualsource"]

    answer = ""
    if isinstance(data.get("output"), list):
        parts = []
        for item in data["output"]:
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("text"):
                        parts.append(str(c["text"]))
        answer = chr(10).join(parts)
    elif data.get("choices"):
        answer = (((data["choices"][0] or {}).get("message") or {}).get("content")) or ""

    return {
        "answer_markdown": answer or str(data)[:2000],
        "sources": [],
        "tables": [],
        "charts": [],
        "files": [],
        "followups": [],
        "status": {"genie": "unknown", "web": "unknown", "files": "skipped"},
        "trace_id": data.get("databricks_output", {}).get("trace") if isinstance(data.get("databricks_output"), dict) else None,
    }