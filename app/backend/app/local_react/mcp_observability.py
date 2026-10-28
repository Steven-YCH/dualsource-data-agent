"""Bounded timing and parent propagation for trusted loopback MCP services.

Stage timings work without LangSmith. Trace headers never include API keys and
are only sent to loopback HTTP endpoints; incoming remote/proxy parents are ignored.
"""
from __future__ import annotations

import ipaddress
import hashlib
import hmac
import json
import secrets
import math
import re
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator
from urllib.parse import urlparse


_PARENT_SIGNING_KEY = secrets.token_bytes(32)


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def trace_headers_for_url(url: str) -> dict[str, str]:
    parsed = urlparse(url)
    if parsed.scheme != "http" or not _loopback(parsed.hostname or ""):
        return {}
    try:
        from app.local_react.langsmith_trace import current_trace_headers
        return current_trace_headers()
    except (ImportError, AttributeError):
        return {}


def bounded_server_observability(value: Any) -> dict[str, Any]:
    """Only accept short phase names/numeric timing; discard arbitrary server data."""
    if not isinstance(value, dict):
        return {}
    component = value.get("component")
    result: dict[str, Any] = {"component": component if component in {"wren_server", "tavily_server"} else "mcp_server", "stages": []}
    items = value.get("stages")
    for item in (items if isinstance(items, list) else [])[:32]:
        if not isinstance(item, dict):
            continue
        name = item.get("stage")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", name):
            continue
        stage = {"stage": name, "ok": item.get("ok") is True}
        for key in ("ms", "start_ms", "end_ms"):
            number = item.get(key)
            if isinstance(number, (int, float)) and math.isfinite(number) and 0 <= number <= 3600000:
                stage[key] = round(number, 2)
        result["stages"].append(stage)
    return result


@contextmanager
def timed_stage(stages: list[dict[str, Any]], name: str, *, origin: float | None = None) -> Iterator[dict[str, Any]]:
    start = time.perf_counter()
    stage: dict[str, Any] = {"stage": name, "ok": True}
    if origin is not None:
        stage["start_ms"] = round((start - origin) * 1000, 2)
    try:
        from app.local_react.langsmith_trace import chain_span
        span = chain_span(name, metadata={"component": "mcp"})
    except (ImportError, AttributeError):
        span = nullcontext()
    try:
        with span as run:
            yield stage
            if stage.get("ok") is False:
                try:
                    from app.local_react.langsmith_trace import attach_outputs
                    attach_outputs(run, {"ok": False, "error": "Operation returned failure"})
                except (ImportError, AttributeError):
                    pass
    except BaseException as exc:
        stage.update(ok=False, error_type=type(exc).__name__)
        raise
    finally:
        end = time.perf_counter()
        stage["ms"] = round((end - start) * 1000, 2)
        if origin is not None:
            stage["end_ms"] = round((end - origin) * 1000, 2)
        stages.append(stage)


def _signed_parent(headers: dict[str, str]) -> dict[str, Any]:
    # HMAC key is server-process-local, never a credential or shared via transport.
    parent = {k: v for k, v in headers.items() if k in {"langsmith-trace", "baggage"} and len(v) <= 8192 and "\r" not in v and "\n" not in v}
    encoded = json.dumps(parent, sort_keys=True).encode()
    return {"headers": parent, "signature": hmac.new(_PARENT_SIGNING_KEY, encoded, hashlib.sha256).hexdigest()}


@contextmanager
def tool_request_trace(ctx: Any) -> Iterator[None]:
    """Recover a middleware-signed parent in FastMCP's session worker context."""
    context = nullcontext()
    try:
        meta = ctx.request_context.meta
        value = meta.model_dump() if hasattr(meta, "model_dump") else meta
        signed = (value or {}).get("ci_parent_trace", {})
        headers = signed.get("headers", {})
        if isinstance(headers, dict) and isinstance(signed.get("signature"), str):
            expected = _signed_parent(headers)
            if expected["headers"] == headers and hmac.compare_digest(expected["signature"], signed["signature"]):
                from app.local_react.langsmith_trace import propagated_trace
                context = propagated_trace(headers)
    except (AttributeError, ImportError, TypeError, ValueError):
        pass
    with context:
        yield


class LoopbackTraceMiddleware:
    """Trust actual loopback peer, ignore forwarding headers, sign request parent.

    MCP session workers may not inherit the ASGI context. Signed request metadata
    lets injected FastMCP Context recover the parent, without trusting client meta.
    """
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        peer = scope.get("client") or ("", 0)
        headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope.get("headers", [])}
        trusted = _loopback(str(peer[0])) and not any(k in headers for k in ("forwarded", "x-forwarded-for", "x-real-ip"))
        parent = _signed_parent(headers)["headers"] if trusted else {}
        context = nullcontext()
        if parent:
            try:
                from app.local_react.langsmith_trace import propagated_trace
                context = propagated_trace(parent)
            except (ImportError, AttributeError):
                pass
        # Only JSON-RPC POST bodies carry tool-call metadata. Preserve other traffic.
        if scope.get("method") == "POST" and receive is not None:
            messages = []
            size = 0
            while True:
                message = await receive()
                messages.append(message)
                size += len(message.get("body", b""))
                if message.get("type") != "http.request" or not message.get("more_body") or size > 1048576:
                    break
            if size <= 1048576 and messages[-1].get("type") == "http.request" and not messages[-1].get("more_body"):
                try:
                    body = json.loads(b"".join(m.get("body", b"") for m in messages))
                    if isinstance(body, dict) and body.get("method") == "tools/call":
                        params = body.setdefault("params", {})
                        meta = params.setdefault("_meta", {})
                        # Ignore any parent metadata supplied by a caller.
                        meta.pop("ci_parent_trace", None)
                        if parent:
                            meta["ci_parent_trace"] = _signed_parent(parent)
                        payload = json.dumps(body).encode()
                        messages = [{"type": "http.request", "body": payload, "more_body": False}]
                        scope = dict(scope, headers=[(k, v) for k, v in scope.get("headers", []) if k.lower() != b"content-length"] + [(b"content-length", str(len(payload)).encode())])
                except (ValueError, TypeError, AttributeError):
                    pass
            original_receive = receive
            async def replay_receive() -> Any:
                return messages.pop(0) if messages else await original_receive()
            receive = replay_receive
        with context:
            await self.app(scope, receive, send)


def install_loopback_tracing(mcp: Any) -> None:
    """Wrap the app factory used by FastMCP.run without starting a server."""
    factory = mcp.streamable_http_app
    def traced_app(*args: Any, **kwargs: Any) -> Any:
        app = factory(*args, **kwargs)
        # Starlette middleware preserves the app's lifespan/session manager.
        app.add_middleware(LoopbackTraceMiddleware)
        return app
    mcp.streamable_http_app = traced_app
