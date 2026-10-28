"""Request-wide local-agent timing without depending on a tracing service.

Browser receipt timing is measured separately. These elapsed times all use the
server's monotonic clock and include setup before the agent is entered.
"""
from __future__ import annotations

import os
import asyncio
import time
from contextlib import contextmanager
from threading import Lock
from typing import Any, Callable, Iterator
from starlette.requests import ClientDisconnect

from app.local_react.langsmith_trace import RequestTrace, chain_span


class RequestObservation:
    def __init__(self, path: str, method: str):
        self.started = time.perf_counter()
        self.trace = RequestTrace(
            "app.chat.request", inputs={"path": path, "method": method},
            metadata={"component": "app", "agent_mode": "local_react"},
        )
        self._lock = Lock()
        self.intervals: list[dict[str, Any]] = []
        self.error: str | None = None
        self.first_answer_ms: float | None = None
        self.first_progress_ms: float | None = None

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000, 3)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = self.elapsed_ms()
        ok = False
        try:
            with chain_span(f"app.{name}", metadata={"request_id": self.trace.request_id}):
                yield
            ok = True
        finally:
            end = self.elapsed_ms()
            with self._lock:
                self.intervals.append({"name": name, "start_ms": start,
                                       "end_ms": end, "elapsed_ms": round(end - start, 3),
                                       "ok": ok})

    def call(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self.stage(name):
            return fn(*args, **kwargs)

    def mark_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind in {"stage", "tool_start", "tool_end"} and self.first_progress_ms is None:
            self.first_progress_ms = self.elapsed_ms()
        if kind == "token" and event.get("delta") and self.first_answer_ms is None:
            self.first_answer_ms = self.elapsed_ms()
        if kind == "error":
            # Error bodies may contain provider details: don't put them in root metadata.
            self.error = "chat stream failed"

    def snapshot(self) -> dict[str, Any]:
        run = self.trace.run
        with self._lock:
            intervals = [dict(item) for item in self.intervals]
        return {
            "request_id": self.trace.request_id,
            "langsmith_trace_id": str(run.trace_id) if run is not None else None,
            "elapsed_to_snapshot_ms": self.elapsed_ms(),
            "first_progress_ms": self.first_progress_ms,
            "first_answer_delta_ms": self.first_answer_ms,
            "intervals": intervals,
            "clock": "server_monotonic",
            "completion": "before_final_delivery",
        }


class RequestObservabilityMiddleware:
    """Pure ASGI root spans include dependencies and the streaming body.

Only enabled for the local agent. Tracing parent headers from browsers are
deliberately ignored. Finish after ASGI delivery, including disconnect/error.
"""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (scope.get("type") != "http"
                or scope.get("path") not in {"/api/chat", "/api/chat/stream"}
                or os.getenv("CI_AGENT_MODE", "").strip().lower() != "local_react"):
            await self.app(scope, receive, send)
            return
        observation = RequestObservation(scope["path"], scope.get("method", "POST"))
        scope["ci.observation"] = observation
        disconnected = False
        completed = False
        status = 500
        failure: str | None = None
        cancelled = False

        async def receive_observed():
            nonlocal disconnected
            message = await receive()
            if message.get("type") == "http.disconnect":
                disconnected = True
            return message

        async def send_observed(message):
            nonlocal status, completed, disconnected
            if message.get("type") == "http.response.start":
                status = message["status"]
                message = dict(message)
                message["headers"] = [
                    *[(k, v) for k, v in message.get("headers", [])
                      if k.lower() != b"x-ci-request-id"],
                    (b"x-ci-request-id", observation.trace.request_id.encode("ascii")),
                ]
            try:
                await send(message)
            except OSError:
                disconnected = True
                raise
            if message.get("type") == "http.response.body" and not message.get("more_body", False):
                completed = True

        try:
            with observation.trace.activate():
                await self.app(scope, receive_observed, send_observed)
        except BaseException as exc:
            failure = type(exc).__name__
            cancelled = isinstance(exc, (asyncio.CancelledError, ClientDisconnect))
            raise
        finally:
            observation.trace.finish(
                {"status_code": status, "server_total_ms": observation.elapsed_ms(),
                 "response_completed": completed, "timing": observation.snapshot()},
                error=failure or observation.error or (f"HTTP {status}" if status >= 400 else None),
                cancelled=cancelled or (disconnected and not completed),
            )


def observation_for(request: Any) -> RequestObservation | None:
    return request.scope.get("ci.observation") if request is not None else None


def observed_call(request: Any, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    observation = observation_for(request)
    return observation.call(name, fn, *args, **kwargs) if observation else fn(*args, **kwargs)


def attach_request_timing(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
    observation = observation_for(request)
    if observation is None:
        return payload
    result = dict(payload)
    timing = dict(result.get("timing") or {})
    timing["api"] = observation.snapshot()
    result["timing"] = timing
    result["request_id"] = observation.trace.request_id
    result["langsmith_trace_id"] = timing["api"]["langsmith_trace_id"]
    result["debug"] = {**(result.get("debug") or {}), "request": timing["api"]}
    return result
