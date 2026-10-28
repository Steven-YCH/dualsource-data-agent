"""Offline request timing/correlation tests; no network or App startup."""
from __future__ import annotations

import asyncio
import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch
from uuid import uuid4

from app.request_observability import (
    RequestObservabilityMiddleware, attach_request_timing, observed_call,
)


class FakeTrace:
    instances = []

    def __init__(self, name, *, inputs=None, metadata=None):
        self.request_id = str(uuid4())
        self.run = None
        self.outputs = None
        self.error = None
        self.cancelled = None
        self.instances.append(self)

    def activate(self):
        return nullcontext()

    def finish(self, outputs=None, *, error=None, cancelled=False):
        self.outputs, self.error, self.cancelled = outputs, error, cancelled


class RequestObservationTests(IsolatedAsyncioTestCase):
    async def run_app(self, app, *, mode="local_react", receive=None):
        scope = {"type": "http", "path": "/api/chat/stream", "method": "POST",
                 "headers": [(b"langsmith-trace", b"untrusted-browser-value")]}
        sent = []

        async def default_receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        with patch.dict(os.environ, {"CI_AGENT_MODE": mode,
                                     "LANGSMITH_TRACING": "false",
                                     "LANGCHAIN_TRACING_V2": "false"}), \
                patch("app.request_observability.RequestTrace", FakeTrace):
            await RequestObservabilityMiddleware(app)(scope, receive or default_receive, send)
        return scope, sent

    async def test_root_covers_setup_and_body_with_correlation(self):
        payloads = []

        async def app(scope, receive, send):
            request = SimpleNamespace(scope=scope)
            observed_call(request, "history.read", lambda: None)
            observation = scope["ci.observation"]
            observation.mark_event({"type": "stage"})
            observation.mark_event({"type": "token", "delta": "answer"})
            payloads.append(attach_request_timing(request, {"timing": {"total_ms": 3}}))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"part", "more_body": True})
            self.assertIsNone(observation.trace.outputs, "Root ended before body completion")
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        scope, sent = await self.run_app(app)
        trace = scope["ci.observation"].trace
        timing = payloads[0]["timing"]
        self.assertEqual(timing["total_ms"], 3)
        self.assertEqual(timing["api"]["request_id"], trace.request_id)
        self.assertEqual(timing["api"]["intervals"][0]["name"], "history.read")
        self.assertIsNotNone(timing["api"]["first_answer_delta_ms"])
        self.assertEqual(dict(sent[0]["headers"])[b"x-ci-request-id"].decode(), trace.request_id)
        self.assertTrue(trace.outputs["response_completed"])
        self.assertGreaterEqual(trace.outputs["server_total_ms"], timing["api"]["elapsed_to_snapshot_ms"])

    async def test_stream_error_is_not_reported_success(self):
        async def app(scope, receive, send):
            scope["ci.observation"].mark_event({"type": "error", "detail": "private detail"})
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        scope, _ = await self.run_app(app)
        self.assertEqual(scope["ci.observation"].trace.error, "chat stream failed")

    async def test_cancellation_finishes_root(self):
        async def app(scope, receive, send):
            raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.run_app(app)
        self.assertTrue(FakeTrace.instances[-1].cancelled)

    async def test_other_agent_modes_unchanged(self):
        async def app(scope, receive, send):
            self.assertNotIn("ci.observation", scope)
            await send({"type": "http.response.start", "status": 200, "headers": []})
        _, sent = await self.run_app(app, mode="serving")
        self.assertEqual(sent[0]["headers"], [])

    async def test_failed_stage_is_recorded_and_exception_propagates(self):
        async def app(scope, receive, send):
            request = SimpleNamespace(scope=scope)
            def fail():
                raise ValueError("fixture")
            observed_call(request, "history.read", fail)
        with self.assertRaises(ValueError):
            await self.run_app(app)
        trace = FakeTrace.instances[-1]
        self.assertEqual(trace.error, "ValueError")
        self.assertFalse(trace.outputs["timing"]["intervals"][0]["ok"])
