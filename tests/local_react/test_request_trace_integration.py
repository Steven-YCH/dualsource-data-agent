"""Offline real-SDK integration of ASGI root, streamed harness and tool workers."""
from __future__ import annotations

import json
import ast
import sys
from pathlib import Path

import anyio
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "backend"))
pytest.importorskip("langchain")
import langsmith
from langsmith import Client
from langsmith.schemas import LangSmithInfo
from langsmith.run_helpers import get_current_run_tree
from langchain_core.tracers.langchain import wait_for_all_tracers
from starlette.responses import StreamingResponse
from starlette.requests import ClientDisconnect

from app.local_react import langsmith_trace as tracing
from app.local_react.langchain_harness import stream_retrieval
from app.request_observability import RequestObservabilityMiddleware, observed_call
from types import SimpleNamespace


@pytest.fixture
def offline_sdk(monkeypatch):
    creates, updates = [], []
    monkeypatch.setenv("CI_AGENT_MODE", "local_react")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGCHAIN_CALLBACKS_BACKGROUND", "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.setattr(tracing, "tracing_enabled", lambda: True)
    original_init = Client.__init__
    def offline_init(self, *args, **kwargs):
        kwargs["api_url"] = "http://offline.invalid"
        kwargs["auto_batch_tracing"] = False
        kwargs["info"] = LangSmithInfo()
        return original_init(self, *args, **kwargs)
    monkeypatch.setattr(Client, "__init__", offline_init)
    monkeypatch.setattr(Client, "info", property(lambda self: LangSmithInfo()))
    monkeypatch.setattr(Client, "create_run", lambda self, *args, **kwargs: creates.append(dict(kwargs)))
    monkeypatch.setattr(Client, "update_run", lambda self, *args, **kwargs: updates.append({**kwargs, "id": kwargs.get("run_id", args[0] if args else None)}))
    with langsmith.tracing_context(enabled=True):
        yield creates, updates
    wait_for_all_tracers()
    assert get_current_run_tree() is None


class FakeFoundry:
    fast_model = "test-fast-model"
    def __init__(self):
        self.calls = 0
    def chat(self, **kwargs):
        self.calls += 1
        tools = ([{"id": "search-a", "type": "function", "function": {"name": "web_search", "arguments": '{"query":"A"}'}},
                  {"id": "search-b", "type": "function", "function": {"name": "web_search", "arguments": '{"query":"B"}'}}]
                 if self.calls == 1 else [{"id": "done", "type": "function", "function": {"name": "complete_retrieval", "arguments": '{}'}}])
        with tracing.llm_span("foundry.transport"):
            return {"content": "", "tool_calls": tools, "raw": {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}}


class FakeRuntime:
    tables, sources = [], []
    def execute(self, name, arguments):
        with tracing.tool_span("runtime." + name):
            assert get_current_run_tree() is not None
            return {"ok": True, "elapsed_ms": 1, "text": "test evidence"}


async def run_request():
    sent = []
    scope = {"type": "http", "path": "/api/chat/stream", "method": "POST",
             "asgi": {"spec_version": "2.4"}, "headers": [(b"langsmith-trace", b"untrusted-browser-value")],
             "http_version": "1.1"}
    async def app(scope, receive, send):
        observed_call(SimpleNamespace(scope=scope), "history.read", lambda: None)
        retrieval = stream_retrieval(
            foundry=FakeFoundry(), runtime=FakeRuntime(),
            tools=[{"type": "function", "function": {"name": "web_search", "description": "search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}],
            messages=[{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}],
            kpi_ask=False, peer_ask=False, ytd_ask=False, facts={}, max_rounds=3,
            coverage=lambda tables: 0, web_snippets=[],
        )
        def body():
            stream = tracing.wrap_generator("turn", retrieval)
            try:
                for event in stream:
                    yield json.dumps(event)
            finally:
                stream.close()
        await StreamingResponse(body())(scope, receive, send)
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    async def send(message):
        sent.append(message)
    await RequestObservabilityMiddleware(app)(scope, receive, send)
    return scope, sent


def test_real_sdk_langchain_and_worker_spans_share_asgi_root(offline_sdk):
    creates, updates = offline_sdk
    scope, sent = anyio.run(run_request)
    wait_for_all_tracers()
    root = scope["ci.observation"].trace.run
    assert root.parent_run_id is None
    assert root.outputs["response_completed"] is True
    assert sent[-1]["type"] == "http.response.body" and not sent[-1].get("more_body")
    assert len(creates) > 10  # Includes graph, native model/tool and transport spans.
    assert {str(run["trace_id"]) for run in creates} == {str(root.trace_id)}
    assert sum(run.get("name") == "runtime.web_search" for run in creates) == 2
    assert sum(run.get("name") == "foundry.transport" for run in creates) == 2
    assert all(run.get("parent_run_id") is not None for run in creates if str(run["id"]) != str(root.id))
    assert len({str(run["id"]) for run in creates}) == len(creates)


def closing_response_class():
    # Load the actual response class without importing main, dotenv or history.
    source = Path(__file__).resolve().parents[2] / "app/backend/app/main.py"
    definition = next(node for node in ast.parse(source.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == "ClosingStreamingResponse")
    namespace = {"StreamingResponse": StreamingResponse}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["ClosingStreamingResponse"]


def test_disconnect_closes_descendants_before_request_root(offline_sdk):
    creates, updates = offline_sdk
    closed = []
    response_class = closing_response_class()
    scope = {"type": "http", "path": "/api/chat/stream", "method": "POST",
             "asgi": {"spec_version": "2.4"}, "headers": [], "http_version": "1.1"}
    async def app(scope, receive, send):
        def agent():
            with tracing.llm_span("cancelled_model"):
                try:
                    yield "first"
                    yield "second"
                finally:
                    closed.append("model")
        def body():
            stream = tracing.wrap_generator("cancelled_turn", agent())
            try:
                yield from stream
            finally:
                stream.close()
                closed.append("body")
        await response_class(tracing.ContextIterator(body()))(scope, receive, send)
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("synthetic client disconnect")
    async def run():
        with pytest.raises(ClientDisconnect):
            await RequestObservabilityMiddleware(app)(scope, receive, send)
    anyio.run(run)
    assert closed == ["model", "body"]
    root = scope["ci.observation"].trace.run
    root_index = next(i for i, run in enumerate(updates) if str(run["id"]) == str(root.id))
    descendant_indices = [i for i, run in enumerate(updates)
                          if str(run.get("trace_id")) == str(root.trace_id) and str(run["id"]) != str(root.id)]
    assert descendant_indices and all(i < root_index for i in descendant_indices)
    assert root.outputs["response_completed"] is False
    assert root.outputs["cancelled"] is True
    assert root.error
