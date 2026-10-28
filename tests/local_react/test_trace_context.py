"""Offline real-SDK regressions: no env secrets, servers or trace uploads."""
from __future__ import annotations

import json
import sys
from contextvars import Context
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "backend"))
langsmith = pytest.importorskip("langsmith")
from langsmith.run_helpers import get_current_run_tree
from langsmith.run_trees import RunTree
from app.local_react import langsmith_trace as tracing


@pytest.fixture
def recorded(monkeypatch):
    runs, completed = [], []
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.setattr(tracing, "tracing_enabled", lambda: True)
    monkeypatch.setattr(RunTree, "post", lambda self, *a, **kw: runs.append(self))
    def offline_patch(self, *args, **kwargs):
        # Real RunTree.patch finalizes lazily before sending its update.
        if self.end_time is None:
            self.end()
        completed.append(self)
    monkeypatch.setattr(RunTree, "patch", offline_patch)
    # SDK bookkeeping is enabled while network post/patch is intercepted above.
    with langsmith.tracing_context(enabled=True):
        yield runs, completed
    assert get_current_run_tree() is None


def fresh_next(iterator):
    return Context().run(next, iterator)


def stream_children():
    with tracing.tool_span("before_yield"):
        yield {"type": "progress"}
    with tracing.ContextThreadPoolExecutor(max_workers=2) as pool:
        def worker():
            with tracing.tool_span("worker") as run:
                return run.id
        assert pool.submit(worker).result()
    with tracing.llm_span("after_yield"):
        yield {"type": "final", "response": {"answer_markdown": "answer"}}


def test_fresh_context_stream_and_workers_keep_parent(recorded):
    runs, completed = recorded
    stream = tracing.wrap_generator("turn", stream_children())
    assert fresh_next(stream)["type"] == "progress"
    assert get_current_run_tree() is None
    assert fresh_next(stream)["type"] == "final"
    with pytest.raises(StopIteration):
        fresh_next(stream)
    root = next(run for run in runs if run.name == "turn")
    assert all(run.trace_id == root.trace_id for run in runs)
    # worker is a direct turn child; before_yield closes on the next iteration.
    assert next(run for run in runs if run.name == "worker").parent_run_id == root.id
    assert all(run.end_time is not None for run in completed)
    assert root.outputs["ok"] is True
    assert next(run for run in runs if run.name == "after_yield").run_type == "llm"


def test_two_interleaved_request_streams_are_isolated(recorded):
    runs, completed = recorded
    requests = [tracing.RequestTrace(f"request_{i}") for i in range(2)]
    streams = []
    for request in requests:
        with request.activate():
            with tracing.chain_span("history"):
                pass
            streams.append(tracing.wrap_generator("turn", stream_children()))
        assert get_current_run_tree() is None
    for stream in streams:
        assert fresh_next(stream)["type"] == "progress"
    for stream, request in zip(streams, requests):
        assert fresh_next(stream)["type"] == "final"
        with pytest.raises(StopIteration):
            fresh_next(stream)
        request.finish({"done": True})
        request.finish({"done": False})  # completion is idempotent
    roots = [request.run for request in requests]
    assert roots[0].trace_id != roots[1].trace_id
    assert all(sum(run.trace_id == root.trace_id for run in runs) == 6 for root in roots)
    assert all(root.outputs["done"] is True for root in roots)
    assert all(sum(run.id == root.id for run in completed) == 1 for root in roots)


def test_stream_close_cleans_child_and_root(recorded):
    runs, completed = recorded
    stream = tracing.wrap_generator("cancel_turn", stream_children())
    fresh_next(stream)
    Context().run(stream.close)
    root = next(run for run in runs if run.name == "cancel_turn")
    assert root.outputs["cancelled"] is True
    assert root.error
    assert all(run.end_time is not None for run in runs)
    assert all(run in completed for run in runs)


def test_stream_exception_ends_and_does_not_leak(recorded):
    runs, completed = recorded
    def failing():
        yield "first"
        raise ValueError("synthetic failure")
    stream = tracing.wrap_generator("failed_turn", failing())
    assert fresh_next(stream) == "first"
    with pytest.raises(ValueError, match="synthetic failure"):
        fresh_next(stream)
    root = next(run for run in runs if run.name == "failed_turn")
    assert root.error and root.outputs["ok"] is False
    assert root in completed


def test_request_cancel_finish_is_idempotent(recorded):
    _, completed = recorded
    request = tracing.RequestTrace("cancel_request")
    request.finish(cancelled=True)
    request.finish()
    assert request.run.outputs["cancelled"] is True
    assert request.run.error == "request cancelled"
    assert completed.count(request.run) == 1


def test_stream_and_thread_helper_disabled_are_noop(monkeypatch):
    monkeypatch.setattr(tracing, "tracing_enabled", lambda: False)
    gen = iter([1, 2])
    assert tracing.wrap_generator("off", gen) is gen
    request = tracing.RequestTrace("off")
    assert request.request_id and request.run is None
    with request.activate():
        with tracing.ContextThreadPoolExecutor() as pool:
            assert pool.submit(lambda: 3).result() == 3
    assert request.headers() == {}
    request.finish()


def test_internal_headers_continue_parent_and_reject_injection(recorded):
    runs, _ = recorded
    request = tracing.RequestTrace("client")
    headers = request.headers()
    assert "langsmith-trace" in headers
    with tracing.propagated_trace({**headers, "Authorization": "secret", "bad": "x"}):
        with tracing.chain_span("internal_server") as run:
            assert run.trace_id == request.run.trace_id
    assert tracing._allowed_headers({"baggage": "bad\nheader", "langsmith-trace": "x"}) == {"langsmith-trace": "x"}
    request.finish()


def test_foundry_llm_type_usage_and_stream_first_token(recorded, monkeypatch):
    from app.local_react import foundry_client as foundry
    runs, _ = recorded
    monkeypatch.setattr(foundry, "configure_langsmith", lambda: {})
    monkeypatch.setattr(foundry, "tracing_enabled", lambda: True)
    class Response:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
        def iter_lines(self, **kw):
            yield 'data: ' + json.dumps({"choices": [{"delta": {"content": "ok"}}]})
            yield 'data: ' + json.dumps({"choices": [], "usage": self.json()["usage"]})
            yield 'data: [DONE]'
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(foundry.requests, "post", lambda *a, **kw: Response())
    client = foundry.FoundryClient(endpoint="https://test.invalid", api_key="test-do-not-record")
    with tracing.chain_span("request"):
        assert client.chat(model="test", messages=[{"role": "user", "content": "private"}])["content"] == "ok"
        stream = client.stream_text(model="test", messages=[])
    assert fresh_next(stream) == "ok"
    with pytest.raises(StopIteration):
        fresh_next(stream)
    llms = [run for run in runs if run.name.startswith("foundry.")]
    assert len(llms) == 2 and all(run.run_type == "llm" for run in llms)
    assert all(run.outputs["usage_metadata"]["total_tokens"] == 5 for run in llms)
    assert llms[-1].outputs["first_token_ms"] >= 0
    assert "private" not in str(llms[0].inputs)
    assert "test-do-not-record" not in str([(run.inputs, run.outputs, run.extra) for run in runs])


def test_request_root_ignores_ambient_parent(recorded):
    with tracing.chain_span("untrusted_ambient") as ambient:
        request = tracing.RequestTrace("independent_request")
        assert request.run.parent_run_id is None
        assert request.run.trace_id != ambient.trace_id
        with request.activate():
            assert get_current_run_tree().id == request.run.id
        assert get_current_run_tree().id == ambient.id
        request.finish()


def test_trace_upload_setup_failure_does_not_break_stream(recorded, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("synthetic SDK setup failure")
    monkeypatch.setattr(RunTree, "post", unavailable)
    request = tracing.RequestTrace("request")
    assert request.run is None
    with request.activate():
        stream = tracing.wrap_generator("turn", iter([1, 2]))
        assert fresh_next(stream) == 1
        assert fresh_next(stream) == 2
        with pytest.raises(StopIteration):
            fresh_next(stream)
    request.finish()


def test_output_and_upload_finalization_failures_do_not_break_response(recorded, monkeypatch):
    request = tracing.RequestTrace("request")
    def unavailable(*args, **kwargs):
        raise RuntimeError("synthetic upload failure")
    monkeypatch.setattr(RunTree, "patch", unavailable)
    monkeypatch.setattr(RunTree, "add_outputs", unavailable)
    with request.activate():
        with tracing.llm_span("model") as run:
            tracing.attach_outputs(run, {"ok": True})
    request.finish({"done": True})
    assert request.run.outputs["done"]


def test_throw_resumes_in_preserved_context(recorded):
    def generator():
        try:
            yield "first"
        except ValueError:
            with tracing.tool_span("thrown_child"):
                yield "recovered"
    stream = tracing.wrap_generator("turn", generator())
    assert fresh_next(stream) == "first"
    assert Context().run(stream.throw, ValueError("synthetic")) == "recovered"
    with pytest.raises(StopIteration):
        fresh_next(stream)
    runs, _ = recorded
    assert len({run.trace_id for run in runs}) == 1


def test_returned_error_payload_marks_span_error_without_provider_details(recorded):
    with tracing.tool_span("failed_tool") as run:
        tracing.attach_outputs(run, {"ok": False, "error": "synthetic provider detail"})
    assert run.error == "operation returned an error"
    assert run.outputs["ok"] is False


def test_existing_span_error_is_not_overwritten_by_returned_failure(recorded):
    with tracing.tool_span("failed_tool") as run:
        run.error = "request cancelled"
        tracing.attach_outputs(run, {"ok": False})
    assert run.error == "request cancelled"
