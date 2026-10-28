"""Offline tests against the pinned real create_agent runtime; no model/MCP calls."""
import json
import os
from pathlib import Path
import sys
import threading

import pytest

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "backend"))
pytest.importorskip("langchain")

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from app.local_react.langchain_harness import stream_retrieval
from app.local_react.langchain_model import FoundryChatModel, wire_messages
from app.local_react.tools import openai_tools


def call(name, **args):
    return {"id": f"{name}-{len(json.dumps(args))}", "type": "function", "function": {
        "name": name, "arguments": json.dumps(args),
    }}


class FakeFoundry:
    fast_model = "grok-4-1-fast-non-reasoning"
    strong_model = "gpt-4.1"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": "", "tool_calls": next(self.responses), "raw": {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }}


class FakeRuntime:
    def __init__(self, sql_results=None):
        self.tables = []
        self.sources = []
        self.calls = []
        self.sql_results = iter(sql_results or [9])
        self.started = None
        self.release = None

    def execute(self, name, args):
        self.calls.append((name, args))
        if self.started:
            self.started.set()
            assert self.release.wait(timeout=5), "test failed to release tool"
        result = {"ok": True, "elapsed_ms": 10.0}
        if name == "wren_run_sql":
            table = {"columns": ["year_month", "passengers"], "rows": [[f"2026-{m:02}", m] for m in range(1, next(self.sql_results) + 1)]}
            self.tables.append(table)
            result["table"] = table
        elif name == "web_search":
            self.sources.append({"type": "public", "url": "https://example.com", "label": "Airport"})
            result.update(text="Published passenger count", sources_n=1)
        return result


def stream(foundry, runtime, **kwargs):
    return stream_retrieval(
        foundry=foundry, runtime=runtime, tools=openai_tools(enable_genie=False),
        messages=[{"role": "system", "content": "Metric rules"}, {"role": "user", "content": "Airport question"}],
        kpi_ask=kwargs.pop("kpi_ask", True), peer_ask=kwargs.pop("peer_ask", False),
        ytd_ask=kwargs.pop("ytd_ask", False),
        facts={"expected_ytd_months": 9, "ytd_window": "Jan-Sep 2026", "ytd_sql_start": "2026-01-01", "ytd_sql_end_exclusive": "2026-10-01"},
        max_rounds=kwargs.pop("max_rounds", 6), coverage=lambda tables: max(len(t["rows"]) for t in tables),
        web_snippets=kwargs.pop("web_snippets", []), **kwargs,
    )


def test_real_agent_keeps_artifacts_and_stops_without_fast_final_prose():
    foundry = FakeFoundry([[call("wren_run_sql", sql="SELECT monthly")], [call("complete_retrieval")]])
    runtime = FakeRuntime()
    events = list(stream(foundry, runtime))
    assert [event["type"] for event in events] == ["tool_start", "tool_end"]
    assert len(runtime.tables[0]["rows"]) == 9
    # SQL is forced first. The model, not a one-row stop, calls complete_retrieval.
    assert len(foundry.calls) == 2
    assert foundry.calls[0]["tool_choice"]["function"]["name"] == "wren_run_sql"
    assert foundry.calls[1]["tool_choice"] == "auto"
    assert all(item["model"] == foundry.fast_model for item in foundry.calls)
    assert "Do not write the user's" in foundry.calls[0]["messages"][0]["content"]


def test_refusal_is_retried_with_forced_sql():
    foundry = FakeFoundry([[], [call("wren_run_sql", sql="SELECT monthly")], [call("complete_retrieval")]])
    list(stream(foundry, FakeRuntime()))
    assert foundry.calls[1]["tool_choice"]["function"]["name"] == "wren_run_sql"


def test_peer_search_forced_after_hkia_and_snippets_retained():
    foundry = FakeFoundry([[call("wren_run_sql", sql="SELECT the hub")], [call("web_search", query="Changi 2026")], [call("complete_retrieval")]])
    snippets = []
    runtime = FakeRuntime()
    events = list(stream(foundry, runtime, peer_ask=True, web_snippets=snippets))
    assert foundry.calls[1]["tool_choice"] == "auto"
    assert "one web_search for each named airport" in foundry.calls[1]["messages"][0]["content"]
    assert snippets == ["Published passenger count"]
    assert events[-1]["sources_n"] == 1


def test_peer_is_not_stopped_after_the_first_web_result():
    foundry = FakeFoundry([
        [call("wren_run_sql", sql="SELECT the hub")],
        [call("web_search", query="ICN 2026")],
        [call("web_search", query="SIN 2026"), call("web_search", query="TPE 2026")],
        [call("complete_retrieval")],
    ])
    runtime = FakeRuntime()
    list(stream(foundry, runtime, peer_ask=True))
    assert [name for name, _ in runtime.calls if name == "web_search"] == ["web_search", "web_search", "web_search"]
    assert len(foundry.calls) == 4


def test_ytd_incomplete_rows_force_full_window_repair():
    foundry = FakeFoundry([[call("wren_run_sql", sql="SELECT Q1")], [call("wren_run_sql", sql="SELECT completed_months")], [call("complete_retrieval")]])
    list(stream(foundry, FakeRuntime(sql_results=[3, 9]), ytd_ask=True))
    assert foundry.calls[1]["tool_choice"]["function"]["name"] == "wren_run_sql"
    assert "2026-10-01" in foundry.calls[1]["messages"][-1]["content"]


def test_model_round_budget_stops_without_another_model_call():
    foundry = FakeFoundry([[call("wren_run_sql", sql="SELECT monthly")]])
    list(stream(foundry, FakeRuntime(), max_rounds=1))
    assert len(foundry.calls) == 1


def test_spare_exploration_skipped_without_calling_runtime():
    foundry = FakeFoundry([[call("wren_get_instructions"), call("wren_describe_schema")], [call("wren_run_sql", sql="SELECT monthly")], [call("complete_retrieval")]])
    runtime = FakeRuntime()
    events = list(stream(foundry, runtime))
    explored = [name for name, _ in runtime.calls if name.startswith("wren_") and name != "wren_run_sql"]
    assert len(explored) == 1
    assert len(events) == 6


def test_tool_start_streams_while_tool_is_still_running():
    foundry = FakeFoundry([[call("wren_run_sql", sql="SELECT monthly")], [call("complete_retrieval")]])
    runtime = FakeRuntime()
    runtime.started, runtime.release = threading.Event(), threading.Event()
    iterator = stream(foundry, runtime)
    try:
        first = next(iterator)
        assert first["type"] == "tool_start"
        assert not runtime.release.is_set()
    finally:
        runtime.release.set()
    assert list(iterator)[0]["type"] == "tool_end"


def test_native_model_and_tool_callbacks_are_children_of_graph():
    class Recorder(BaseCallbackHandler):
        def __init__(self):
            self.models, self.tools, self.chains = [], [], []
        def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
            self.models.append((run_id, parent_run_id))
        def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
            self.tools.append((run_id, parent_run_id))
        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
            self.chains.append((run_id, parent_run_id))
    recorder = Recorder()
    foundry = FakeFoundry([[call("wren_run_sql", sql="SELECT monthly")], [call("complete_retrieval")]])
    list(stream(foundry, FakeRuntime(), callbacks=[recorder]))
    assert len(recorder.models) == 2
    assert len(recorder.tools) == 2
    assert all(parent for _, parent in recorder.models + recorder.tools)
    known = {run_id for run_id, _ in recorder.chains}
    assert all(parent in known for _, parent in recorder.models + recorder.tools)


def test_adapter_preserves_tool_ids_and_token_usage():
    messages = [HumanMessage(content="Question"), AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "2026"}, "id": "abc"}]), ToolMessage(content="Result", tool_call_id="abc")]
    wire = wire_messages(messages)
    assert wire[1]["tool_calls"][0]["id"] == "abc"
    assert wire[2]["tool_call_id"] == "abc"
    foundry = FakeFoundry([[]])
    message = FoundryChatModel(client=foundry, model_name=foundry.fast_model).invoke([HumanMessage(content="Hello")])
    assert message.usage_metadata == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


@pytest.mark.parametrize("harness", ["legacy", "langchain"])
def test_orchestrator_preserves_response_and_uses_one_strong_synthesis(monkeypatch, harness):
    from app.local_react import orchestrator
    class Runtime(FakeRuntime):
        def __init__(self):
            super().__init__()
            self.charts, self.tool_calls = [], []
            self.timings_ms, self.tool_wall_ms = {}, 10.0
            self.status, self.last_sql = {"genie": "ok", "web": "skipped", "files": "skipped"}, None
        def enable_genie(self):
            return False
    class Foundry(FakeFoundry):
        strong_calls = 0
        closed = False
        def ready(self):
            return True
        def stream_text(self, **kwargs):
            self.strong_calls += 1
            assert kwargs["model"] == self.strong_model
            try:
                yield "Verified passenger results [Internal]."
            finally:
                self.closed = True
    monkeypatch.setenv("CI_LOCAL_REACT_HARNESS", harness)
    monkeypatch.setattr(orchestrator, "ToolRuntime", Runtime)
    responses = [[call("wren_run_sql", sql="SELECT monthly")], [call("complete_retrieval")] if harness == "langchain" else []]
    foundry = Foundry(responses)
    events = list(orchestrator.LocalReactOrchestrator(foundry).handle_stream(question="the hub passenger traffic"))
    payload = events[-1]["response"]
    assert payload["answer_markdown"] == "Verified passenger results [Internal]."
    assert payload["tables"] and payload["charts"] == []
    assert payload["timing"]["harness"] == payload["debug"]["harness"] == harness
    assert payload["timing"]["retrieval"]["model_calls"] == 2
    assert payload["timing"]["tool_wall_ms"] == 10.0
    assert foundry.strong_calls == 1 and foundry.closed
    stages = [(ev.get("stage"), ev.get("status")) for ev in events if ev["type"] == "stage"]
    assert ("genie", "start") in stages and ("genie", "end") in stages


def test_chart_followup_reuses_prior_tables_without_retrieval(monkeypatch):
    from app.local_react import orchestrator
    class Runtime(FakeRuntime):
        def __init__(self):
            super().__init__()
            self.charts, self.tool_calls, self.timings_ms = [], [], {}
            self.status, self.last_sql = {"genie": "skipped", "web": "skipped", "files": "skipped"}, None
        def enable_genie(self):
            return False
    class Foundry(FakeFoundry):
        def ready(self):
            return True
        def stream_text(self, **kwargs):
            yield "Passenger trend [Internal]."
    monkeypatch.setenv("CI_LOCAL_REACT_HARNESS", "langchain")
    monkeypatch.setattr(orchestrator, "ToolRuntime", Runtime)
    foundry = Foundry([])
    table = {"columns": ["year_month", "passengers"], "rows": [["2026-01", 10], ["2026-02", 12]]}
    payload = orchestrator.LocalReactOrchestrator(foundry).handle(question="Show this as a line chart", history=[
        {"role": "user", "content": "the hub passengers"},
        {"role": "assistant", "content": "Trend", "tables": [table], "sources": [{"label": "Internal"}]},
    ])
    assert foundry.calls == []
    assert payload["tables"] == [table]
    assert payload["charts"][0]["echarts"]
    assert "[[CHART:0]]" in payload["answer_markdown"]


def test_cancelled_synthesis_stream_is_closed(monkeypatch):
    from app.local_react import orchestrator
    class Runtime(FakeRuntime):
        def __init__(self):
            super().__init__()
            self.charts, self.tool_calls, self.timings_ms = [], [], {}
            self.status, self.last_sql = {}, None
        def enable_genie(self):
            return False
    class Foundry(FakeFoundry):
        closed = False
        def ready(self):
            return True
        def stream_text(self, **kwargs):
            try:
                yield "First token"
                yield "Second token"
            finally:
                self.closed = True
    monkeypatch.setenv("CI_LOCAL_REACT_HARNESS", "langchain")
    monkeypatch.setattr(orchestrator, "ToolRuntime", Runtime)
    foundry = Foundry([[]])
    iterator = orchestrator.LocalReactOrchestrator(foundry).handle_stream(question="Explain this project")
    for event in iterator:
        if event["type"] == "token":
            iterator.close()
            break
    assert foundry.closed


def test_tool_error_is_streamed_and_observed_without_hidden_retry():
    class BrokenRuntime(FakeRuntime):
        def execute(self, name, args):
            self.calls.append((name, args))
            raise RuntimeError("test transport failed")
    runtime = BrokenRuntime()
    foundry = FakeFoundry([[call("wren_run_sql", sql="SELECT monthly")]])
    events = list(stream(foundry, runtime, max_rounds=1))
    assert events[-1]["ok"] is False
    assert "test transport failed" in events[-1]["detail"]
    assert len(runtime.calls) == 1
