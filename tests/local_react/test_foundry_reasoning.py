"""Payload regressions for model-aware reasoning policy; no live API calls."""
import json

import pytest

from app.local_react import foundry_client as foundry

TOOL = {"type": "function", "function": {"name": "acknowledge", "description": "Synthetic test only",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}


@pytest.fixture
def requests_sent(monkeypatch):
    sent = []
    monkeypatch.setattr(foundry, "configure_langsmith", lambda: {})
    monkeypatch.setattr(foundry, "tracing_enabled", lambda: False)
    class Response:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "ok", "tool_calls": []}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                              "completion_tokens_details": {"reasoning_tokens": 0}}}
        def iter_lines(self, **kwargs):
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]})
            yield "data: [DONE]"
        def __enter__(self): return self
        def __exit__(self, *args): pass
    def post(url, **kwargs):
        sent.append(kwargs["json"])
        return Response()
    monkeypatch.setattr(foundry.requests, "post", post)
    return sent


def client(effort="none"):
    return foundry.FoundryClient(endpoint="https://test.invalid/openai/v1", api_key="test-only", reasoning_effort=effort)


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-6-sol", "gpt-5.1", "gpt-5.2", "gpt-5.4",
                                   "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "gpt-5.6", "gpt-5.6-sol",
                                   "gpt-5.1-2025-11-13", "GPT-6-Luna"])
def test_none_sent_for_supported_tool_models(requests_sent, model):
    client().chat(model=model, messages=[], tools=[TOOL], tool_choice="required")
    assert requests_sent[0]["reasoning_effort"] == "none"
    assert requests_sent[0]["tools"] == [TOOL]
    assert requests_sent[0]["tool_choice"] == "required"


@pytest.mark.parametrize("model", ["gpt-4.1", "gpt-4o", "grok-4-1-fast-non-reasoning",
                                   "DeepSeek-V4.1-Flash", "custom-foundry-deployment"])
def test_unsupported_or_unverified_providers_do_not_receive_openai_field(requests_sent, model):
    client().chat(model=model, messages=[], tools=[TOOL])
    assert "reasoning_effort" not in requests_sent[0]
    assert client().reasoning_settings(model)["parameter"] is None


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-5.2", "gpt-4.1"])
def test_streaming_uses_same_policy(requests_sent, model):
    assert list(client().stream_text(model=model, messages=[])) == ["ok"]
    assert requests_sent[0].get("reasoning_effort") == (None if model == "gpt-4.1" else "none")
    assert requests_sent[0]["stream"] is True


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "o3", "gpt-6-astra", "gpt-6.1-sol"])
def test_models_requiring_reasoning_fail_before_network(requests_sent, model):
    with pytest.raises(foundry.FoundryError, match="does not support reasoning_effort=none"):
        client().chat(model=model, messages=[], tools=[TOOL])
    assert not requests_sent
    assert client().reasoning_settings(model)["compatible"] is False


def test_supported_non_none_override_omits_sampling_parameters(requests_sent):
    client("low").chat(model="gpt-5.4-mini", messages=[], tools=[TOOL], temperature=0.1)
    assert requests_sent[0]["reasoning_effort"] == "low"
    assert "temperature" not in requests_sent[0]


def test_luna_reasoning_with_tools_requires_responses(requests_sent):
    with pytest.raises(foundry.FoundryError, match="requires the Responses API"):
        client("low").chat(model="gpt-6-luna", messages=[], tools=[TOOL])
    assert not requests_sent


def test_reasoning_without_tools_is_allowed(requests_sent):
    client("low").chat_text(model="gpt-6-luna", messages=[])
    assert requests_sent[0]["reasoning_effort"] == "low"


def test_policy_default_env_and_explicit_override(monkeypatch):
    monkeypatch.delenv("CI_LLM_REASONING_EFFORT", raising=False)
    assert foundry.FoundryClient().reasoning_effort == "none"
    monkeypatch.setenv("CI_LLM_REASONING_EFFORT", "NONE")
    assert foundry.FoundryClient().reasoning_effort == "none"
    assert client("low").reasoning_effort == "low"
    with pytest.raises(foundry.FoundryError, match="CI_LLM_REASONING_EFFORT"):
        client("invalid")


def test_trace_inputs_and_usage_expose_reasoning_setting_without_secrets():
    values = client()._trace_inputs("gpt-6-luna", [{"role": "user", "content": "private"}], 100, [TOOL])
    assert values["reasoning"]["parameter"] == "none"
    assert "private" not in str(values) and "test-only" not in str(values)
    assert client()._usage_metadata({"completion_tokens_details": {"reasoning_tokens": 0}})["reasoning_tokens"] == 0
