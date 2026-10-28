"""FAST primary -> backup failover on Foundry 408/429."""
import json

import pytest

from app.local_react import foundry_client as foundry


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(foundry.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(foundry, "configure_langsmith", lambda: {})
    monkeypatch.setattr(foundry, "tracing_enabled", lambda: False)


def _client(**kwargs):
    return foundry.FoundryClient(
        endpoint="https://test.invalid/openai/v1",
        api_key="test-only",
        fast_model="grok-4-1-fast-non-reasoning",
        fast_backup_model="gpt-5.4-nano",
        strong_model="gpt-4.1",
        **kwargs,
    )


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {
            "choices": [{"message": {"role": "assistant", "content": "ok", "tool_calls": []}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload

    def iter_lines(self, **kwargs):
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]})
        yield "data: [DONE]"


def test_chat_failsover_from_fast_primary_on_429(monkeypatch):
    calls = []

    def post(url, **kwargs):
        body = kwargs["json"]
        calls.append(body["model"])
        if body["model"] == "grok-4-1-fast-non-reasoning":
            return _Resp(429, text='{"error":{"code":"Too Many Requests"}}')
        return _Resp(200)

    monkeypatch.setattr(foundry.requests, "post", post)
    out = _client(timeout=5).chat(
        model="grok-4-1-fast-non-reasoning",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert out["fast_failover"] is True
    assert out["model_used"] == "gpt-5.4-nano"
    assert "grok-4-1-fast-non-reasoning" in calls
    assert calls[-1] == "gpt-5.4-nano"


def test_chat_does_not_failover_strong_model(monkeypatch):
    def post(url, **kwargs):
        return _Resp(429, text='{"error":{"code":"Too Many Requests"}}')

    monkeypatch.setattr(foundry.requests, "post", post)
    client = _client(timeout=5)
    client.max_retries = 0
    with pytest.raises(foundry.FoundryError, match="429"):
        client.chat(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_stream_failsover_from_fast_primary_on_408(monkeypatch):
    calls = []

    def post(url, **kwargs):
        body = kwargs["json"]
        calls.append(body["model"])
        if body["model"] == "grok-4-1-fast-non-reasoning":
            return _Resp(408, text='{"error":{"code":"Timeout"}}')
        return _Resp(200)

    monkeypatch.setattr(foundry.requests, "post", post)
    text = "".join(_client(timeout=5).stream_text(
        model="grok-4-1-fast-non-reasoning",
        messages=[{"role": "user", "content": "hi"}],
    ))
    assert text == "ok"
    assert calls[-1] == "gpt-5.4-nano"


def test_sticky_backup_skips_primary_for_window(monkeypatch):
    foundry._FAST_STICKY_UNTIL = 0.0
    calls = []

    def post(url, **kwargs):
        body = kwargs["json"]
        calls.append(body["model"])
        if body["model"] == "grok-4-1-fast-non-reasoning":
            return _Resp(429, text='{"error":{"code":"Too Many Requests"}}')
        return _Resp(200)

    monkeypatch.setattr(foundry.requests, "post", post)
    client = _client(timeout=5)
    client.fast_sticky_s = 3600
    out1 = client.chat(model="grok-4-1-fast-non-reasoning", messages=[{"role": "user", "content": "hi"}])
    assert out1["fast_failover"] is True
    assert out1["model_used"] == "gpt-5.4-nano"
    # Second call should go straight to backup while sticky window is active.
    out2 = client.chat(model="grok-4-1-fast-non-reasoning", messages=[{"role": "user", "content": "hi again"}])
    assert out2["model_used"] == "gpt-5.4-nano"
    assert out2.get("fast_sticky") is True
    # Primary may be retried before first failover; sticky means no MORE primary after that.
    assert calls.count("gpt-5.4-nano") >= 2
    assert calls[-1] == "gpt-5.4-nano"
    # Second chat should not add another primary attempt.
    primary_before_second = 0
    # Reconstruct: last call is sticky backup-only.
    assert "grok-4-1-fast-non-reasoning" not in calls[calls.index("gpt-5.4-nano")+1:]


def test_connect_timeout_triggers_failover(monkeypatch):
    foundry._FAST_STICKY_UNTIL = 0.0
    calls = []

    def post(url, **kwargs):
        body = kwargs["json"]
        calls.append(body["model"])
        if body["model"] == "grok-4-1-fast-non-reasoning":
            raise foundry.requests.exceptions.ConnectTimeout("connect timed out")
        return _Resp(200)

    monkeypatch.setattr(foundry.requests, "post", post)
    client = _client(timeout=5)
    client.max_retries = 0
    client.fast_sticky_s = 60
    out = client.chat(model="grok-4-1-fast-non-reasoning", messages=[{"role": "user", "content": "hi"}])
    assert out["fast_failover"] is True
    assert out["model_used"] == "gpt-5.4-nano"
    assert calls[-1] == "gpt-5.4-nano"
