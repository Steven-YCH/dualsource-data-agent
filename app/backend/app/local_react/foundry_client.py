"""Azure AI Foundry / Azure OpenAI chat client (FAST + STRONG)."""

from __future__ import annotations

import json
import os
import re
import time
from time import perf_counter
from typing import Any, Callable, Iterator

import requests

from app.local_react.langsmith_trace import (
    ContextIterator, attach_outputs, configure_langsmith, llm_span, tracing_enabled,
)


class FoundryError(RuntimeError):
    pass


# Process-local sticky FAST backup window. After a failover, later FAST calls
# skip the flaky primary until this timestamp (monotonic seconds).
_FAST_STICKY_UNTIL = 0.0


class FoundryClient:
    """OpenAI-compatible chat completions against AZURE_OPENAI_ENDPOINT."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        fast_model: str | None = None,
        strong_model: str | None = None,
        fast_backup_model: str | None = None,
        timeout: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.endpoint = (
            endpoint
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or ""
        ).strip().rstrip("/")
        self.api_key = (api_key or os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
        self.fast_model = (
            fast_model
            or os.getenv("CI_LLM_MODEL_FAST")
            or "grok-4-1-fast-non-reasoning"
        ).strip()
        self.strong_model = (
            strong_model
            or os.getenv("CI_LLM_MODEL_STRONG")
            or "gpt-4.1"
        ).strip()
        # Optional FAST backup used on transient Foundry 408/429/timeout for primary FAST.
        self.fast_backup_model = (
            fast_backup_model
            if fast_backup_model is not None
            else (os.getenv("CI_LLM_MODEL_FAST_BACKUP") or "gpt-5.4-nano")
        ).strip()
        # After failover, keep using backup for this many seconds (0 disables sticky).
        self.fast_sticky_s = max(0.0, float(os.getenv("CI_FOUNDRY_FAST_STICKY_S") or "3600"))
        # Client wait budget; Foundry may still return HTTP 408 sooner on its side.
        self.timeout = float(
            timeout if timeout is not None else (os.getenv("CI_FOUNDRY_TIMEOUT_S") or "90")
        )
        self.max_retries = max(0, int(os.getenv("CI_FOUNDRY_MAX_RETRIES") or "2"))
        self.reasoning_effort = (
            reasoning_effort if reasoning_effort is not None
            else os.getenv("CI_LLM_REASONING_EFFORT", "none")
        ).strip().lower()
        if self.reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise FoundryError("CI_LLM_REASONING_EFFORT must be none, minimal, low, medium, high, xhigh or max")

    @staticmethod
    def _retryable_status(status_code: int) -> bool:
        return status_code in {408, 429, 500, 502, 503, 504}

    @staticmethod
    def _status_from_error(exc: Exception) -> int | None:
        for token in str(exc).split():
            if token.isdigit():
                return int(token)
        return None

    def _with_retries(self, label: str, fn: Callable[[], Any]) -> Any:
        """Retry transient Foundry gateway errors (408 timeout / 429 rate limit)."""
        attempt = 0
        while True:
            try:
                return fn()
            except FoundryError as exc:
                status = self._status_from_error(exc)
                if status is None or not self._retryable_status(status) or attempt >= self.max_retries:
                    raise
                delay = min(8.0, 0.8 * (2 ** attempt))
                time.sleep(delay)
                attempt += 1

    def _fast_failover_model(self, model: str) -> str | None:
        """Return backup FAST deployment when primary FAST hits a gateway fault."""
        primary = (model or "").strip()
        backup = (self.fast_backup_model or "").strip()
        if not primary or not backup:
            return None
        if primary != self.fast_model:
            return None
        if backup == primary:
            return None
        return backup

    def _should_failover(self, exc: Exception) -> bool:
        status = self._status_from_error(exc)
        if status is not None and self._retryable_status(status):
            return True
        text = str(exc).lower()
        return any(
            token in text
            for token in (
                "connecttimeout",
                "readtimeout",
                "timed out",
                "connection aborted",
                "connection reset",
                "max retries exceeded",
            )
        )

    def _sticky_active(self) -> bool:
        return self.fast_sticky_s > 0 and time.monotonic() < _FAST_STICKY_UNTIL

    def _activate_sticky(self) -> None:
        global _FAST_STICKY_UNTIL
        if self.fast_sticky_s <= 0:
            return
        _FAST_STICKY_UNTIL = time.monotonic() + self.fast_sticky_s

    def _resolve_fast_model(self, model: str) -> tuple[str, bool]:
        """If sticky backup window is active, route primary FAST calls to backup."""
        backup = self._fast_failover_model(model)
        if backup and self._sticky_active():
            return backup, True
        return model, False

    @staticmethod
    def _reasoning_capability(model: str) -> str:
        """Known model IDs/snapshots only; unknown provider/deployment IDs stay untouched.

        Compatibility checked against official OpenAI guidance on 2026-10-14.
        Do not assume that every OpenAI-compatible provider accepts this field.
        """
        name = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model.strip().lower())
        if name in {"gpt-6-luna", "gpt-6-sol", "gpt-5.1", "gpt-5.2", "gpt-5.4",
                    "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "gpt-5.6", "gpt-5.6-sol"}:
            return "supports_none"
        if name in {"gpt-6-astra", "gpt-6.1-sol"}:
            return "requires_reasoning_and_responses_for_tools"
        if name in {"gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro",
                    "o1", "o1-mini", "o3", "o3-mini", "o4-mini"}:
            return "requires_reasoning"
        if name.startswith(("gpt-4", "gpt-3.5", "grok-4-1-fast-non-reasoning")):
            return "not_applicable"
        return "unverified"

    def reasoning_settings(self, model: str) -> dict[str, Any]:
        """Safe diagnostics: requested policy is separate from the sent parameter."""
        capability = self._reasoning_capability(model)
        applicable = capability not in {"not_applicable", "unverified"}
        compatible = not (capability.startswith("requires_reasoning") and self.reasoning_effort == "none")
        return {"requested": self.reasoning_effort, "capability": capability,
                "parameter": self.reasoning_effort if applicable and compatible else None,
                "compatible": compatible if capability != "unverified" else None}

    def _reasoning_body(self, model: str, *, has_tools: bool = False) -> dict[str, str]:
        settings = self.reasoning_settings(model)
        capability = settings["capability"]
        if capability in {"not_applicable", "unverified"}:
            return {}
        if capability.startswith("requires_reasoning") and self.reasoning_effort == "none":
            raise FoundryError(
                f"{model} does not support reasoning_effort=none. "
                "Choose a none-capable model or set a supported CI_LLM_REASONING_EFFORT."
            )
        name = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model.strip().lower())
        if has_tools and (capability == "requires_reasoning_and_responses_for_tools"
                          or name in {"gpt-6-luna", "gpt-6-sol"} and self.reasoning_effort != "none"):
            raise FoundryError(
                f"{model} requires the Responses API for function tools with reasoning enabled. "
                "This client uses Chat Completions; use reasoning_effort=none with Luna/Sol."
            )
        if name == "gpt-5-pro" and self.reasoning_effort != "high":
            raise FoundryError("gpt-5-pro requires reasoning_effort=high")
        return {"reasoning_effort": self.reasoning_effort}

    def ready(self) -> bool:
        return bool(self.endpoint and self.api_key)

    @staticmethod
    def _uses_max_completion_tokens(model: str) -> bool:
        """Newer Foundry GPT-5/GPT-6 deployments reject max_tokens."""
        m = (model or "").strip().lower()
        return (
            m.startswith("gpt-5")
            or m.startswith("gpt-6")
            or m.startswith("o1")
            or m.startswith("o3")
            or m.startswith("o4")
        )

    @staticmethod
    def _force_default_temperature(model: str) -> bool:
        """Some GPT-6 deployments only accept temperature=1 (default)."""
        m = (model or "").strip().lower()
        return m.startswith("gpt-6") or m.startswith("o1") or m.startswith("o3")

    def _token_limit_body(self, model: str, max_tokens: int) -> dict[str, int]:
        if self._uses_max_completion_tokens(model):
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens}

    def _sampling_body(self, model: str, temperature: float) -> dict[str, float]:
        if self._force_default_temperature(model):
            return {}
        if self._reasoning_capability(model).startswith("requires_reasoning"):
            return {}
        if self._reasoning_capability(model) == "supports_none" and self.reasoning_effort != "none":
            return {}
        return {"temperature": temperature}

    def _url(self) -> str:
        if not self.endpoint:
            raise FoundryError("AZURE_OPENAI_ENDPOINT is not set")
        if self.endpoint.endswith("/chat/completions"):
            return self.endpoint
        return f"{self.endpoint}/chat/completions"

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise FoundryError("AZURE_OPENAI_API_KEY is not set")
        return {
            "api-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        configure_langsmith()

        def _chat_model(active_model: str) -> dict[str, Any]:
            body: dict[str, Any] = {
                "model": active_model,
                "messages": messages,
                "stream": False,
            }
            body.update(self._sampling_body(active_model, temperature))
            body.update(self._token_limit_body(active_model, max_tokens))
            body.update(self._reasoning_body(active_model, has_tools=bool(tools)))
            if tools:
                body["tools"] = tools
                body["tool_choice"] = tool_choice or "auto"

            def _once() -> dict[str, Any]:
                try:
                    resp = requests.post(
                        self._url(),
                        headers=self._headers(),
                        json=body,
                        timeout=self.timeout,
                    )
                except requests.RequestException as exc:
                    raise FoundryError(f"Foundry chat transport error: {exc}") from exc
                if resp.status_code >= 400:
                    raise FoundryError(
                        f"Foundry chat failed: {resp.status_code} {resp.text[:800]}"
                    )
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise FoundryError(f"Foundry returned no choices: {str(data)[:400]}")
                msg = choices[0].get("message") or {}
                return {
                    "role": msg.get("role") or "assistant",
                    "content": msg.get("content"),
                    "tool_calls": msg.get("tool_calls") or [],
                    "raw": data,
                    "model_used": active_model,
                    "fast_failover": active_model != model and active_model == self.fast_backup_model,
                }

            return self._with_retries(f"chat:{active_model}", _once)

        def _do_chat() -> dict[str, Any]:
            active, sticky = self._resolve_fast_model(model)
            try:
                out = _chat_model(active)
                if sticky:
                    out["fast_failover"] = True
                    out["fast_sticky"] = True
                    out["model_requested"] = model
                return out
            except FoundryError as exc:
                backup = self._fast_failover_model(model)
                if not backup or active == backup or not self._should_failover(exc):
                    raise
                self._activate_sticky()
                out = _chat_model(backup)
                out["fast_failover"] = True
                out["fast_sticky"] = self.fast_sticky_s > 0
                out["model_requested"] = model
                return out

        if not tracing_enabled():
            return _do_chat()

        with llm_span(
            f"foundry.chat:{model}",
            inputs=self._trace_inputs(model, messages, max_tokens, tools),
            metadata={"provider": "foundry", "kind": "chat", "ls_model_name": model,
                      "fast_backup_model": self.fast_backup_model or None},
        ) as run:
            out = _do_chat()
            raw = out.get("raw") or {}
            attach_outputs(run, {
                "role": out.get("role"),
                "content_chars": len(str(out.get("content") or "")),
                "tool_call_count": len(out.get("tool_calls") or []),
                "tool_names": [str((call.get("function") or {}).get("name") or "")
                               for call in out.get("tool_calls") or []],
                "finish_reason": (raw.get("choices") or [{}])[0].get("finish_reason"),
                "usage_metadata": self._usage_metadata(raw.get("usage")),
                "model_used": out.get("model_used") or model,
                "fast_failover": bool(out.get("fast_failover")),
            })
            return out

    def _trace_inputs(self, model: str, messages: list[dict[str, Any]], max_tokens: int,
                      tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        # Metrics support prompt-size diagnosis without uploading history/SQL rows.
        return {
            "model": model, "message_count": len(messages), "max_tokens": max_tokens,
            "reasoning": self.reasoning_settings(model),
            "message_roles": [str(message.get("role") or "") for message in messages],
            "input_chars": sum(len(json.dumps(message, ensure_ascii=False, default=str))
                               for message in messages),
            "tool_names": [str((tool.get("function") or {}).get("name") or "")
                           for tool in tools or []],
        }

    @staticmethod
    def _usage_metadata(usage: Any) -> dict[str, int]:
        if not isinstance(usage, dict):
            return {}
        names = {"prompt_tokens": "input_tokens", "completion_tokens": "output_tokens",
                 "total_tokens": "total_tokens"}
        result = {target: value for source, target in names.items()
                  if isinstance((value := usage.get(source)), int) and value >= 0}
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if isinstance(reasoning_tokens, int) and reasoning_tokens >= 0:
            result["reasoning_tokens"] = reasoning_tokens
        return result

    def chat_text(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        msg = self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = msg.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts).strip()
        return str(content or "").strip()

    def stream_text(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        configure_langsmith()
        usage: dict[str, int] = {}

        def _open_for(active_model: str):
            body: dict[str, Any] = {
                "model": active_model,
                "messages": messages,
                "stream": True,
            }
            body.update(self._sampling_body(active_model, temperature))
            body.update(self._token_limit_body(active_model, max_tokens))
            body.update(self._reasoning_body(active_model))
            body["stream_options"] = {"include_usage": True}

            def _open_stream():
                try:
                    resp = requests.post(
                        self._url(), headers=self._headers(), json=body,
                        timeout=self.timeout, stream=True,
                    )
                except requests.RequestException as exc:
                    raise FoundryError(f"Foundry stream transport error: {exc}") from exc
                if resp.status_code >= 400:
                    detail = resp.text[:800]
                    closer = getattr(resp, "close", None)
                    if callable(closer):
                        closer()
                    raise FoundryError(
                        f"Foundry stream failed: {resp.status_code} {detail}"
                    )
                return resp

            return self._with_retries(f"stream:{active_model}", _open_stream)

        def _iter() -> Iterator[str]:
            active, _sticky = self._resolve_fast_model(model)
            try:
                resp = _open_for(active)
            except FoundryError as exc:
                backup = self._fast_failover_model(model)
                if not backup or active == backup or not self._should_failover(exc):
                    raise
                self._activate_sticky()
                resp = _open_for(backup)
            try:
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    usage.update(self._usage_metadata(data.get("usage")))
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    content = (choices[0].get("delta") or {}).get("content")
                    if content:
                        yield str(content)
            finally:
                closer = getattr(resp, "close", None)
                if callable(closer):
                    closer()


        if not tracing_enabled():
            return _iter()

        def traced() -> Iterator[str]:
            with llm_span(
                f"foundry.stream:{model}",
                inputs={**self._trace_inputs(model, messages, max_tokens), "stream": True},
                metadata={"provider": "foundry", "kind": "stream", "ls_model_name": model,
                          "reasoning_effort": self.reasoning_settings(model).get("parameter") or "not_sent",
                          "fast_backup_model": self.fast_backup_model or None},
            ) as run:
                chars = 0
                first_token_ms: float | None = None
                started = perf_counter()
                network_stream = _iter()
                try:
                    for chunk in network_stream:
                        if first_token_ms is None:
                            first_token_ms = round((perf_counter() - started) * 1000, 2)
                        chars += len(chunk)
                        yield chunk
                    attach_outputs(run, {"streamed_chars": chars,
                                         "first_token_ms": first_token_ms,
                                         "usage_metadata": usage})
                except GeneratorExit:
                    attach_outputs(run, {"ok": False, "cancelled": True,
                                         "streamed_chars": chars, "usage_metadata": usage})
                    raise
                except Exception:
                    attach_outputs(run, {"ok": False, "streamed_chars": chars})
                    raise
                finally:
                    network_stream.close()

        return ContextIterator(traced())
