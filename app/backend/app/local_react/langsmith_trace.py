"""Optional LangSmith tracing for local_react (Foundry + tools).

Enabled when LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 is truthy and an API key
is present. Soft-deps: if `langsmith` is not installed, all helpers no-op.
"""

from __future__ import annotations

import os
import logging
import sys
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from threading import Lock
from uuid import uuid4
from typing import Any, Callable, Iterator, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_LOGGER = logging.getLogger(__name__)

_CONFIGURED = False
_ENABLED: bool | None = None


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def tracing_enabled() -> bool:
    """Return True when LangSmith tracing should be active."""
    global _ENABLED
    if _ENABLED is not None:
        return _ENABLED
    flag = _truthy(os.getenv("LANGSMITH_TRACING")) or _truthy(
        os.getenv("LANGCHAIN_TRACING_V2")
    )
    key = (
        os.getenv("LANGSMITH_API_KEY")
        or os.getenv("LANGCHAIN_API_KEY")
        or ""
    ).strip()
    try:
        import langsmith  # noqa: F401
    except Exception:
        _ENABLED = False
        return False
    _ENABLED = bool(flag and key)
    return _ENABLED


def configure_langsmith() -> dict[str, Any]:
    """Normalize env vars and report status. Safe to call repeatedly."""
    global _CONFIGURED, _ENABLED
    project = (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "dualsource-agent-dev"
    ).strip()
    endpoint = (
        os.getenv("LANGSMITH_ENDPOINT")
        or os.getenv("LANGCHAIN_ENDPOINT")
        or ""
    ).strip()
    api_key = (
        os.getenv("LANGSMITH_API_KEY")
        or os.getenv("LANGCHAIN_API_KEY")
        or ""
    ).strip()

    # Keep modern + legacy names in sync so either SDK path works.
    if api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", api_key)
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    if project:
        os.environ.setdefault("LANGSMITH_PROJECT", project)
        os.environ.setdefault("LANGCHAIN_PROJECT", project)
    if endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", endpoint)
        os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)

    enabled = tracing_enabled()
    if enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    _CONFIGURED = True
    return {
        "enabled": enabled,
        "project": project,
        "endpoint": endpoint or "https://api.smith.langchain.com",
        "has_api_key": bool(api_key),
        "configured": _CONFIGURED,
    }


def reset_tracing_cache() -> None:
    """Test helper — clear cached enablement."""
    global _ENABLED, _CONFIGURED
    _ENABLED = None
    _CONFIGURED = False


def traceable(
    *,
    name: str,
    run_type: str = "chain",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorator that becomes a no-op when LangSmith is off / missing."""

    def decorator(fn: F) -> F:
        if not tracing_enabled():
            return fn
        try:
            from langsmith import traceable as ls_traceable
        except Exception:
            return fn

        wrapped = ls_traceable(
            name=name,
            run_type=run_type,
            tags=tags or ["dualsource", "local_react"],
            metadata=metadata or {},
        )(fn)
        return wrapped  # type: ignore[return-value]

    return decorator


@contextmanager
def span(
    name: str,
    *,
    run_type: str = "chain",
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    parent: Any = None,
) -> Iterator[Any]:
    """Use SDK context propagation or an explicit trusted parent."""
    if not tracing_enabled():
        yield None
        return
    try:
        from langsmith import trace as ls_trace
    except ImportError:
        yield None
        return
    options = {"parent": parent} if parent is not None else {}
    try:
        scope = ls_trace(
            name=name, run_type=run_type, inputs=inputs or {},
            metadata=metadata or {}, tags=["dualsource", "local_react", run_type],
            **options,
        )
        run = scope.__enter__()
    except Exception as exc:
        _LOGGER.warning("Tracing setup unavailable (%s)", type(exc).__name__)
        yield None
        return
    try:
        yield run
    except BaseException:
        try:
            scope.__exit__(*sys.exc_info())
        except Exception as exc:
            _LOGGER.warning("Tracing finalization unavailable (%s)", type(exc).__name__)
        raise
    else:
        try:
            scope.__exit__(None, None, None)
        except Exception as exc:
            _LOGGER.warning("Tracing finalization unavailable (%s)", type(exc).__name__)


def tool_span(name: str, **kwargs: Any) -> Any:
    return span(name, run_type="tool", **kwargs)


def chain_span(name: str, **kwargs: Any) -> Any:
    return span(name, run_type="chain", **kwargs)


def llm_span(name: str, **kwargs: Any) -> Any:
    # Type must be set when the run is created/posted, not mutated afterwards.
    return span(name, run_type="llm", **kwargs)


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """Capture an independent context for each submitted worker, SDK optional."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        context = copy_context()
        return super().submit(context.run, fn, *args, **kwargs)


class ContextIterator(Iterator[Any]):
    """Resume/close a sync stream in its own persistent context.

    ASGI adapters can call next in a fresh context on each iteration. Keeping
    the generator's context here preserves parent spans and avoids leaking a
    suspended span into the caller. This does not buffer streamed content.
    """

    def __init__(self, iterator: Iterator[Any], context: Context | None = None):
        self._iterator = iter(iterator)
        self._context = context if context is not None else copy_context()
        self._lock = Lock()

    def __iter__(self) -> "ContextIterator":
        return self

    def __next__(self) -> Any:
        with self._lock:
            return self._context.run(next, self._iterator)

    def send(self, value: Any) -> Any:
        with self._lock:
            return self._context.run(self._iterator.send, value)

    def close(self) -> Any:
        close = getattr(self._iterator, "close", None)
        if close is not None:
            with self._lock:
                return self._context.run(close)
        return None

    def throw(self, *args: Any) -> Any:
        with self._lock:
            return self._context.run(self._iterator.throw, *args)


class RequestTrace:
    """Detached request root: activate explicitly; finish once after streaming.

    Construction/finish leave the caller's context unchanged. Never take a
    parent from browser headers. request_id exists even with tracing disabled.
    """

    def __init__(self, name: str, *, inputs: dict[str, Any] | None = None,
                 metadata: dict[str, Any] | None = None):
        self.request_id = str(uuid4())
        self.run: Any = None
        self._finished = False
        self._lock = Lock()
        self._context = copy_context()
        self._scope = chain_span(name, inputs=inputs, parent="ignore",
                                 metadata={**(metadata or {}), "request_id": self.request_id})
        self.run = self._context.run(self._scope.__enter__)

    @contextmanager
    def activate(self) -> Iterator[None]:
        if self.run is None:
            yield
            return
        # Explicit trusted parent activation, with optional SDK failure isolated.
        with _activate_parent(self.run):
            yield

    def headers(self) -> dict[str, str]:
        if self.run is None:
            return {}
        try:
            return _allowed_headers(self.run.to_headers())
        except Exception:
            return {}

    def finish(self, outputs: dict[str, Any] | None = None, *,
               error: str | BaseException | None = None, cancelled: bool = False) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            result = {**(outputs or {}), "request_id": self.request_id,
                      "ok": error is None and not cancelled, "cancelled": cancelled}
            try:
                if self.run is not None:
                    self.run.end(outputs=result, error=(str(error)[:500] if error else
                                                       "request cancelled" if cancelled else None))
            except Exception as exc:
                _LOGGER.warning("Tracing result unavailable (%s)", type(exc).__name__)
            finally:
                self._context.run(self._scope.__exit__, None, None, None)


def _allowed_headers(headers: Any) -> dict[str, str]:
    try:
        values = dict(headers)
    except (TypeError, ValueError):
        return {}
    return {str(k).lower(): v for k, v in values.items()
            if str(k).lower() in {"langsmith-trace", "baggage"}
            and isinstance(v, str) and len(v) <= 8192
            and "\r" not in v and "\n" not in v}


def current_trace_headers() -> dict[str, str]:
    if not tracing_enabled():
        return {}
    try:
        from langsmith.run_helpers import get_current_run_tree
        run = get_current_run_tree()
        return _allowed_headers(run.to_headers()) if run is not None else {}
    except Exception:
        return {}


@contextmanager
def _activate_parent(parent: Any) -> Iterator[None]:
    try:
        from langsmith import tracing_context
        scope = tracing_context(parent=parent)
        scope.__enter__()
    except Exception as exc:
        _LOGGER.warning("Tracing parent unavailable (%s)", type(exc).__name__)
        yield
        return
    try:
        yield
    finally:
        try:
            scope.__exit__(None, None, None)
        except Exception as exc:
            _LOGGER.warning("Tracing parent cleanup unavailable (%s)", type(exc).__name__)


@contextmanager
def propagated_trace(headers: Any) -> Iterator[None]:
    """Accept ONLY caller-verified internal service headers, never browser input."""
    parent = _allowed_headers(headers)
    if not tracing_enabled() or "langsmith-trace" not in parent:
        yield
        return
    with _activate_parent(parent):
        yield


def current_run_ids() -> dict[str, str]:
    """Return {run_id, trace_id, project} when inside an active LangSmith run."""
    if not tracing_enabled():
        return {}
    try:
        from langsmith.run_helpers import get_current_run_tree
    except Exception:
        return {}
    try:
        rt = get_current_run_tree()
    except Exception:
        return {}
    if rt is None:
        return {}
    out: dict[str, str] = {}
    rid = getattr(rt, "id", None)
    tid = getattr(rt, "trace_id", None) or getattr(rt, "id", None)
    if rid is not None:
        out["langsmith_run_id"] = str(rid)
    if tid is not None:
        out["langsmith_trace_id"] = str(tid)
    project = (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or ""
    ).strip()
    if project:
        out["langsmith_project"] = project
    return out


def attach_outputs(run: Any, outputs: dict[str, Any]) -> None:
    """Attach outputs without ending a span before its context exits."""
    if run is not None:
        try:
            run.add_outputs(outputs)
            # Returned failure payloads do not raise through the span context.
            # Mark these runs as errors without exporting provider response bodies.
            if outputs.get("ok") is False and not getattr(run, "error", None):
                run.error = "operation returned an error"
        except Exception as exc:
            _LOGGER.warning("Tracing outputs unavailable (%s)", type(exc).__name__)


def wrap_generator(name: str, gen: Iterator[Any], **kwargs: Any) -> Iterator[Any]:
    """One live chain per streamed turn; context survives fresh worker calls."""
    if not tracing_enabled():
        return gen

    def traced() -> Iterator[Any]:
        with span(name, run_type=kwargs.get("run_type", "chain"),
                  inputs=kwargs.get("inputs"), metadata=kwargs.get("metadata"),
                  parent=kwargs.get("parent")) as run:
            final_payload: dict[str, Any] = {}
            error_detail: str | None = None
            def observe(event: Any) -> Any:
                nonlocal final_payload, error_detail
                if isinstance(event, dict):
                    if event.get("type") == "final":
                        response = event.get("response")
                        response = response if isinstance(response, dict) else {}
                        final_payload = {
                            "has_final": True,
                            "answer_chars": len(str(response.get("answer_markdown") or "")),
                            "tables": len(response.get("tables") or []),
                            "charts": len(response.get("charts") or []),
                        }
                    elif event.get("type") == "error":
                        error_detail = str(event.get("detail") or "error")[:500]
                return event

            class ObservedIterator:
                # yield-from delegates send/throw/close rather than swallowing them.
                def __iter__(self) -> Any:
                    return self

                def __next__(self) -> Any:
                    return observe(next(gen))

                def send(self, value: Any) -> Any:
                    return observe(gen.send(value))

                def throw(self, *args: Any) -> Any:
                    throw = getattr(gen, "throw", None)
                    if throw is None:
                        raise args[1] if len(args) > 1 else args[0]
                    return observe(throw(*args))

                def close(self) -> Any:
                    close = getattr(gen, "close", None)
                    return close() if close is not None else None

            try:
                yield from ObservedIterator()
                attach_outputs(run, {"ok": error_detail is None, **final_payload})
                if error_detail and run is not None:
                    run.error = error_detail
            except GeneratorExit:
                attach_outputs(run, {"ok": False, "cancelled": True})
                # Treat an intentional disconnect as cancellation, not success.
                if run is not None:
                    run.error = "stream cancelled"
                raise
            except BaseException:
                attach_outputs(run, {"ok": False})
                raise
            finally:
                close = getattr(gen, "close", None)
                if close is not None:
                    close()

    return ContextIterator(traced())
