"""MLflow tracing utilities for DualSource Agent.

Requires mlflow>=3.1.0 for production tracing on Model Serving.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def start_trace(name: str = "dualsource-turn") -> Iterator[str]:
    """Best-effort MLflow trace context; always yields a trace id.

    MLflow 3+ traces are automatically captured when the model runs inside
    a Databricks Serving endpoint. This wrapper adds a named span so that
    each turn is clearly visible in the Traces tab.
    """
    trace_id = uuid.uuid4().hex
    try:
        import mlflow

        with mlflow.start_span(name=name) as span:
            span_id = (
                getattr(span, "span_id", None)
                or getattr(span, "request_id", None)
                or getattr(span, "trace_id", None)
            )
            if span_id:
                trace_id = str(span_id)
            yield trace_id
            return
    except Exception:
        pass
    yield trace_id
