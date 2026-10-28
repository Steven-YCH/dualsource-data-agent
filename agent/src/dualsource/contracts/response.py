"""Response contract for DualSource Agent agent.

Clickable-citation model (v11 — ECharts):
- Sources carry a `cite_tag` string that appears verbatim in answer_markdown:
  internal = "[Internal]", external = "[Source Name](url)" (standard markdown link).
- ReactMarkdown renders external citations as clickable links with zero frontend work.
- Charts use ECharts option payloads under `echarts` (not Chart.js).
  `position` hint: "inline" means the LLM placed [[CHART:n]] marker; "end" appends after text.
  `index` tracks which [[CHART:n]] placeholder maps to this chart.
  `type` is the chart type: "line", "bar", or "pie".
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


SourceType = Literal["hkia_internal", "public"]
ToolState = Literal["ok", "insufficient", "error", "skipped"]


class Source(BaseModel):
    type: SourceType
    label: str
    url: str | None = None
    cite_tag: str = ""  # Tag used in answer_markdown, e.g. "[Internal]" or "[Label](url)"


class TablePayload(BaseModel):
    title: str
    columns: list[str]
    rows: list[list[Any]]


class ChartPayload(BaseModel):
    id: str
    title: str
    type: Literal["line", "bar", "pie"] = "line"
    echarts: dict[str, Any]
    index: int = 0  # Maps to [[CHART:n]] placeholder in answer_markdown
    position: Literal["inline", "end"] = "end"  # "inline" if LLM placed [[CHART:n]]


class FilePayload(BaseModel):
    id: str
    type: Literal["xlsx"]  # PPT disabled (Wave 1)
    filename: str
    download_path: str


class ToolStatus(BaseModel):
    genie: ToolState = "skipped"
    web: ToolState = "skipped"
    files: ToolState = "skipped"


class StageEvent(BaseModel):
    """Progress stage event (WS4). See docs/handoff/PROGRESS_EVENTS_CONTRACT.md."""
    type: Literal["stage"] = "stage"
    stage: str
    status: str  # "start" or "end"
    label: str
    ts: float


class AgentResponse(BaseModel):
    answer_markdown: str
    sources: list[Source] = Field(default_factory=list)
    tables: list[TablePayload] = Field(default_factory=list)
    export_tables: list[TablePayload] = Field(default_factory=list)  # full rows for Excel (up to 500)
    charts: list[ChartPayload] = Field(default_factory=list)
    files: list[FilePayload] = Field(default_factory=list)
    followups: list[str] = Field(default_factory=list)
    status: ToolStatus = Field(default_factory=ToolStatus)
    trace_id: str | None = None
    progress_events: list[StageEvent] = Field(default_factory=list)  # WS4: stage timeline
