import { useCallback, useEffect, useMemo, useState } from "react";
import type { ProgressEvent, StreamTimings } from "../lib/api";
import { formatTimingMs } from "../lib/api";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";

/** Clipboard copy button for assistant messages. */
export function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const onClick = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  }, [text]);
  return (
    <button
      className="absolute right-2 top-2 rounded-[8px] border border-line bg-white/92 px-2.5 py-1 text-[11px]! font-semibold! text-muted-foreground opacity-0 group-hover:opacity-100 hover:bg-brand-soft hover:text-brand-deep"
      onClick={onClick}
      aria-label="Copy to clipboard"
      title="Copy"
    >
      {copied ? "✓ Copied" : "Copy"}
    </button>
  );
}

/** Full-panel history load state (sidebar session click): skeleton chat layout. */
export function HistoryLoadingPanel({ label = "Loading chat" }: { label?: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      aria-busy="true"
      aria-label={label}
      className="mx-auto flex min-h-0 w-full max-w-[820px] flex-1 flex-col gap-4 overflow-hidden py-6"
    >
      <div className="flex justify-end">
        <Skeleton className="h-11 w-56 rounded-[18px] rounded-br-[8px] bg-brand-soft" />
      </div>
      <div className="flex justify-start">
        <div className="flex w-full max-w-[720px] flex-col gap-3 rounded-[18px] rounded-bl-[8px] border border-border bg-white p-4 shadow-sm">
          <Skeleton className="h-3.5 w-11/12" />
          <Skeleton className="h-3.5 w-4/5" />
          <Skeleton className="h-3.5 w-3/5" />
          <Skeleton className="mt-1 h-36 w-full rounded-xl" />
        </div>
      </div>
      <div className="flex justify-end">
        <Skeleton className="h-11 w-40 rounded-[18px] rounded-br-[8px] bg-brand-soft" />
      </div>
      <div className="flex justify-start">
        <div className="flex w-full max-w-[720px] flex-col gap-3 rounded-[18px] rounded-bl-[8px] border border-border bg-white p-4 shadow-sm">
          <Skeleton className="h-3.5 w-10/12" />
          <Skeleton className="h-3.5 w-2/5" />
        </div>
      </div>
    </div>
  );
}

/** Streaming placeholder: stage-aware label + shimmering answer lines. */
export function TypingIndicator({ label = "Thinking" }: { label?: string }) {
  return (
    <div
      role="status"
      aria-label="Assistant is typing"
      className="flex w-full flex-col gap-3"
    >
      <div className="flex items-center gap-2">
        <span aria-hidden="true" className="size-1.5 animate-pulse rounded-full bg-primary" />
        <span className="text-[13px] text-muted-foreground">{label}</span>
      </div>
      <div aria-hidden="true" className="flex flex-col gap-2.5">
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-4/5" />
        <Skeleton className="h-3 w-3/5" />
      </div>
    </div>
  );
}

type ToolStep = {
  id: string;
  label: string;
  detail: string;
  state: "pending" | "running" | "done";
};

type TimelineStep = {
  id: string;
  label: string;
  detail?: string;
  state: "pending" | "running" | "done" | "error";
};

function mergeProgressEvents(events: ProgressEvent[]): TimelineStep[] {
  const order: string[] = [];
  const map = new Map<string, TimelineStep>();
  for (const ev of events || []) {
    const id = String(ev.stage || ev.label || "stage").toLowerCase();
    if (id === "complete") continue;
    if (!map.has(id)) {
      order.push(id);
      map.set(id, {
        id,
        label: ev.label || id,
        detail: ev.detail,
        state: "pending",
      });
    }
    const step = map.get(id)!;
    if (ev.label) step.label = ev.label;
    if (ev.detail) step.detail = ev.detail;
    const st = String(ev.status || "").toLowerCase();
    if (st === "end" || st === "done" || st === "complete") step.state = "done";
    else if (st === "error" || st === "failed") step.state = "error";
    else step.state = "running";
  }
  return order.map((id) => map.get(id)!);
}

/** Compact latency extras — Debug mode only (First update / Answer started / Total). */
export function ResponseTiming({
  timings,
  live = false,
  debug = false,
}: {
  timings: StreamTimings | null | undefined;
  live?: boolean;
  debug?: boolean;
}) {
  if (!debug || !timings) return null;
  const mode = timings.fellBack
    ? "Standard reply"
    : timings.usedStream
      ? "Live updates"
      : live
        ? "Working"
        : "Reply";
  return (
    <div
      className={cn(
        "mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-[10px] border border-dashed border-line bg-tint-bg-2 px-2.5 py-1.5 text-[11px] text-muted-foreground",
        live
          ? "mt-0 border-solid border-timing-line bg-run-gradient"
          : "",
      )}
      data-testid="response-timing"
      title="Timing for this reply (local clock)"
    >
      <span className="font-[650] tracking-[0.02em] text-brand-deep">{mode}</span>
      <span>
        First update{" "}
        <strong className="font-[650] tabular-nums text-ink-soft">
          {formatTimingMs(timings.firstProgressMs)}
        </strong>
      </span>
      <span>
        Answer started{" "}
        <strong className="font-[650] tabular-nums text-ink-soft">
          {formatTimingMs(timings.firstDeltaMs)}
        </strong>
      </span>
      <span>
        Total{" "}
        <strong className="font-[650] tabular-nums text-ink-soft">
          {formatTimingMs(timings.totalMs)}
        </strong>
      </span>
    </div>
  );
}

/** Empty / soft-error assistant body when stream/chat returns no text. */
export function EmptyAnswerState({
  fellBack = false,
}: {
  fellBack?: boolean;
}) {
  return (
    <div className="px-0.5 pt-2.5 pb-1" role="status" data-testid="empty-answer">
      <div className="text-sm font-[650] text-brand-deep">No answer text returned</div>
      <div className="mt-1 text-[13px] leading-[1.45] text-ink-soft">
        {fellBack
          ? "Live updates were unavailable and the standard reply came back empty. Please retry or start a new chat."
          : "The reply finished without any text. Please retry your question."}
      </div>
    </div>
  );
}

/**
 * Live stage timeline driven by SSE / progress_events.
 * Prefer this over the TypingIndicator skeleton during send (LibreChat mid-wait tool visibility).
 */
export function ProgressTimeline({
  events,
  busy,
  timings,
  debug = false,
}: {
  events: ProgressEvent[];
  busy: boolean;
  timings?: StreamTimings | null;
  debug?: boolean;
}) {
  const steps = useMemo(() => mergeProgressEvents(events), [events]);

  if (!busy) return null;

  // Before first event arrives, show a single pending chip (not the skeleton panel).
  const display: TimelineStep[] =
    steps.length > 0
      ? steps
      : [
          {
            id: "route",
            label: "Understanding question",
            detail: "Starting…",
            state: "running",
          },
        ];

  const active = display.find((s) => s.state === "running") || display[display.length - 1];

  return (
    <div
      className="mx-auto mb-3 grid w-[min(820px,100%)] animate-[rise_220ms_var(--ease)] gap-2"
      aria-live="polite"
      data-testid="progress-timeline"
    >
      <div className="flex flex-wrap gap-1.5" role="list">
        {display.map((s) => (
          <div
            key={s.id}
            role="listitem"
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full border border-line bg-white/95 px-2.5 py-[5px] text-xs text-ink-soft shadow-sm",
              s.state === "running" &&
                "border-run-line bg-run-gradient text-brand-deep",
              s.state === "done" && "border-ok-line bg-ok-bg text-ok-fg",
              s.state === "error" && "border-err-line bg-err-bg text-err-fg",
            )}
            title={s.detail || s.label}
          >
            <span
              className={cn(
                "size-[7px] shrink-0 rounded-full",
                s.state === "running"
                  ? "animate-[dot-pulse_1.1s_infinite_ease-in-out] bg-primary shadow-[0_0_0_3px_rgba(11,92,171,0.14)]"
                  : s.state === "done"
                    ? "bg-ok-dot"
                    : s.state === "error"
                      ? "bg-err-dot-strong"
                      : "bg-dot-idle",
              )}
            />
            <span className="font-semibold tracking-[0.01em]">{s.label}</span>
          </div>
        ))}
      </div>
      {active && (
        <div className="inline-flex items-center gap-2 rounded-[12px] border border-line bg-white/95 px-3 py-2 shadow-sm">
          <span
            className={cn(
              "size-2 rounded-full",
              active.state === "done"
                ? "bg-ok-dot"
                : "animate-[dot-pulse_1.1s_infinite_ease-in-out] bg-primary shadow-[0_0_0_4px_rgba(11,92,171,0.12)]",
            )}
          />
          <span className="text-[13px] text-ink-soft">
            {active.state === "done"
              ? "Finishing…"
              : active.detail
                ? `${active.label} — ${active.detail}`
                : `${active.label}…`}
          </span>
        </div>
      )}
      <ResponseTiming timings={timings} live debug={debug} />
    </div>
  );
}

type LiveToolCard = {
  id: string;
  label: string;
  detail: string;
  state: "running" | "done" | "error";
  toolName?: string;
  sql?: string;
  query?: string;
  elapsedMs?: number;
  startedAt?: number;
  ok?: boolean;
};

function formatToolDuration(ms?: number | null): string {
  if (ms == null || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms >= 10000 ? 1 : 2)} s`;
}

function mergeToolCards(events: ProgressEvent[]): LiveToolCard[] {
  const order: string[] = [];
  const map = new Map<string, LiveToolCard>();
  for (const ev of events || []) {
    const t = String(ev.type || "").toLowerCase();
    const isTool = t === "tool_start" || t === "tool_end" || Boolean(ev.tool_id);
    if (!isTool) continue;
    const id = String(ev.tool_id || `${ev.tool_name || ev.label}-${order.length}`);
    if (!map.has(id)) {
      order.push(id);
      map.set(id, {
        id,
        label: ev.label || "Working",
        detail: ev.detail || "",
        state: "running",
        toolName: ev.tool_name,
        startedAt: typeof ev.ts === "number" ? ev.ts : undefined,
      });
    }
    const card = map.get(id)!;
    if (ev.label) card.label = ev.label;
    if (ev.detail) card.detail = ev.detail;
    if (ev.tool_name) card.toolName = ev.tool_name;
    if (ev.sql) card.sql = ev.sql;
    if (ev.query) card.query = ev.query;
    if (typeof ev.elapsed_ms === "number") card.elapsedMs = ev.elapsed_ms;
    if (typeof ev.ok === "boolean") card.ok = ev.ok;
    if (typeof ev.ts === "number" && t === "tool_start") card.startedAt = ev.ts;
    // If end event lacks elapsed_ms, derive from start/end timestamps.
    if (
      (t === "tool_end" || String(ev.status || "").toLowerCase() === "end") &&
      card.elapsedMs == null &&
      typeof card.startedAt === "number" &&
      typeof ev.ts === "number"
    ) {
      card.elapsedMs = Math.max(0, (ev.ts - card.startedAt) * 1000);
    }
    const st = String(ev.status || "").toLowerCase();
    if (t === "tool_end" || st === "end" || st === "done" || st === "complete") {
      card.state = ev.ok === false || st === "error" ? "error" : "done";
    } else if (st === "error" || st === "failed") {
      card.state = "error";
    } else {
      card.state = "running";
    }
  }
  return order.map((id) => map.get(id)!);
}

/**
 * LibreChat-style live tool cards.
 * Prefer SSE tool_start/tool_end events. Soft fallback only when no live events.
 */
export function ToolCallTimeline({
  busy,
  events = [],
  debug = false,
  persist = false,
}: {
  busy: boolean;
  events?: ProgressEvent[];
  debug?: boolean;
  persist?: boolean;
  question?: string;
  hasLiveProgress?: boolean;
}) {
  const liveCards = useMemo(() => mergeToolCards(events), [events]);
  const [fallback, setFallback] = useState<ToolStep[]>([]);
  // Debug stays expanded after finish so tool timings stay visible without an extra click.
  const [expanded, setExpanded] = useState(busy || debug);

  useEffect(() => {
    // Live wait: always expanded. Finished + Debug ON: keep open. Else collapse to summary.
    setExpanded(busy || debug);
  }, [busy, debug]);

  useEffect(() => {
    if (!busy || liveCards.length > 0) {
      setFallback([]);
      return;
    }
    setFallback([
      {
        id: "orchestrator",
        label: "DualSource Agent",
        detail: "Understanding your question…",
        state: "running",
      },
    ]);
    const t1 = window.setTimeout(() => {
      setFallback([
        {
          id: "orchestrator",
          label: "DualSource Agent",
          detail: "Querying internal data and public sources…",
          state: "running",
        },
      ]);
    }, 2200);
    const t2 = window.setTimeout(() => {
      setFallback([
        {
          id: "orchestrator",
          label: "DualSource Agent",
          detail: "Preparing your answer…",
          state: "running",
        },
      ]);
    }, 8000);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
    };
  }, [busy, liveCards.length]);

  const show = busy || (persist && liveCards.length > 0);
  if (!show) return null;
  if (!liveCards.length && fallback.length === 0) return null;

  const doneMs = liveCards.reduce((sum, c) => sum + (c.elapsedMs || 0), 0);
  const summary =
    liveCards.length > 0
      ? `${liveCards.length} step${liveCards.length === 1 ? "" : "s"}${doneMs ? ` · ${formatToolDuration(doneMs)}` : ""}`
      : "Working…";

  return (
    <div
      className="mx-auto mb-3 grid w-[min(820px,100%)] gap-2"
      aria-live="polite"
      data-testid="tool-timeline"
    >
      <button
        type="button"
        className="flex w-full cursor-pointer items-center justify-between gap-2.5 border-0 bg-transparent p-0 text-left"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        <span className="px-0.5 text-[11px] font-[650] uppercase tracking-[0.06em] text-muted-foreground">
          Working steps
        </span>
        <span className="whitespace-nowrap text-[11px] font-semibold text-muted-foreground">
          {expanded ? (busy ? "Live" : "Hide") : summary}
        </span>
      </button>
      {expanded &&
        (liveCards.length > 0
          ? liveCards.map((s) => (
              <div
                key={s.id}
                className={cn(
                  "rounded-md border border-line bg-white/92 px-3 py-2.5 shadow-sm animate-[rise_220ms_var(--ease)]",
                  s.state === "running" && "border-run-line bg-run-gradient",
                  s.state === "done" && "opacity-[0.92]",
                  s.state === "error" && "border-err-line bg-err-bg",
                )}
                data-testid="tool-card"
              >
                <div className="flex items-center gap-2">
                  <span
                    className={cn(
                      "size-2 rounded-full",
                      s.state === "running"
                        ? "animate-[dot-pulse_1.1s_infinite_ease-in-out] bg-primary shadow-[0_0_0_4px_rgba(11,92,171,0.12)]"
                        : s.state === "done"
                          ? "bg-ok-dot"
                          : s.state === "error"
                            ? "bg-err-dot"
                            : "bg-dot-idle",
                    )}
                  />
                  <span className="text-[13px] font-[650] text-brand-deep">{s.label}</span>
                  <span
                    className="ml-auto text-[11px] uppercase tracking-[0.04em] text-muted-foreground tabular-nums"
                    title={s.elapsedMs != null ? `Tool runtime ${formatToolDuration(s.elapsedMs)}` : undefined}
                  >
                    {s.state === "running"
                      ? "Working"
                      : s.state === "error"
                        ? s.elapsedMs != null
                          ? `Failed · ${formatToolDuration(s.elapsedMs)}`
                          : "Failed"
                        : s.elapsedMs != null
                          ? formatToolDuration(s.elapsedMs)
                          : "Done"}
                  </span>
                </div>
                {s.detail && <div className="mt-1 text-[12.5px] text-ink-soft">{s.detail}</div>}
                {debug && (
                  <div className="mt-2 grid gap-1.5 border-t border-dashed border-line-strong pt-2">
                    <div className="flex items-start gap-2 text-[11px] text-muted-foreground">
                      <span className="w-[42px] shrink-0 grow-0 font-[650] uppercase tracking-[0.04em]">duration</span>
                      <code className="m-0 whitespace-pre-wrap break-words rounded-[8px] border border-line bg-code-bg px-2 py-1.5 font-mono text-[11px] text-ink" data-testid="tool-duration">
                        {formatToolDuration(s.elapsedMs)}
                      </code>
                    </div>
                    {s.toolName && (
                      <div className="flex items-start gap-2 text-[11px] text-muted-foreground">
                        <span className="w-[42px] shrink-0 grow-0 font-[650] uppercase tracking-[0.04em]">tool</span>
                        <code className="m-0 whitespace-pre-wrap break-words rounded-[8px] border border-line bg-code-bg px-2 py-1.5 font-mono text-[11px] text-ink">{s.toolName}</code>
                      </div>
                    )}
                    {s.query && (
                      <div className="flex items-start gap-2 text-[11px] text-muted-foreground">
                        <span className="w-[42px] shrink-0 grow-0 font-[650] uppercase tracking-[0.04em]">query</span>
                        <code className="m-0 whitespace-pre-wrap break-words rounded-[8px] border border-line bg-code-bg px-2 py-1.5 font-mono text-[11px] text-ink">{s.query}</code>
                      </div>
                    )}
                    {s.sql && (
                      <div className="grid gap-1.5">
                        <div className="flex items-start gap-2 text-[11px] text-muted-foreground">
                          <span className="w-[42px] shrink-0 grow-0 font-[650] uppercase tracking-[0.04em]">sql</span>
                        </div>
                        <pre className="m-0 max-h-[180px] overflow-auto whitespace-pre-wrap break-words rounded-[8px] border border-line bg-code-bg px-2 py-1.5 font-mono text-[11px] text-ink">{s.sql}</pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))
          : fallback.map((s) => (
              <div
                key={s.id}
                className="rounded-md border border-run-line bg-run-gradient px-3 py-2.5 shadow-sm animate-[rise_220ms_var(--ease)]"
              >
                <div className="flex items-center gap-2">
                  <span className="size-2 animate-[dot-pulse_1.1s_infinite_ease-in-out] rounded-full bg-primary shadow-[0_0_0_4px_rgba(11,92,171,0.12)]" />
                  <span className="text-[13px] font-[650] text-brand-deep">{s.label}</span>
                  <span className="ml-auto text-[11px] uppercase tracking-[0.04em] text-muted-foreground">Working</span>
                </div>
                <div className="mt-1 text-[12.5px] text-ink-soft">{s.detail}</div>
              </div>
            )))}
    </div>
  );
}

/** Post-answer debug panel (SQL / tool trace + per-tool ms). Hidden unless Debug mode on. */
export function DebugPanel({
  debug,
  payload,
}: {
  debug: boolean;
  payload?: {
    last_sql?: string | null;
    tool_calls?: any[];
    status?: Record<string, string>;
    timings_ms?: Record<string, number>;
  } | null;
}) {
  const [open, setOpen] = useState(false);
  if (!debug || !payload) return null;
  const tools = Array.isArray(payload.tool_calls) ? payload.tool_calls : [];
  const sql = payload.last_sql || tools.map((t) => t?.sql).find(Boolean) || "";
  const totalToolMs = tools.reduce((sum, t) => sum + (Number(t?.ms) || 0), 0);
  return (
    <div className="mt-2.5" data-testid="debug-panel">
      <button
        type="button"
        className="rounded-[10px] border border-dashed border-line-strong bg-tint-bg-2 px-2.5 py-1.5 text-xs! text-ink-soft"
        onClick={() => setOpen((v) => !v)}
      >
        <span>{open ? "Hide debug" : "Show debug (SQL / tools / timings)"}</span>
      </button>
      {open && (
        <div className="mt-2 grid gap-2">
          {payload.status && (
            <div className="grid gap-1">
              <div className="text-[11px] font-bold uppercase tracking-[0.05em] text-muted-foreground">status</div>
              <pre className="m-0 max-h-60 overflow-auto whitespace-pre-wrap break-words rounded-[8px] border border-line bg-code-bg px-2 py-1.5 font-mono text-[11px] text-ink">{JSON.stringify(payload.status, null, 2)}</pre>
            </div>
          )}
          {tools.length > 0 && (
            <div className="grid gap-1">
              <div className="text-[11px] font-bold uppercase tracking-[0.05em] text-muted-foreground">
                tool timings (sum {formatToolDuration(totalToolMs)})
              </div>
              {/* Cell padding/weight overrides live in shadcn.css scoped
                  [data-debug-timing] rules — global table/th/td element rules
                  are unlayered and beat utility classes. */}
              <table data-debug-timing className="mt-1">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>tool</th>
                    <th>bucket</th>
                    <th>duration</th>
                    <th>ok</th>
                  </tr>
                </thead>
                <tbody>
                  {tools.map((t, idx) => (
                    <tr key={`${t?.name || "tool"}-${idx}`}>
                      <td>{idx + 1}</td>
                      <td><code className="font-mono text-[11px]">{String(t?.name || "?")}</code></td>
                      <td>{String(t?.bucket || "—")}</td>
                      <td>{formatToolDuration(Number(t?.ms))}</td>
                      <td>{t?.ok === false ? "no" : "yes"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {payload.timings_ms && (
            <div className="grid gap-1">
              <div className="text-[11px] font-bold uppercase tracking-[0.05em] text-muted-foreground">bucket totals</div>
              <pre className="m-0 max-h-60 overflow-auto whitespace-pre-wrap break-words rounded-[8px] border border-line bg-code-bg px-2 py-1.5 font-mono text-[11px] text-ink">{JSON.stringify(payload.timings_ms, null, 2)}</pre>
            </div>
          )}
          {sql && (
            <div className="grid gap-1">
              <div className="text-[11px] font-bold uppercase tracking-[0.05em] text-muted-foreground">last SQL</div>
              <pre className="m-0 max-h-60 overflow-auto whitespace-pre-wrap break-words rounded-[8px] border border-line bg-code-bg px-2 py-1.5 font-mono text-[11px] text-ink">{sql}</pre>
            </div>
          )}
          {tools.length > 0 && (
            <div className="grid gap-1">
              <div className="text-[11px] font-bold uppercase tracking-[0.05em] text-muted-foreground">tool calls (raw)</div>
              <pre className="m-0 max-h-60 overflow-auto whitespace-pre-wrap break-words rounded-[8px] border border-line bg-code-bg px-2 py-1.5 font-mono text-[11px] text-ink">{JSON.stringify(tools, null, 2)}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Hide trivial 1-row KPI tables; the answer text already carries the figure. */
export function shouldShowTable(table: any): boolean {
  if (table?.ui_hidden) return false;
  const columns: string[] = Array.isArray(table?.columns) ? table.columns : [];
  const rows: any[][] = Array.isArray(table?.rows) ? table.rows : [];
  if (!columns.length || !rows.length) return false;
  if (rows.length === 1 && columns.length <= 2) return false;
  return true;
}

/** Prefer a single under-chart table when peer compare + Internal monthly both exist. */
export function pickDisplayTables(tables: any[] | undefined | null, hasChart: boolean): any[] {
  const visible = (Array.isArray(tables) ? tables : []).filter((t) => shouldShowTable(t));
  if (!hasChart || visible.length <= 1) return visible;
  const compare = visible.filter(
    (t) =>
      String(t?.source_kind || "") === "peer_compare" ||
      /airport/i.test(String((t?.columns || [])[0] || "")) ||
      /passengers_millions|asian airports|peer/i.test(String(t?.title || "")),
  );
  if (compare.length) return compare.slice(0, 1);
  return visible.slice(0, 1);
}

function tableSourceLabel(table: any): string {
  const raw = String(table?.display_source || table?.source_kind || "").trim();
  if (/peer|mix|public/i.test(raw)) return "Internal + public";
  if (raw) return raw;
  return "Internal";
}

export function polishTitle(title: string | undefined | null): string {
  let t = String(title || "").trim();
  // Strip accidental HTML / duplicated title artifacts from model/tool output.
  t = t.replace(/[<>]/g, " ");
  t = t.replace(/"+/g, "");
  // Strip Genie restatement openers / verbose wrappers.
  t = t.replace(
    /^(?:you want(?:ed)? to see|here (?:is|are)|this (?:shows?|displays?)|the total number of|the total)\s+/i,
    "",
  );
  t = t.replace(/\s*\(showing\s+\d+\s+of\s+\d+\s+rows\)\s*$/i, "");
  t = t.replace(/,?\s*showing\b.*$/i, "");
  t = t.replace(/\s+at\s+(?:the hub|Hong Kong International Airport)\b/gi, "");
  t = t.replace(/\s+for each of the months?\b.*$/i, "");
  t = t.replace(/\s+for each month\b.*$/i, " by month");
  t = t.replace(/\s+over the (?:last|latest) 12 months\.?$/i, " — latest 12 months");
  t = t.replace(/\s+/g, " ").trim();
  if (!t) return "Query result";
  t = t.charAt(0).toUpperCase() + t.slice(1);
  if (t.length > 72) t = `${t.slice(0, 69)}…`;
  return t;
}

export function humanizeColumn(col: string): string {
  const raw = String(col || "").trim();
  if (!raw) return "";
  const key = raw.toLowerCase();
  const map: Record<string, string> = {
    year_month: "Year month",
    od_passengers: "OD passengers",
    od_yoy_change: "OD YoY change",
    od_yoy_pct: "OD YoY %",
    od_share_pct: "OD share %",
    transfer_passengers: "Transfer passengers",
    transfer_yoy_change: "Transfer YoY change",
    transfer_yoy_pct: "Transfer YoY %",
    transfer_share_pct: "Transfer share %",
    total_passengers: "Total passengers",
    cargo_volume_tonnes: "Cargo (tonnes)",
    monthly_cargo_tonnes: "Monthly cargo (tonnes)",
    passenger_share_pct: "Passenger share %",
    carrier_passengers: "Carrier passengers",
    airline_name: "Airline",
  };
  if (map[key]) return map[key];
  return raw
    .replace(/_/g, " ")
    .replace(/\bpct\b/gi, "%")
    .replace(/\byoy\b/gi, "YoY")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatDisplayDate(value: unknown, column?: string): string | null {
  // Month-grain columns / YYYY-MM → YYYY-MM. Daily/weekly keep YYYY-MM-DD.
  const col = (column || "").toLowerCase();
  const monthGrainCol = col === "year_month" || col.endsWith("_month") || col === "month";

  if (value instanceof Date && !Number.isNaN(value.getTime())) {
    const y = value.getUTCFullYear();
    const m = String(value.getUTCMonth() + 1).padStart(2, "0");
    const d = String(value.getUTCDate()).padStart(2, "0");
    if (monthGrainCol) return `${y}-${m}`;
    return `${y}-${m}-${d}`;
  }
  if (typeof value !== "string") return null;
  const s = value.trim();
  if (/^\d{4}-\d{2}$/.test(s)) return s;
  const iso = /^(\d{4})-(\d{2})-(\d{2})(?:[T\s].*)?$/.exec(s);
  if (!iso) return null;
  const [, y, m, d] = iso;
  if (monthGrainCol) return `${y}-${m}`;
  // year_month often arrives as timestamp on the 1st — collapse only then.
  if (d === "01" && /T00:00:00/.test(s) && !/week|date|day|flight_date/.test(col)) {
    return `${y}-${m}`;
  }
  return `${y}-${m}-${d}`;
}

/** Normalize table rows: agent often returns objects; FE table expects arrays. */
export function normalizeTableRows(table: any): any[][] {
  const columns: string[] = Array.isArray(table?.columns) ? table.columns.map(String) : [];
  const raw = Array.isArray(table?.rows) ? table.rows : [];
  return raw.map((row: any) => {
    if (Array.isArray(row)) return row;
    if (row && typeof row === "object") {
      if (columns.length) return columns.map((c) => (row as any)[c]);
      return Object.values(row);
    }
    return [row];
  });
}

export function DataTableCard({
  table,
  defaultCollapsed = false,
}: {
  table: any;
  /** When a ChartCard already visualizes this SQL result, keep the raw table hidden. */
  defaultCollapsed?: boolean;
}) {
  const columns: string[] = Array.isArray(table?.columns) ? table.columns : [];
  const rows = normalizeTableRows(table);
  const [open, setOpen] = useState(!defaultCollapsed);
  useEffect(() => {
    setOpen(!defaultCollapsed);
  }, [defaultCollapsed, table?.title, rows.length]);
  if (!shouldShowTable({ ...table, rows })) return null;

  if (!open) {
    return (
      <div
        className="mt-3 overflow-hidden rounded-md border border-dashed border-line bg-tint-bg-2"
        data-testid="table-card-collapsed"
      >
        <button
          type="button"
          className="flex w-full cursor-pointer items-center justify-between gap-2.5 border-0 bg-transparent px-3 py-2.5 text-left hover:bg-tint-hover"
          onClick={() => setOpen(true)}
          title="Show the SQL result table behind this chart"
        >
          <span className="text-xs font-[650] text-brand-deep">Show data table</span>
          <span className="text-[11px] text-muted-foreground">
            {rows.length} rows · from {tableSourceLabel(table)}
          </span>
        </button>
      </div>
    );
  }

  return (
    <div className="mt-3 overflow-hidden rounded-md border border-line bg-tint-bg" data-testid="table-card">
      <div className="flex items-center justify-between gap-2 border-b border-cell-line bg-head-gradient px-3 py-2.5">
        <div className="text-[13px] font-[650] text-brand-deep">{polishTitle(table?.title)}</div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="rounded-full border border-line bg-white px-2 py-0.5 text-[11px]! text-ink-soft"
            onClick={() => setOpen(false)}
            title="Collapse the table"
          >
            <span>Hide</span>
          </button>
          <div className="text-[11px] text-muted-foreground">{rows.length} rows</div>
        </div>
      </div>
      {/* Table cells keep the global app.css table/th/td element styling
          (shared with markdown tables) — only the chrome is utility-based. */}
      <div className="overflow-x-auto">
        <table>
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={String(c)}>{humanizeColumn(String(c))}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 20).map((row, rIdx) => (
              <tr key={rIdx}>
                {(Array.isArray(row) ? row : [row]).map((cell, cIdx) => (
                  <td key={cIdx}>{formatCell(cell, columns[cIdx])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function parseNumeric(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return null;
  const cleaned = value.replace(/,/g, "").replace(/%/g, "").trim();
  if (!/^-?\d+(\.\d+)?$/.test(cleaned)) return null;
  const n = Number(cleaned);
  return Number.isFinite(n) ? n : null;
}

function isPercentColumn(column?: string): boolean {
  const col = (column || "").toLowerCase();
  return (
    col.includes("pct") ||
    col.includes("percent") ||
    col.includes("share") ||
    col.endsWith("_pp_change") ||
    col.includes("yoy_pct")
  );
}

function formatCell(value: unknown, column?: string): string {
  const asDate = formatDisplayDate(value, column);
  if (asDate) return asDate;

  const n = parseNumeric(value);
  if (n === null) return String(value ?? "");
  const col = (column || "").toLowerCase();
  // Keep calendar years ungrouped (avoid "2,026").
  if (
    Number.isInteger(n) &&
    n >= 1900 &&
    n <= 2100 &&
    (col === "year" || col.endsWith("_year") || (col.includes("year") && !col.includes("year_month")))
  ) {
    return String(n);
  }
  if (isPercentColumn(column)) {
    // Agent/Genie pct & share columns are already on 0-100 scale
    // (e.g. 50.06, 1.42, 0.89). Do NOT auto-multiply small values by 100 —
    // the old abs<=1.5 heuristic turned United's 1.42 into 142%.
    return `${n.toLocaleString(undefined, {
      maximumFractionDigits: Number.isInteger(n) ? 0 : 2,
    })}%`;
  }
  return n.toLocaleString(undefined, {
    maximumFractionDigits: Number.isInteger(n) ? 0 : 2,
  });
}
