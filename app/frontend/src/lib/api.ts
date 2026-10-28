export type UserInfo = {
  email: string;
  display_name: string;
  initials: string;
  identity_source?: string;
};

export type ClientHistoryTurn = {
  role: "user" | "assistant";
  content: string;
  /** Prior structured tables for chart/follow-up continuity (assistant turns). */
  tables?: any[];
  sources?: any[];
};

/** Stage / tool progress event (Serving / local_react contract). */
export type ProgressEvent = {
  type?: string;
  stage: string;
  status: "start" | "end" | string;
  label: string;
  ts?: number;
  detail?: string;
  /** LibreChat-style per-tool card fields */
  tool_id?: string;
  tool_name?: string;
  sql?: string;
  query?: string;
  ok?: boolean;
  elapsed_ms?: number;
  rows?: number;
  sources_n?: number;
};

export type DebugPayload = {
  last_sql?: string | null;
  tool_calls?: any[];
  status?: Record<string, string>;
};

/** Progressive chart event — chart payload ready before the final response. */
export type ChartStreamEvent = {
  index: number;
  chart: any;
  ts?: number;
};

export type ChatResponsePayload = {
  answer_markdown?: string;
  sources?: any[];
  tables?: any[];
  charts?: any[];
  files?: any[];
  followups?: string[];
  status?: Record<string, string>;
  trace_id?: string;
  progress_events?: ProgressEvent[];
  debug?: DebugPayload;
};

export type ChatResult = {
  session_id: string;
  message?: any;
  response?: ChatResponsePayload;
};

export type StreamHandlers = {
  onProgress?: (event: ProgressEvent) => void;
  onDelta?: (text: string, accumulated: string) => void;
  /** Fired when a chart is ready mid-stream (before the final payload). */
  onChart?: (event: ChartStreamEvent) => void;
  onFinal?: (result: ChatResult) => void;
  /** Fired once when SSE is unavailable and JSON /api/chat will be used. */
  onFallback?: (reason: string) => void;
};

/** Client-side wall-clock timings for Phase 1 latency acceptance. */
export type StreamTimings = {
  /** performance.now() at send start */
  t0: number;
  /** ms until first progress/stage event (null if never) */
  firstProgressMs: number | null;
  /** ms until first answer delta (null if never) */
  firstDeltaMs: number | null;
  /** ms until final/complete (null while in flight) */
  totalMs: number | null;
  usedStream: boolean;
  fellBack: boolean;
};

async function readError(res: Response, fallback: string): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") return data.detail;
    if (Array.isArray(data?.detail)) return JSON.stringify(data.detail);
    return fallback;
  } catch {
    return fallback;
  }
}

function friendlyError(raw: string, fallback: string): string {
  const msg = (raw || "").trim();
  if (!msg) return fallback;
  if (/failed to fetch|networkerror|load failed/i.test(msg)) {
    return "Network error — check your connection and retry.";
  }
  if (/aborted|aborterror/i.test(msg)) {
    return "Request cancelled.";
  }
  if (/\b429\b|too many requests|rate limit/i.test(msg)) {
    return "Model rate limit hit — wait ~30–60s and retry the same question.";
  }
  if (/\b408\b|operation was timeout|gateway timeout/i.test(msg)) {
    return "Model timed out on Foundry — retry once; peer/compare asks may need a second try.";
  }
  return msg;
}

export async function getMe(): Promise<UserInfo> {
  const res = await fetch("/api/me");
  if (!res.ok) throw new Error(await readError(res, "failed to load user"));
  return res.json();
}

export async function listSessions() {
  const res = await fetch("/api/sessions");
  if (!res.ok) throw new Error(await readError(res, "failed to list sessions"));
  return res.json();
}

export async function createSession() {
  const res = await fetch("/api/sessions", { method: "POST" });
  if (!res.ok) throw new Error(await readError(res, "failed to create session"));
  return res.json();
}

/** Rename (title) or archive/delete (status) a session. */
export async function patchSession(
  sessionId: string,
  body: { title?: string; status?: string },
) {
  const res = await fetch(`/api/sessions/${sessionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await readError(res, "failed to update session"));
  return res.json();
}

export async function getMessages(sessionId: string) {
  const res = await fetch(`/api/sessions/${sessionId}/messages`);
  if (!res.ok) throw new Error(await readError(res, "failed to load messages"));
  return res.json();
}

function buildChatBody(
  sessionId: string | null,
  message: string,
  clientHistory?: ClientHistoryTurn[],
) {
  const body: {
    session_id: string | null;
    message: string;
    client_history?: ClientHistoryTurn[];
  } = { session_id: sessionId, message };
  if (clientHistory && clientHistory.length > 0) {
    body.client_history = clientHistory.slice(-12);
  }
  return body;
}

export async function sendChat(
  sessionId: string | null,
  message: string,
  signal?: AbortSignal,
  clientHistory?: ClientHistoryTurn[],
): Promise<ChatResult> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildChatBody(sessionId, message, clientHistory)),
    signal,
  });
  if (!res.ok) {
    throw new Error(
      friendlyError(await readError(res, "Chat failed — please retry."), "Chat failed — please retry."),
    );
  }
  return res.json();
}

/** null = unknown, true/false after first probe. */
let streamAvailable: boolean | null = null;

export function getStreamAvailability(): boolean | null {
  return streamAvailable;
}

export function resetStreamAvailability(): void {
  streamAvailable = null;
}

function asProgressEvent(raw: any): ProgressEvent | null {
  if (!raw || typeof raw !== "object") return null;
  const stage = String(raw.stage || raw.name || raw.id || "").trim();
  const label = String(raw.label || raw.message || raw.detail || stage || "Working").trim();
  if (!stage && !label && !raw.tool_id) return null;
  const status = String(raw.status || (raw.type === "tool_end" ? "end" : "start")).toLowerCase();
  const ev: ProgressEvent = {
    type: String(raw.type || "stage"),
    stage: stage || slugStage(label),
    status:
      status === "end" || status === "done" || status === "complete"
        ? "end"
        : status === "error"
          ? "error"
          : "start",
    label,
    ts: typeof raw.ts === "number" ? raw.ts : Date.now() / 1000,
    detail: typeof raw.detail === "string" ? raw.detail : undefined,
  };
  if (raw.tool_id) ev.tool_id = String(raw.tool_id);
  if (raw.tool_name) ev.tool_name = String(raw.tool_name);
  if (typeof raw.sql === "string") ev.sql = raw.sql;
  if (typeof raw.query === "string") ev.query = raw.query;
  if (typeof raw.ok === "boolean") ev.ok = raw.ok;
  if (typeof raw.elapsed_ms === "number") ev.elapsed_ms = raw.elapsed_ms;
  if (typeof raw.rows === "number") ev.rows = raw.rows;
  if (typeof raw.sources_n === "number") ev.sources_n = raw.sources_n;
  return ev;
}

function slugStage(label: string): string {
  return label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "")
    .slice(0, 40) || "stage";
}

function extractDelta(raw: any): string {
  if (!raw || typeof raw !== "object") return "";
  if (typeof raw.delta === "string") return raw.delta;
  if (typeof raw.text === "string") return raw.text;
  if (typeof raw.content === "string") return raw.content;
  if (typeof raw.token === "string") return raw.token;
  if (typeof raw.answer_delta === "string") return raw.answer_delta;
  return "";
}

function normalizeFinalPayload(raw: any): ChatResult {
  // Shapes accepted:
  // 1) { session_id, message, response }
  // 2) { type:"final", ...same }
  // 3) { type:"final", response: {...}, session_id }
  // 4) bare response fields at top level
  if (raw?.session_id || raw?.message || raw?.response) {
    return {
      session_id: raw.session_id,
      message: raw.message,
      response: raw.response || {
        answer_markdown: raw.answer_markdown,
        sources: raw.sources,
        tables: raw.tables,
        charts: raw.charts,
        files: raw.files,
        followups: raw.followups,
        status: raw.status,
        trace_id: raw.trace_id,
        progress_events: raw.progress_events,
        debug: raw.debug,
      },
    };
  }
  return {
    session_id: raw?.session_id || "",
    response: {
      answer_markdown: raw?.answer_markdown || "",
      sources: raw?.sources || [],
      tables: raw?.tables || [],
      charts: raw?.charts || [],
      files: raw?.files || [],
      followups: raw?.followups || [],
      status: raw?.status,
      trace_id: raw?.trace_id,
      progress_events: raw?.progress_events || [],
      debug: raw?.debug,
    },
  };
}

function dispatchStreamEvent(
  data: any,
  handlers: StreamHandlers,
  state: { accumulated: string; gotFinal: boolean; result: ChatResult | null },
): void {
  if (data == null) return;
  const t = String(data.type || data.event || "").toLowerCase();

  // Stage / progress / tool events
  if (
    t === "stage" ||
    t === "progress" ||
    t === "tool" ||
    t === "tool_start" ||
    t === "tool_end" ||
    data.stage
  ) {
    const pe = asProgressEvent(data);
    if (pe) handlers.onProgress?.(pe);
    // Continue — some payloads may also carry deltas
  }

  // Token / answer deltas
  if (t === "token" || t === "delta" || t === "answer_delta" || t === "message_delta") {
    const piece = extractDelta(data);
    if (piece) {
      state.accumulated += piece;
      handlers.onDelta?.(piece, state.accumulated);
    }
    return;
  }

  // Progressive chart payload (ready before final)
  if (t === "chart" || (data.chart && typeof data.chart === "object" && !t)) {
    const chart = data.chart;
    const rawIndex = Number(data.index ?? chart?.index);
    const index = Number.isFinite(rawIndex) ? rawIndex : -1;
    handlers.onChart?.({
      index,
      chart,
      ts: typeof data.ts === "number" ? data.ts : undefined,
    });
    return;
  }

  // Final / complete
  if (t === "final" || t === "complete" || t === "done" || t === "result") {
    const result = normalizeFinalPayload(data);
    // Prefer streamed text if final answer empty
    if (!result.response?.answer_markdown && state.accumulated) {
      result.response = {
        ...(result.response || {}),
        answer_markdown: state.accumulated,
      };
    }
    state.gotFinal = true;
    state.result = result;
    handlers.onFinal?.(result);
    return;
  }

  // Error event from server
  if (t === "error") {
    const msg = String(data.error || data.detail || data.message || "Stream error");
    throw new Error(friendlyError(msg, "Chat failed — please retry."));
  }

  // Bare progress_events array batch
  if (Array.isArray(data.progress_events)) {
    for (const ev of data.progress_events) {
      const pe = asProgressEvent(ev);
      if (pe) handlers.onProgress?.(pe);
    }
  }

  // Delta field without typed event
  const piece = extractDelta(data);
  if (piece && !t) {
    state.accumulated += piece;
    handlers.onDelta?.(piece, state.accumulated);
  }
}

/**
 * Parse SSE (`text/event-stream`) or NDJSON body from fetch ReadableStream.
 * Supports `data: {...}` lines and raw JSON lines.
 */
async function consumeEventStream(
  res: Response,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<ChatResult> {
  if (!res.body) {
    throw new Error("Stream body missing");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const state = { accumulated: "", gotFinal: false, result: null as ChatResult | null };

  const handleBlock = (block: string) => {
    const trimmed = block.trim();
    if (!trimmed || trimmed.startsWith(":")) return;
    // SSE: one or more data: lines
    const dataLines = trimmed
      .split(/\r?\n/)
      .filter((ln) => ln.startsWith("data:"))
      .map((ln) => ln.slice(5).trimStart());
    const payload = dataLines.length ? dataLines.join("\n") : trimmed;
    if (!payload || payload === "[DONE]") return;
    let parsed: any;
    try {
      parsed = JSON.parse(payload);
    } catch {
      // Non-JSON data line → treat as answer delta
      state.accumulated += payload;
      handlers.onDelta?.(payload, state.accumulated);
      return;
    }
    dispatchStreamEvent(parsed, handlers, state);
  };

  try {
    while (true) {
      if (signal?.aborted) {
        throw new DOMException("Aborted", "AbortError");
      }
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Prefer SSE event blocks separated by blank lines
      if (buffer.includes("\n\n") || buffer.includes("\r\n\r\n")) {
        const parts = buffer.split(/\r?\n\r?\n/);
        buffer = parts.pop() || "";
        for (const block of parts) handleBlock(block);
      } else if (buffer.includes("\n")) {
        // NDJSON fallback: process complete lines
        const parts = buffer.split(/\r?\n/);
        buffer = parts.pop() || "";
        for (const line of parts) {
          if (!line.trim()) continue;
          handleBlock(line);
        }
      }
    }
    if (buffer.trim()) handleBlock(buffer);
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* ignore */
    }
  }

  if (state.result) return state.result;
  if (state.accumulated) {
    return {
      session_id: "",
      response: { answer_markdown: state.accumulated, progress_events: [] },
    };
  }
  throw new Error("Stream ended without a final payload");
}

/**
 * POST /api/chat/stream — SSE progress + answer deltas.
 * Throws with status info so caller can fall back to /api/chat.
 */
export async function sendChatStream(
  sessionId: string | null,
  message: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
  clientHistory?: ClientHistoryTurn[],
): Promise<ChatResult> {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream, application/json",
    },
    body: JSON.stringify(buildChatBody(sessionId, message, clientHistory)),
    signal,
  });

  if (res.status === 404 || res.status === 405 || res.status === 501) {
    streamAvailable = false;
    const err = new Error("STREAM_UNAVAILABLE") as Error & { code?: string; status?: number };
    err.code = "STREAM_UNAVAILABLE";
    err.status = res.status;
    throw err;
  }

  if (!res.ok) {
    throw new Error(
      friendlyError(await readError(res, "Chat failed — please retry."), "Chat failed — please retry."),
    );
  }

  const ctype = (res.headers.get("content-type") || "").toLowerCase();
  if (ctype.includes("application/json") && !ctype.includes("text/event-stream")) {
    // Server returned a normal JSON chat response on the stream route
    const json = await res.json();
    const result = normalizeFinalPayload(json);
    if (Array.isArray(result.response?.progress_events)) {
      for (const ev of result.response!.progress_events!) {
        const pe = asProgressEvent(ev);
        if (pe) handlers.onProgress?.(pe);
      }
    }
    if (result.response?.answer_markdown) {
      handlers.onDelta?.(result.response.answer_markdown, result.response.answer_markdown);
    }
    handlers.onFinal?.(result);
    streamAvailable = true;
    return result;
  }

  const result = await consumeEventStream(res, handlers, signal);
  streamAvailable = true;
  return result;
}

function isStreamUnavailableError(e: unknown): boolean {
  if (!e || typeof e !== "object") return false;
  const err = e as { code?: string; message?: string; status?: number };
  if (err.code === "STREAM_UNAVAILABLE") return true;
  if (err.status === 404 || err.status === 405 || err.status === 501) return true;
  return /STREAM_UNAVAILABLE|Failed to fetch|NetworkError|load failed/i.test(String(err.message || ""));
}

/**
 * Prefer SSE `/api/chat/stream`; fall back to `/api/chat` when stream is unavailable.
 * On fallback, `onFallback` fires once with a user-safe reason, then JSON chat runs.
 */
export async function sendChatPreferStream(
  sessionId: string | null,
  message: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
  clientHistory?: ClientHistoryTurn[],
): Promise<ChatResult & { usedStream: boolean; fellBack: boolean }> {
  const tryStream = streamAvailable !== false;
  if (tryStream) {
    try {
      const result = await sendChatStream(sessionId, message, handlers, signal, clientHistory);
      return { ...result, usedStream: true, fellBack: false };
    } catch (e) {
      if (signal?.aborted) throw e;
      if (isStreamUnavailableError(e)) {
        streamAvailable = false;
        handlers.onFallback?.(
          "Live updates unavailable — switching to standard reply.",
        );
        // fall through to JSON chat
      } else {
        throw e;
      }
    }
  } else {
    handlers.onFallback?.("Live updates unavailable — using standard reply.");
  }

  const result = await sendChat(sessionId, message, signal, clientHistory);
  if (Array.isArray(result.response?.progress_events)) {
    for (const ev of result.response!.progress_events!) {
      const pe = asProgressEvent(ev);
      if (pe) handlers.onProgress?.(pe);
    }
  }
  handlers.onFinal?.(result);
  return { ...result, usedStream: false, fellBack: true };
}

export function formatTimingMs(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const s = ms / 1000;
  return s < 10 ? `${s.toFixed(1)} s` : `${Math.round(s)} s`;
}
