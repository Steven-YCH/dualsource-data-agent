import {
  FormEvent,
  ReactNode,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { AnswerBody } from "./components/AnswerBody";
import { SourcesBar } from "./components/SourcesBar";
import {
  CopyButton,
  DataTableCard,
  DebugPanel,
  EmptyAnswerState,
  HistoryLoadingPanel,
  ProgressTimeline,
  ResponseTiming,
  ToolCallTimeline,
  TypingIndicator,
  pickDisplayTables,
} from "./components/ChatElements";
import {
  getMe,
  getMessages,
  listSessions,
  patchSession,
  sendChatPreferStream,
  ProgressEvent,
  StreamTimings,
  UserInfo,
} from "./lib/api";
import { exportConversationXlsx } from "./lib/exportXlsx";
import { exportConversationPptx } from "./lib/exportPptx";
import {
  MoreHorizontal,
  PanelLeft,
  PanelLeftClose,
  Pencil,
  Plus,
  SquarePen,
  Trash2,
  FileSpreadsheet,
  Presentation,
  Loader2,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "sonner";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

const logo = "/logo.svg";
const DEBUG_KEY = "dualsource_debug";

type Session = {
  session_id: string;
  title: string;
  created_at?: string;
  updated_at?: string;
  status?: string;
};

/** ChatGPT-style date bucket for the sidebar session groups. */
function sessionBucket(s: Session): string {
  const t = Date.parse(s.updated_at || s.created_at || "");
  if (!Number.isFinite(t)) return "Recent";
  const now = new Date();
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const day = 86_400_000;
  if (t >= startToday) return "Today";
  if (t >= startToday - day) return "Yesterday";
  if (t >= startToday - 7 * day) return "Previous 7 days";
  if (t >= startToday - 30 * day) return "Previous 30 days";
  return "Older";
}

const STARTERS = [
  "How many passengers did Nova International (NVA) handle in 2025?",
  "Show NVA cargo trend for the last 6 months of 2025 as a chart",
  "Compare NVA passenger volumes with Singapore Changi using sources",
  "What is Pacific Rim Hub retail revenue by category in 2025?",
];

function dedupeSources(sources: any[]): any[] {
  const out: any[] = [];
  const seen = new Set<string>();
  for (const s of sources || []) {
    const key = `${s?.type || ""}|${(s?.url || "").toLowerCase()}|${(s?.label || "").toLowerCase()}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(s);
  }
  return out;
}

function isPptFollowup(text: string): boolean {
  const t = (text || "").toLowerCase();
  return (
    t.includes("powerpoint") ||
    t.includes("power point") ||
    t.includes("slide deck") ||
    /\bpptx?\b/.test(t) ||
    /\bslides?\b/.test(t)
  );
}

function isExcelOfferFollowup(text: string): boolean {
  const t = (text || "").toLowerCase();
  return /excel|xlsx|spreadsheet|workbook|worksheet/.test(t);
}


function sanitizeFollowups(items: string[], opts?: { blockExcelOffers?: boolean }): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const raw of items || []) {
    const f = (raw || "").trim();
    if (!f || isPptFollowup(f)) continue;
    if (opts?.blockExcelOffers && isExcelOfferFollowup(f)) continue;
    const key = f.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(f);
  }
  return out.slice(0, 5);
}

function fallbackFollowups(messages: any[]): string[] {
  const last = [...messages].reverse().find((m) => m?.role === "assistant");
  if (!last) return [];
  const hasTable = Array.isArray(last.tables_json) && last.tables_json.length > 0;
  const hasChart = Array.isArray(last.charts_json) && last.charts_json.length > 0;
  const out: string[] = [];
  if (hasTable && !hasChart) out.push("Show this as a monthly trend chart");
  out.push("Compare with the same period last year");
  out.push("Export this as Excel");
  return sanitizeFollowups(out).slice(0, 3);
}

/** Brand avatar for the signed-in user (initials on the the hub gradient). */
function UserAvatar({ user }: { user: UserInfo }) {
  return (
    <Avatar className="size-[30px]">
      <AvatarFallback className="bg-avatar-gradient text-[11px] font-bold text-white">
        {user.initials}
      </AvatarFallback>
    </Avatar>
  );
}

function isAbortError(e: unknown): boolean {
  return (
    (typeof DOMException !== "undefined" && e instanceof DOMException && e.name === "AbortError") ||
    (e instanceof Error && (/abort/i.test(e.name) || /abort/i.test(e.message)))
  );
}

/** True when answer markdown already includes a GFM table (avoid duplicate SQL card). */
function markdownHasTable(md: string): boolean {
  const text = md || "";
  return /^\s*\|(.|\n)+\|/m.test(text) && /^\s*\|?\s*:?-{3,}/m.test(text);
}

async function downloadExport(fileId: string, label: string) {
  try {
    const res = await fetch(`/api/files/${fileId}`);
    if (!res.ok) {
      let detail = "Export file is not available. Please try exporting again.";
      try {
        const data = await res.json();
        if (typeof data?.detail === "string") detail = data.detail;
      } catch {
        /* ignore */
      }
      toast.error("Export failed", {
        description:
          detail === "file not found"
            ? "Export file is not available. Please try exporting again."
            : detail,
      });
      return;
    }
    const blob = await res.blob();
    const cd = res.headers.get("content-disposition") || "";
    const match = /filename="?([^";]+)"?/i.exec(cd);
    const filename = match?.[1] || `${fileId}_${label.toLowerCase()}`;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toast.success(`${(label || "file").toUpperCase()} exported`, {
      description: filename,
    });
  } catch (e) {
    toast.error("Export failed", { description: String(e) });
  }
}

export default function App() {
  const [user, setUser] = useState<UserInfo | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("");
  const [statusKind, setStatusKind] = useState<"error" | "info">("error");
  const [busy, setBusy] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [latestFollowups, setLatestFollowups] = useState<string[]>([]);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [progressEvents, setProgressEvents] = useState<ProgressEvent[]>([]);
  const [streamingMarkdown, setStreamingMarkdown] = useState("");
  const [usedStream, setUsedStream] = useState(false);
  const [liveTimings, setLiveTimings] = useState<StreamTimings | null>(null);
  const [debugMode, setDebugMode] = useState(() => {
    try {
      return window.localStorage.getItem(DEBUG_KEY) === "1";
    } catch {
      return false;
    }
  });
  const bottomRef = useRef<HTMLDivElement | null>(null);
  // True while the user is at/near the transcript bottom. Auto-follow only runs in
  // that state — someone who scrolled up to re-read is never yanked around.
  const nearBottomRef = useRef(true);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const chatRef = useRef<HTMLDivElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const requestSeq = useRef(0);
  const historySeq = useRef(0);

  const hasMessages = messages.length > 0;
  const showWelcome = !hasMessages && !busy && !historyLoading;

  function showInfo(msg: string) {
    setStatusKind("info");
    setStatus(msg);
  }

  function showError(msg: string) {
    setStatusKind("error");
    setStatus(msg);
  }

  const [renaming, setRenaming] = useState<Session | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [deleting, setDeleting] = useState<Session | null>(null);

  /** Conversation → multi-sheet XLSX, fully client-side (spec: conversation-xlsx-export). */
  async function onExportXlsx() {
    if (!activeId || busy || exporting) return;
    setExporting(true);
    try {
      const active = sessions.find((s) => s.session_id === activeId);
      const result = await exportConversationXlsx({
        sessionId: activeId,
        sessionTitle: active?.title,
      });
      if (result.sheets === 0) {
        toast.info("Nothing to export", {
          description: "No tabular data in this conversation.",
        });
        return;
      }
      toast.success("XLSX exported", {
        description: `${result.sheets} sheet${result.sheets > 1 ? "s" : ""} · ${result.rows} rows · ${result.filename}`,
      });
    } catch (e) {
      toast.error("Export failed", {
        description: isAbortError(e)
          ? "Export was cancelled — please retry."
          : "Could not export this conversation — please retry.",
      });
    } finally {
      setExporting(false);
    }
  }

  /** Conversation → branded 16:9 PPTX deck, fully client-side
   *  (spec: conversation-ppt-export). Shares the `exporting` state with the
   *  XLSX button — either export disables both. */
  async function onExportPptx() {
    if (!activeId || busy || exporting) return;
    setExporting(true);
    try {
      const active = sessions.find((s) => s.session_id === activeId);
      const result = await exportConversationPptx({
        sessionId: activeId,
        sessionTitle: active?.title,
      });
      if (result.slides === 0) {
        toast.info("Nothing to export", {
          description: "No answers, charts, or tables in this conversation.",
        });
        return;
      }
      toast.success("PPTX exported", {
        description: `${result.slides} slides · ${result.charts} charts · ${result.tables} tables · ${result.filename}`,
      });
    } catch (e) {
      toast.error("Export failed", {
        description: isAbortError(e)
          ? "Export was cancelled — please retry."
          : "Could not export this conversation — please retry.",
      });
    } finally {
      setExporting(false);
    }
  }

  async function commitRename() {
    if (!renaming) return;
    const title = renameValue.trim();
    if (!title) return;
    try {
      await patchSession(renaming.session_id, { title });
      setRenaming(null);
      await refreshSessions();
    } catch (e) {
      showError(String(e));
    }
  }

  async function commitDelete() {
    if (!deleting) return;
    const id = deleting.session_id;
    try {
      await patchSession(id, { status: "deleted" });
      setDeleting(null);
      if (activeId === id) onNewChat();
      await refreshSessions();
    } catch (e) {
      showError(String(e));
    }
  }

  async function refreshSessions(selectId?: string | null) {
    const data = await listSessions();
    const seen = new Set<string>();
    const unique: Session[] = [];
    for (const s of data as Session[]) {
      if (!s?.session_id || seen.has(s.session_id)) continue;
      seen.add(s.session_id);
      unique.push(s);
    }
    setSessions(unique);
    if (selectId) setActiveId(selectId);
  }

  useEffect(() => {
    (async () => {
      const me = await getMe();
      setUser(me);
      await refreshSessions();
    })().catch((e) => showError(String(e)));
  }, []);

  async function openSession(sessionId: string) {
    if (busy) {
      abortInFlight("Switched chat — previous answer discarded.");
    }
    if (sessionId === activeId && !busy && !historyLoading) return;
    const seq = ++historySeq.current;
    setActiveId(sessionId);
    setHistoryLoading(true);
    // Land a loaded conversation on its latest message, not its first chart.
    nearBottomRef.current = true;
    setMessages([]);
    setLatestFollowups([]);
    setStatus("");
    try {
      const msgs = await getMessages(sessionId);
      if (seq !== historySeq.current) return;
      setMessages(msgs);
      setLatestFollowups(fallbackFollowups(msgs));
    } catch (e) {
      if (seq !== historySeq.current) return;
      showError(String(e));
    } finally {
      if (seq === historySeq.current) setHistoryLoading(false);
    }
  }

  // Auto-scroll policy — follow the conversation, never jump backwards:
  // - Only auto-scroll while the user is at/near the bottom (nearBottomRef);
  //   if they scrolled up mid-answer, leave them where they are.
  // - On completion, land on the bottom of the finished answer and STAY there.
  //   Never scroll back up to a chart after the text has painted: the backward
  //   jump (and the old querySelector grabbing the FIRST chart of the whole
  //   transcript, i.e. an earlier turn's) reads as a page refresh. Charts render
  //   in place above the tail and their skeleton slots hold the height, so
  //   pinning the bottom is stable across the completion paint.
  // - useLayoutEffect: the completion re-pin lands in the same frame as the
  //   final paint, so no intermediate shifted frame is ever visible.
  useLayoutEffect(() => {
    if (!nearBottomRef.current) return;
    bottomRef.current?.scrollIntoView({
      behavior: busy ? "smooth" : "auto",
      block: "end",
    });
  }, [messages, busy, status, latestFollowups, progressEvents, streamingMarkdown, liveTimings]);

  // Keep live Total ticking during long tool waits (SSE may be silent for many seconds).
  useEffect(() => {
    if (!busy || !liveTimings) return;
    const id = window.setInterval(() => {
      setLiveTimings((prev) =>
        prev ? { ...prev, totalMs: performance.now() - prev.t0 } : prev,
      );
    }, 250);
    return () => window.clearInterval(id);
  }, [busy, liveTimings?.t0]);

  const onChatScroll = useCallback(() => {
    const el = chatRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    setShowScrollBtn(distance > 200);
    // Same threshold as the FAB: follow only while near the bottom. Content
    // growth alone never fires onScroll (scrollTop is unchanged), so this only
    // flips when the user actually scrolls.
    nearBottomRef.current = distance <= 200;
  }, []);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  function abortInFlight(notice?: string) {
    const controller = abortRef.current;
    if (controller) {
      controller.abort();
      abortRef.current = null;
    }
    requestSeq.current += 1;
    setBusy(false);
    setProgressEvents([]);
    setStreamingMarkdown("");
    setUsedStream(false);
    setLiveTimings(null);
    if (notice) showInfo(notice);
  }

  function onStop() {
    abortInFlight("Stopped — previous answer discarded. You can ask again or start a new chat.");
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last?.role === "user" && String(last.message_id || "").startsWith("tmp-user-")) {
        return prev.slice(0, -1);
      }
      return prev;
    });
    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  function onNewChat() {
    const wasBusy = busy;
    abortInFlight(
      wasBusy
        ? "Started a new chat — previous answer discarded."
        : "",
    );
    historySeq.current += 1;
    setHistoryLoading(false);
    setActiveId(null);
    nearBottomRef.current = true;
    setMessages([]);
    if (!wasBusy) setStatus("");
    setInput("");
    setLatestFollowups([]);
    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  async function onSend(raw?: string) {
    const text = (raw ?? input).trim();
    if (!text || busy) return;
    const controller = new AbortController();
    abortRef.current = controller;
    const seq = ++requestSeq.current;
    setBusy(true);
    setStatus("");
    setInput("");
    setLatestFollowups([]);
    setProgressEvents([]);
    setStreamingMarkdown("");
    setUsedStream(false);
    const t0 = performance.now();
    const timings: StreamTimings = {
      t0,
      firstProgressMs: null,
      firstDeltaMs: null,
      totalMs: null,
      usedStream: false,
      fellBack: false,
    };
    setLiveTimings({ ...timings });
    const tempUser = {
      message_id: `tmp-user-${Date.now()}`,
      role: "user",
      content_markdown: text,
    };
    const tempAssistantId = `tmp-assistant-${Date.now()}`;
    let liveMarkdown = "";
    const liveProgress: ProgressEvent[] = [];
    // Charts emitted mid-stream (before final payload), buffered by index order.
    const liveCharts: any[] = [];
    setMessages((prev) => [...prev, tempUser]);
    try {
      const clientHistory =
        activeId && !historyLoading && messages.length > 0
          ? messages
              .filter(
                (m) =>
                  m.message_id !== tempUser.message_id &&
                  (m.role === "user" || m.role === "assistant"),
              )
              .slice(-12)
              .map((m) => {
                const turn: {
                  role: "user" | "assistant";
                  content: string;
                  tables?: any[];
                  sources?: any[];
                } = {
                  role: m.role as "user" | "assistant",
                  content: m.content_markdown || "",
                };
                // Pass prior structured artifacts so chart follow-ups can reuse them.
                if (m.role === "assistant") {
                  const tables = Array.isArray(m.tables_json) ? m.tables_json.slice(0, 3) : [];
                  const sources = Array.isArray(m.sources_json) ? m.sources_json.slice(0, 8) : [];
                  if (tables.length) {
                    turn.tables = tables.map((t: any) => ({
                      title: t?.title,
                      columns: Array.isArray(t?.columns) ? t.columns.slice(0, 20) : [],
                      rows: Array.isArray(t?.rows) ? t.rows.slice(0, 40) : [],
                    }));
                  }
                  if (sources.length) turn.sources = sources;
                }
                return turn;
              })
          : undefined;

      const result = await sendChatPreferStream(
        activeId,
        text,
        {
          onFallback: (reason) => {
            if (seq !== requestSeq.current) return;
            timings.fellBack = true;
            timings.usedStream = false;
            setUsedStream(false);
            setLiveTimings({ ...timings });
            showInfo(reason);
          },
          onProgress: (ev) => {
            if (seq !== requestSeq.current) return;
            if (timings.firstProgressMs == null) {
              timings.firstProgressMs = performance.now() - timings.t0;
            }
            if (!timings.fellBack) timings.usedStream = true;
            setUsedStream(!timings.fellBack);
            liveProgress.push(ev);
            setProgressEvents([...liveProgress]);
            setLiveTimings({ ...timings, totalMs: performance.now() - timings.t0 });
          },
          onChart: (ev) => {
            if (seq !== requestSeq.current) return;
            const idx = Number(ev.index);
            const chart =
              ev.chart && typeof ev.chart === "object"
                ? { ...ev.chart, index: Number.isFinite(idx) && idx >= 0 ? idx : liveCharts.length }
                : ev.chart;
            liveCharts.push(chart);
            if (!timings.fellBack) timings.usedStream = true;
            setUsedStream(!timings.fellBack);
            // If the streaming bubble already exists, refresh its charts so any
            // [[CHART:n]] markers already in the streamed text resolve now.
            setMessages((prev) => {
              const i = prev.findIndex((m) => m.message_id === tempAssistantId);
              if (i === -1) return prev;
              const next = [...prev];
              next[i] = { ...next[i], charts_json: [...liveCharts] };
              return next;
            });
          },
          onDelta: (_piece, accumulated) => {
            if (seq !== requestSeq.current) return;
            if (timings.firstDeltaMs == null) {
              timings.firstDeltaMs = performance.now() - timings.t0;
            }
            liveMarkdown = accumulated;
            if (!timings.fellBack) timings.usedStream = true;
            setUsedStream(!timings.fellBack);
            setStreamingMarkdown(accumulated);
            setLiveTimings({ ...timings, totalMs: performance.now() - timings.t0 });
            // Progressive answer bubble while tokens arrive.
            setMessages((prev) => {
              const withoutTempAssistant = prev.filter((m) => m.message_id !== tempAssistantId);
              const hasUser = withoutTempAssistant.some((m) => m.message_id === tempUser.message_id);
              const base = hasUser ? withoutTempAssistant : [...withoutTempAssistant, tempUser];
              return [
                ...base,
                {
                  message_id: tempAssistantId,
                  role: "assistant",
                  content_markdown: accumulated,
                  sources_json: [],
                  tables_json: [],
                  charts_json: [...liveCharts],
                  files_meta_json: [],
                  streaming: true,
                },
              ];
            });
          },
        },
        controller.signal,
        clientHistory,
      );

      if (seq !== requestSeq.current) return;
      timings.usedStream = Boolean(result.usedStream) && !result.fellBack;
      timings.fellBack = Boolean(result.fellBack);
      timings.totalMs = performance.now() - timings.t0;
      // JSON fallback may only expose progress_events on final — treat first as first update.
      if (timings.firstProgressMs == null && Array.isArray(result.response?.progress_events) && result.response!.progress_events!.length) {
        timings.firstProgressMs = timings.totalMs;
      }
      if (timings.firstDeltaMs == null && (result.response?.answer_markdown || liveMarkdown)) {
        timings.firstDeltaMs = timings.totalMs;
      }
      setUsedStream(timings.usedStream);
      setLiveTimings({ ...timings });
      if (result.session_id) setActiveId(result.session_id);

      // Final paint: tables / charts / followups / sources on complete.
      // Streamed chart events win if the final payload somehow omits them.
      const finalCharts =
        Array.isArray(result.response?.charts) && result.response!.charts!.length
          ? result.response!.charts!
          : liveCharts;
      const assistant = result.message || {
        message_id: tempAssistantId,
        role: "assistant",
        content_markdown:
          result.response?.answer_markdown || liveMarkdown || "",
        sources_json: result.response?.sources || [],
        tables_json: result.response?.tables || [],
        charts_json: finalCharts,
        files_meta_json: result.response?.files || [],
      };
      const finalMarkdown =
        assistant.content_markdown ||
        result.response?.answer_markdown ||
        liveMarkdown ||
        "";
      const painted = [
        { ...tempUser, message_id: `user-${assistant.message_id || Date.now()}` },
        {
          ...assistant,
          role: "assistant",
          content_markdown: finalMarkdown,
          sources_json: assistant.sources_json?.length
            ? assistant.sources_json
            : result.response?.sources || [],
          tables_json: assistant.tables_json?.length
            ? assistant.tables_json
            : result.response?.tables || [],
          // NOTE: `[]` is truthy in JS — an empty message.charts_json used to
          // silently override the streamed charts in finalCharts. Prefer the
          // first NON-EMPTY list so a chart that painted mid-stream survives
          // the final paint.
          charts_json: assistant.charts_json?.length
            ? assistant.charts_json
            : finalCharts,
          files_meta_json: assistant.files_meta_json?.length
            ? assistant.files_meta_json
            : result.response?.files || [],
          streaming: false,
          timings: { ...timings },
          fellBack: timings.fellBack,
          debug: result.response?.debug || null,
          // Prefer live SSE tool cards; fall back to final payload stages.
          progress_events:
            liveProgress.length > 0
              ? liveProgress
              : result.response?.progress_events || [],
        },
      ];
      setMessages((prev) => {
        const withoutTemps = prev.filter(
          (m) => m.message_id !== tempUser.message_id && m.message_id !== tempAssistantId,
        );
        return [...withoutTemps, ...painted];
      });
      setStreamingMarkdown("");
      const blockExcelOffers = /\b(did i ask|did i tell|why excel|why export|no excel|not excel|don'?t want excel|dont want excel)\b/i.test(text);
      const fromAgent = sanitizeFollowups(result.response?.followups || [], { blockExcelOffers });
      setLatestFollowups(
        fromAgent.length
          ? fromAgent
          : blockExcelOffers
            ? sanitizeFollowups(fallbackFollowups(painted), { blockExcelOffers: true })
            : fallbackFollowups(painted),
      );
      if (result.session_id) {
        void refreshSessions(result.session_id).catch(() => undefined);
      }
    } catch (e) {
      if (seq !== requestSeq.current || isAbortError(e)) return;
      timings.totalMs = performance.now() - timings.t0;
      setLiveTimings({ ...timings });
      const errText = String(e);
      showError(errText);
      setInput(text);
      // Always keep the user turn. Wiping temps with no token deltas used to snap
      // back to the landing/welcome screen after a late stream/tool failure.
      const failureMarkdown = liveMarkdown.trim()
        ? liveMarkdown
        : [
            "**Couldn't finish this answer.**",
            "",
            errText.replace(/^Error:\s*/i, ""),
            "",
            "Your question is still here — edit below and send again, or start a new chat.",
          ].join("\n");
      setMessages((prev) => {
        const withoutTemps = prev.filter(
          (m) => m.message_id !== tempUser.message_id && m.message_id !== tempAssistantId,
        );
        return [
          ...withoutTemps,
          { ...tempUser, message_id: `user-partial-${Date.now()}` },
          {
            message_id: `assistant-partial-${Date.now()}`,
            role: "assistant",
            content_markdown: failureMarkdown,
            sources_json: [],
            tables_json: [],
            // Keep charts that already rendered mid-stream — a late stream
            // failure (e.g. final-frame error) must not wipe them.
            charts_json: [...liveCharts],
            files_meta_json: [],
            streaming: false,
            timings: { ...timings },
            fellBack: timings.fellBack,
            partial: true,
            progress_events: liveProgress,
          },
        ];
      });
    } finally {
      if (seq === requestSeq.current) {
        abortRef.current = null;
        setBusy(false);
        setStreamingMarkdown("");
        // Keep last progress briefly then clear on next send; clear now for clean UI.
        setProgressEvents([]);
        // Keep timings on the painted message; clear live strip only.
        setLiveTimings(null);
        requestAnimationFrame(() => textareaRef.current?.focus());
      }
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void onSend();
  }

  const composer = useMemo(
    () => (
      <form className={`composer ${showWelcome ? "composer-center" : "composer-bottom"}`} onSubmit={onSubmit}>
        <div className="composer-shell">
          {!showWelcome && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className="composer-export"
                  onClick={() => void onExportXlsx()}
                  disabled={busy || exporting}
                  aria-label="Export all tables in this chat as XLSX"
                  data-testid="composer-export-xlsx"
                >
                  {exporting ? (
                    <Loader2 size={18} className="animate-spin" />
                  ) : (
                    <FileSpreadsheet size={18} />
                  )}
                </button>
              </TooltipTrigger>
              <TooltipContent side="top">
                Export all tables in this chat as XLSX
              </TooltipContent>
            </Tooltip>
          )}
          {!showWelcome && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className="composer-export"
                  onClick={() => void onExportPptx()}
                  disabled={busy || exporting}
                  aria-label="Export this chat as a presentation (PPTX)"
                  data-testid="composer-export-pptx"
                >
                  {exporting ? (
                    <Loader2 size={18} className="animate-spin" />
                  ) : (
                    <Presentation size={18} />
                  )}
                </button>
              </TooltipTrigger>
              <TooltipContent side="top">
                Export this chat as a presentation (PPTX)
              </TooltipContent>
            </Tooltip>
          )}
          <textarea
            ref={textareaRef}
            value={input}
            rows={1}
            placeholder="Ask about hub KPIs, cargo, routes, retail, OTP…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void onSend();
              }
            }}
            disabled={busy}
          />
          {busy ? (
            <button
              type="button"
              className="composer-stop"
              onClick={onStop}
              aria-label="Stop"
              title="Stop generating"
              data-testid="composer-stop"
            >
              ■
            </button>
          ) : input.trim() ? (
            <button type="submit" aria-label="Send" data-testid="composer-send">
              ↑
            </button>
          ) : null}
        </div>
        <div className="composer-hint">
          {busy
            ? "Generating… click ■ to stop, or New chat to discard"
            : "DualSource Agent · Internal DB + Online Sources"}
        </div>
      </form>
    ),
    [input, busy, showWelcome, exporting, activeId, sessions],
  );

  /** Shared account menu — sidebar bottom (expanded) + topbar (collapsed). */
  const userMenu = user && (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button type="button" className="user-chip" title={user.email}>
          <UserAvatar user={user} />
          <div className="user-meta">
            <div className="user-name">{user.display_name}</div>
            <div className="muted">{user.email}</div>
          </div>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-64">
        <DropdownMenuLabel>
          <div className="flex items-center gap-2.5">
            <UserAvatar user={user} />
            <div className="min-w-0">
              <div className="truncate text-sm font-medium">
                {user.display_name}
              </div>
              <div className="truncate text-xs text-muted-foreground">
                {user.email}
              </div>
            </div>
          </div>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={() => onNewChat()}>
          <Plus />
          New chat
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );

  return (
    <div className={`app-shell ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
      <aside className="sidebar" aria-hidden={!sidebarOpen}>
        <div className="sidebar-top">
          <button type="button" className="sidebar-brand brand-link" onClick={onNewChat} title="New chat">
            <div>
              <div className="sidebar-title">DualSource Agent</div>
              <div className="sidebar-sub">Internal DB + web research agent</div>
            </div>
          </button>
        </div>
        <button className="primary" onClick={onNewChat}>
          <SquarePen size={16} aria-hidden="true" />
          New chat
        </button>
        <ScrollArea className="min-h-0 flex-1">
          <div className="flex flex-col gap-0.5 px-0.5 pb-1">
            {sessions.length === 0 && <div className="session-empty">No chats yet</div>}
            {(() => {
              const out: ReactNode[] = [];
              let lastBucket = "";
              sessions.forEach((s, i) => {
                const bucket = sessionBucket(s);
                if (bucket !== lastBucket) {
                  out.push(
                    <div key={`sg-${i}`} className="session-group-label">
                      {bucket}
                    </div>,
                  );
                  lastBucket = bucket;
                }
                out.push(
                  <div
                    key={s.session_id}
                    className={`session-item ${activeId === s.session_id ? "active" : ""}`}
                  >
                    <button
                      type="button"
                      className="session-open"
                      onClick={() => void openSession(s.session_id)}
                      disabled={historyLoading && activeId === s.session_id}
                      title={s.title || "Untitled"}
                    >
                      <span className="session-title">{s.title || "Untitled"}</span>
                    </button>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <button type="button" className="session-more" aria-label="Chat options">
                          <MoreHorizontal size={16} aria-hidden="true" />
                        </button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="start" className="w-40">
                        <DropdownMenuItem
                          onSelect={() => {
                            setRenameValue(s.title || "");
                            setRenaming(s);
                          }}
                        >
                          <Pencil />
                          Rename
                        </DropdownMenuItem>
                        <DropdownMenuItem variant="destructive" onSelect={() => setDeleting(s)}>
                          <Trash2 />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>,
                );
              });
              return out;
            })()}
          </div>
        </ScrollArea>
        {user && <div className="sidebar-bottom">{userMenu}</div>}
      </aside>

      <section className="main">
        <header className="topbar">
          <div className="top-left">
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  className="ghost dark"
                  onClick={() => setSidebarOpen((v) => !v)}
                  aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
                >
                  {sidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeft size={18} />}
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                {sidebarOpen ? "Close sidebar" : "Open sidebar"}
              </TooltipContent>
            </Tooltip>
            <button type="button" className="brand-link topbar-brand" onClick={onNewChat} title="New chat">
              <img src={logo} alt="DualSource" className="topbar-logo" />
              <span>DualSource Agent</span>
            </button>
          </div>
          <div className="top-right">
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className={`debug-toggle ${debugMode ? "on" : ""}`}
                  aria-pressed={debugMode}
                  onClick={() => {
                    setDebugMode((v) => {
                      const next = !v;
                      try {
                        window.localStorage.setItem(DEBUG_KEY, next ? "1" : "0");
                      } catch {
                        /* ignore */
                      }
                      return next;
                    });
                  }}
                >
                  {debugMode ? "Debug ON" : "Debug"}
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                Show SQL / tool names while working (local debug)
              </TooltipContent>
            </Tooltip>
            {user && <div className="topbar-user">{userMenu}</div>}
          </div>
        </header>

        <div className={`stage ${showWelcome ? "welcome" : "chatting"}`}>
          {historyLoading ? (
            <HistoryLoadingPanel label="Loading chat" />
          ) : showWelcome ? (
            <div className="welcome-panel">
              <img src={logo} alt="DualSource" className="welcome-logo" />
              <h1>How can I help today?</h1>
              <p>Ask internal warehouse metrics, compare with public sources, or export Excel.</p>
              <div className="starter-grid">
                {STARTERS.map((q) => (
                  <button key={q} className="starter" onClick={() => void onSend(q)} disabled={busy}>
                    {q}
                  </button>
                ))}
              </div>
              {composer}
            </div>
          ) : (
            <>
              <ScrollArea
                className="min-h-0 overflow-hidden"
                viewportRef={chatRef}
                onViewportScroll={onChatScroll}
              >
                <div className="px-5 pb-2 pt-[22px]">
                {status && (
                  <div className={`status ${statusKind === "info" ? "status-info" : ""}`}>
                    <span>{status}</span>
                    <button className="status-dismiss" onClick={() => setStatus("")}>
                      ✕
                    </button>
                  </div>
                )}
                {messages.map((m) => {
                  const sources = dedupeSources(Array.isArray(m.sources_json) ? m.sources_json : []);
                  const files = Array.isArray(m.files_meta_json)
                    ? m.files_meta_json.filter(
                        (f: any) => f?.id && String(f?.type || "").toLowerCase() !== "pptx",
                      )
                    : [];
                  const charts = Array.isArray(m.charts_json) ? m.charts_json : [];
                  const hasChart = charts.length > 0;
                  const mdHasTable = markdownHasTable(String(m.content_markdown || ""));
                  // Tables default to open in chat; users can still collapse via
                  // the header toggle (previously auto-collapsed when a chart existed).
                  const collapseSqlTables = false;
                  const hideDuplicateSqlTables = mdHasTable && !hasChart && !debugMode;
                  const displayTables = pickDisplayTables(m.tables_json, hasChart);
                  const msgProgress = Array.isArray(m.progress_events) ? m.progress_events : [];
                  const hasAnswerText = Boolean(String(m.content_markdown || "").trim());
                  const hasExtras =
                    sources.length > 0 ||
                    displayTables.length > 0 ||
                    hasChart ||
                    files.length > 0;
                  return (
                    <div key={m.message_id} className={`row ${m.role}`}>
                      {/* `group` marker (style-free) lets the converted CopyButton
                          reveal itself via group-hover: utilities. */}
                      <div className={`bubble ${m.role} group`}>
                        {/* Working steps first (LibreChat / Databricks-style thought trail). */}
                        {m.role === "assistant" && !m.streaming && msgProgress.length > 0 && (
                          <ToolCallTimeline
                            busy={false}
                            events={msgProgress}
                            debug={debugMode}
                            persist
                          />
                        )}
                        {m.role === "assistant" ? (
                          <div className={m.streaming ? "answer-streaming" : undefined}>
                            {hasAnswerText ? (
                              <AnswerBody
                                markdown={m.content_markdown || ""}
                                charts={charts}
                                sources={sources}
                                streaming={Boolean(m.streaming)}
                              />
                            ) : !m.streaming && !hasExtras ? (
                              <EmptyAnswerState fellBack={Boolean(m.fellBack)} />
                            ) : !m.streaming && hasExtras ? (
                              <AnswerBody
                                markdown={m.content_markdown || ""}
                                charts={charts}
                                sources={sources}
                              />
                            ) : null}
                          </div>
                        ) : (
                          <div className="user-text">{m.content_markdown || ""}</div>
                        )}
                        {m.role === "assistant" && hasAnswerText && (
                          <CopyButton text={m.content_markdown} />
                        )}
                        {m.role === "assistant" && <SourcesBar sources={sources} />}
                        {m.role === "assistant" &&
                          !hideDuplicateSqlTables &&
                          displayTables.map((t: any, idx: number) => (
                              <DataTableCard
                                key={idx}
                                table={t}
                                defaultCollapsed={collapseSqlTables}
                              />
                            ))}
                        {m.role === "assistant" && files.length > 0 && (
                          <div className="download-row">
                            {files.map((f: any) => (
                              <button
                                key={f.id}
                                type="button"
                                className="download-btn"
                                onClick={() => void downloadExport(f.id, f.type || "file")}
                              >
                                Download {(f.type || "file").toUpperCase()}
                              </button>
                            ))}
                          </div>
                        )}
                        {m.role === "assistant" && !m.streaming && m.timings && (
                          <ResponseTiming timings={m.timings} debug={debugMode} />
                        )}
                        {m.role === "assistant" && !m.streaming && (
                          <DebugPanel debug={debugMode} payload={m.debug} />
                        )}
                      </div>
                    </div>
                  );
                })}
                {!busy &&
                  latestFollowups.length > 0 &&
                  messages.length > 0 &&
                  messages[messages.length - 1]?.role === "assistant" && (
                    <div className="followup-row">
                      {latestFollowups.map((f, idx) => (
                        <button
                          key={idx}
                          className="followup-chip"
                          onClick={() => void onSend(f)}
                          disabled={busy}
                        >
                          {f}
                        </button>
                      ))}
                    </div>
                  )}
                {busy && (
                  <>
                    <ProgressTimeline
                      events={progressEvents}
                      busy={busy}
                      timings={liveTimings}
                      debug={debugMode}
                    />
                    <ToolCallTimeline
                      busy={busy}
                      events={progressEvents}
                      debug={debugMode}
                      persist={false}
                    />
                    {/* Keep a visible assistant draft so the stage never looks blank mid-wait. */}
                    {!streamingMarkdown && (
                      <div className="row assistant" data-testid="draft-waiting">
                        <div className="bubble assistant draft-waiting">
                          <TypingIndicator
                            label={(() => {
                              const stages = progressEvents.map((e) => String(e.stage || "").toLowerCase());
                              const labels = progressEvents.map((e) => String(e.label || ""));
                              const last = progressEvents[progressEvents.length - 1];
                              if (stages.includes("web") || labels.some((l) => /online|web|public/i.test(l))) {
                                return "Checking online sources…";
                              }
                              if (String(last?.label || "").toLowerCase().includes("chart")) {
                                return "Preparing chart…";
                              }
                              if (stages.includes("synthesize") || /draft/i.test(String(last?.label || ""))) {
                                return "Drafting answer…";
                              }
                              if (stages.includes("genie") || labels.some((l) => /internal|sql|query|duckdb|zeap/i.test(l))) {
                                return "Retrieving internal figures…";
                              }
                              if (progressEvents.length) return "Working on your answer";
                              if (liveTimings?.fellBack) return "Preparing standard reply";
                              return "Preparing your answer";
                            })()}
                          />
                        </div>
                      </div>
                    )}
                  </>
                )}
                <div ref={bottomRef} />
                {showScrollBtn && (
                  <button
                    className="scroll-bottom-fab"
                    onClick={() => bottomRef.current?.scrollIntoView({ behavior: "smooth" })}
                    aria-label="Scroll to bottom"
                  >
                    ↓
                  </button>
                )}
                </div>
              </ScrollArea>
              {composer}
            </>
          )}
        </div>
      </section>

      <Dialog open={!!renaming} onOpenChange={(o) => !o && setRenaming(null)}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Rename chat</DialogTitle>
            <DialogDescription>Set a new title for this conversation.</DialogDescription>
          </DialogHeader>
          <input
            className="focus-visible:border-ring h-9 w-full rounded-md border border-input bg-white px-3 text-sm outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
            value={renameValue}
            autoFocus
            onChange={(e) => setRenameValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void commitRename();
            }}
          />
          <DialogFooter className="gap-2">
            <Button variant="outline" onClick={() => setRenaming(null)}>
              Cancel
            </Button>
            <Button disabled={!renameValue.trim()} onClick={() => void commitRename()}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={!!deleting} onOpenChange={(o) => !o && setDeleting(null)}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Delete chat?</DialogTitle>
            <DialogDescription>
              “{deleting?.title || "Untitled"}” will be removed from your chat history. This cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2">
            <Button variant="outline" onClick={() => setDeleting(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={() => void commitDelete()}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Toaster position="bottom-right" />
    </div>
  );
}
