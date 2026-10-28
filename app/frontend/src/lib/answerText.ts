/**
 * Shared answer-markdown helpers — normalization moved verbatim from
 * AnswerBody.tsx, plus the plain-text pipeline used by the PPT export
 * (spec: docs/design/conversation-ppt-export.md, "Text sanitization").
 * Import from both AnswerBody and the PPT export helper; do not copy-paste.
 */
import { domainFromUrl, SourceItem } from "@/components/SourcesBar";

export const CHART_TOKEN_RE = /\[\[CHART(?::(\d+))?\]\]/gi;

export function normalizeAnswerMarkdown(md: string): string {
  let s = md || "";
  // Fullwidth citation brackets from some models -> ASCII.
  s = s.replace(/【\s*Internal\s*】/gi, "[Internal]");
  s = s.replace(/【\s*(\d+)\s*】/g, "[[$1]]");
  s = s.replace(/\u3010\s*Internal\s*\u3011/gi, "[Internal]");
  s = s.replace(/\u3010\s*(\d+)\s*\u3011/g, "[[$1]]");
  // Turn citation tokens into markdown links so we can style them.
  s = s.replace(/\[\[(\d+)\]\]/g, "[[$1]](#cite-$1)");
  s = s.replace(/\[Internal\]/gi, "[Internal](#cite-internal)");
  // Common leak: leading/trailing orphan bold markers on a line.
  s = s.replace(/^\*\*\s+/gm, "");
  s = s.replace(/\s+\*\*$/gm, "");
  // Literal <br> leftovers from export prose.
  s = s.replace(/<br\s*\/?>/gi, "\n");
  // Legacy unnumbered chart marker → index 0.
  s = s.replace(/\[\[CHART\]\]/gi, "[[CHART:0]]");
  return s;
}

export function stripChartTokens(text: string): string {
  return text.replace(CHART_TOKEN_RE, "").trim();
}

/** Chart index referenced by each `[[CHART:n]]` marker, in text order. */
export function chartMarkerOrder(normalized: string): number[] {
  const order: number[] = [];
  const re = /\[\[CHART:(\d+)\]\]/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(normalized)) !== null) {
    const idx = Number(m[1]);
    if (!order.includes(idx)) order.push(idx);
  }
  return order;
}

/** In-chat `resolveChart` semantics: index match first, array position fallback. */
export function resolveChart(charts: any[], index: number): any | null {
  if (!charts.length) return null;
  const byIndex = charts.find((c) => Number(c?.index) === index);
  if (byIndex) return byIndex;
  return charts[index] || null;
}

// ---------------------------------------------------------------------------
// Plain-text pipeline for PPT export (review items 1, 7, 9)
// ---------------------------------------------------------------------------

/** A run of slide text; superscript runs are the footnote markers. */
export type TextSegment = { text: string; superscript?: boolean };

/** Numbered footnote attached to the same slide as its marker(s). */
export type PptFootnote = {
  num: number;
  key: string;
  label: string;
  url?: string | null;
};

export type PlainAnswer = {
  segments: TextSegment[];
  footnotes: PptFootnote[];
};

const PH_OPEN = "\u0000";
const PH_RE = /\u0000(\d+)\u0000/g;

/** Resolve a numbered cite ([[n]]) to its source, same as the chat renderer. */
function numberedSource(n: number, sources: SourceItem[]): SourceItem | null {
  const tag = `[${n}]`;
  return (
    sources.find((s) => (s?.cite_tag || "").includes(tag)) ||
    sources.filter((s) => s?.url)[n - 1] ||
    null
  );
}

/**
 * Convert answer markdown to PPT-ready plain text.
 *
 * Strips every citation / chart-token form, keeps link label text, flattens
 * markdown styling (bold / code / headings / images), and replaces citations
 * with numbered footnote markers (superscript runs). Footnotes are the union
 * of in-text markers and `sources` entries (sources with no inline marker are
 * appended after the cited ones).
 */
export function answerToPlainText(md: string, sources: SourceItem[]): PlainAnswer {
  const sourceList = Array.isArray(sources) ? sources : [];
  let s = stripChartTokens(normalizeAnswerMarkdown(md || ""));
  if (!s) return { segments: [], footnotes: [] };

  const footnotes: PptFootnote[] = [];
  const byKey = new Map<string, PptFootnote>();
  /** Identity: resolved URL wins, internal/Internal collapses to one, else marker. */
  const keyFor = (src: SourceItem | null, fallback: string): string => {
    if (src?.url) return `u:${src.url.replace(/\/$/, "")}`;
    if (
      src &&
      (String(src.type || "").includes("internal") ||
        String(src.label || "").toLowerCase().includes("zeap") ||
        String(src.label || "").toLowerCase().includes("internal"))
    ) {
      return "zeap";
    }
    return fallback;
  };
  const addFootnote = (key: string, label: string, url?: string | null): number => {
    const existing = byKey.get(key);
    if (existing) return existing.num;
    const fn: PptFootnote = { num: footnotes.length + 1, key, label, url: url || null };
    footnotes.push(fn);
    byKey.set(key, fn);
    return fn.num;
  };

  // Drop image syntax before link handling so ![alt](url) never matches links.
  s = s.replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1");

  // Citation links in order of occurrence -> footnote placeholders.
  // Group numbering: [Internal] has no group; (\d+) is group 1 (backref \1);
  // external links use groups 2 (label) and 3 (url).
  const citeRe =
    /\[Internal\]\(#cite-internal\)|\[\[(\d+)\]\]\(#cite-\1\)|\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g;
  s = s.replace(citeRe, (_full, num: string | undefined, label: string | undefined, url: string) => {
    let num2: number;
    if (num != null) {
      // Numbered cite [[n]].
      const src = numberedSource(Number(num), sourceList);
      const domain = src?.url ? domainFromUrl(src.url) : "";
      const key = keyFor(src, `n:${num}`);
      num2 = addFootnote(
        key,
        src ? `${domain || "source"} — ${src.label || `Source ${num}`}` : `Source ${num}`,
        src?.url || null,
      );
    } else if (label != null && url) {
      // External [Label](url) link.
      const matched = sourceList.find(
        (x) => x?.url && x.url.replace(/\/$/, "") === url.replace(/\/$/, ""),
      );
      const domain = domainFromUrl(url) || "source";
      const key = keyFor(matched || null, `u:${url.replace(/\/$/, "")}`);
      num2 = addFootnote(
        key,
        matched ? `${domain} — ${matched.label || label}` : `${domain} — ${label}`,
        url,
      );
    } else {
      // Internal internal cite.
      num2 = addFootnote("zeap", "Internal - client warehouse", null);
    }
    return `${PH_OPEN}${num2}${PH_OPEN}`;
  });

  // Flatten markdown styling (placeholders are NUL-delimited, untouched).
  s = s
    .replace(/```[\s\S]*?```/g, (m2) => m2.replace(/```\w*\n?|```/g, "")) // fenced code
    .replace(/^#{1,6}\s*/gm, "") // headings
    .replace(/\*\*(.+?)\*\*/gs, "$1") // bold
    .replace(/__(.+?)__/gs, "$1") // bold alt
    .replace(/`([^`]*)`/g, "$1") // inline code
    .replace(/~~(.+?)~~/gs, "$1") // strike
    .replace(/\*\[(.*?)\]\(#cite-\d+\)/g, "$1") // any leftover styled cites
    .replace(/\[[^\]]*\]\(#cite-[^)]*\)/g, "") // any leftover cite link (safety net)
    .replace(/^[-*+]\s+/gm, "- ") // bullets
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  // Union: sources with no inline marker are appended as footnotes.
  for (const src of sourceList) {
    if (!src) continue;
    if (src.url) {
      const key = `u:${src.url.replace(/\/$/, "")}`;
      if (!byKey.has(key)) {
        addFootnote(key, `${domainFromUrl(src.url) || "source"} — ${src.label || "source"}`, src.url);
      }
    } else if (
      String(src.type || "").includes("internal") ||
      String(src.label || "").toLowerCase().includes("zeap") ||
      String(src.label || "").toLowerCase().includes("internal")
    ) {
      if (!byKey.has("zeap")) addFootnote("zeap", "Internal - client warehouse", null);
    }
  }

  // Split into segments; placeholder numbers become superscript runs.
  const segments: TextSegment[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  PH_RE.lastIndex = 0;
  while ((m = PH_RE.exec(s)) !== null) {
    if (m.index > last) segments.push({ text: s.slice(last, m.index) });
    segments.push({ text: m[1], superscript: true });
    last = m.index + m[0].length;
  }
  if (last < s.length) segments.push({ text: s.slice(last) });

  return { segments, footnotes };
}
