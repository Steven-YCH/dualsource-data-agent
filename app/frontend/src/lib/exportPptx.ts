import * as echarts from "echarts";
import { getMessages } from "./api";
import { SourceItem } from "@/components/SourcesBar";
import { polishTitle } from "@/components/ChatElements";
import {
  computeCanvasHeight,
  isHorizontalBar,
  resolveEchartsOption,
  scrubNulls,
} from "./chartOption";
import {
  answerToPlainText,
  chartMarkerOrder,
  normalizeAnswerMarkdown,
  resolveChart,
  PptFootnote,
  TextSegment,
} from "./answerText";
import { ExportTable, saveBlob, slugify } from "./exportXlsx";
import {
  CONTENT_W,
  MASTER_CONTENT,
  MASTER_COVER,
  PPTX_THEME as T,
} from "./pptxTheme";

/**
 * Conversation → branded 16:9 PPTX deck, fully client-side (pptxgenjs).
 * Spec: docs/design/conversation-ppt-export.md (incl. "Design system v2").
 *
 * Cover → agenda → one slide per Q/A turn (answer card, chart caption + image,
 * zebra native table, numbered source footnotes; overflow to appendix slides)
 * → closing sources slide. All styling comes from lib/pptxTheme.ts tokens and
 * two slide masters (defineZeaMasters); chart images are captured offscreen
 * with a PPT-tuned ECharts theme (registerZeaPptTheme) via the SAME option
 * pipeline as the in-chat ChartCard — minus the toolbox and the in-chart title
 * (the title moves to a styled slide caption).
 */

export type PptExportResult = {
  filename: string;
  slides: number;
  charts: number;
  tables: number;
};

type Turn = {
  index: number;
  question: string;
  answer: string;
  charts: any[];
  tables: ExportTable[];
  sources: SourceItem[];
};

const NUM_CELL_RE = /^[+-]?[\d,.]+%?$/;

/* ---------------------------------------------------------------- helpers */

/** Assistant turns, chronological, each paired with its preceding question. */
export function collectPptTurns(messages: unknown): Turn[] {
  const list = Array.isArray(messages) ? messages : [];
  const turns: Turn[] = [];
  let question = "";
  let index = 0;
  for (const m of list) {
    const msg = (m || {}) as Record<string, unknown>;
    const role = String(msg.role || "");
    const content = String(msg.content_markdown ?? msg.content ?? "");
    if (role === "user") {
      question = content;
      continue;
    }
    if (role !== "assistant") continue;
    const charts = Array.isArray(msg.charts_json) ? (msg.charts_json as any[]) : [];
    const full = Array.isArray(msg.export_tables_json)
      ? (msg.export_tables_json as ExportTable[])
      : null;
    const display = Array.isArray(msg.tables_json) ? (msg.tables_json as ExportTable[]) : null;
    const tables = (full?.length ? full : display || []).filter(
      (t) =>
        t &&
        Array.isArray((t as any).columns) &&
        (t as any).columns.length &&
        Array.isArray((t as any).rows),
    );
    const sources = Array.isArray(msg.sources_json) ? (msg.sources_json as SourceItem[]) : [];
    const hasText = content.trim().length > 0;
    if (!hasText && !charts.length && !tables.length) continue;
    turns.push({
      index: ++index,
      question: (question || content.slice(0, 60) || `Turn ${index}`).trim(),
      answer: content,
      charts,
      tables,
      sources,
    });
  }
  return turns;
}

/**
 * Chart order for slides: `[[CHART:n]]` marker order first (resolveChart
 * semantics), then unused charts appended in charts_json order — same
 * precedence as the chat UI's "append unused charts" rule.
 */
function orderedCharts(charts: any[], answerMd: string): any[] {
  const markerOrder = chartMarkerOrder(normalizeAnswerMarkdown(answerMd));
  const out: any[] = [];
  const used = new Set<unknown>();
  for (const idx of markerOrder) {
    const c = resolveChart(charts, idx);
    if (c && !used.has(c)) {
      out.push(c);
      used.add(c);
    }
  }
  charts.forEach((c, i) => {
    const resolved = resolveChart(charts, i);
    if (resolved && !used.has(resolved)) {
      out.push(resolved);
      used.add(resolved);
    } else if (!resolved && c && !used.has(c)) {
      out.push(c);
      used.add(c);
    }
  });
  return out;
}

/* ------------------------------------------- Level 2: PPT-tuned chart theme */

let zeaThemeRegistered = false;
/** ECharts theme for slide capture: brand palette + slide-sized fonts.
 *  In-chat charts are styled for a ~900px chat column; slides are projected. */
export function registerZeaPptTheme(): void {
  if (zeaThemeRegistered) return;
  zeaThemeRegistered = true;
  echarts.registerTheme("zea-ppt", {
    color: [
      "#0A3D73", // brand
      "#2E7CC1", // brand light
      "#1C8C7E", // accent teal
      "#E8A33D", // amber
      "#7C5CBF", // violet
      "#C0576B", // rose
      "#3AA6B9", // cyan
    ],
    backgroundColor: T.capture.background,
    textStyle: { fontFamily: "Microsoft YaHei, PingFang SC, Arial, sans-serif" },
    legend: { textStyle: { fontSize: 16, color: `#${T.colors.text}` } },
    categoryAxis: {
      axisLabel: { fontSize: 15, color: "#475569" },
      axisLine: { lineStyle: { color: `#${T.colors.line}` } },
      axisTick: { lineStyle: { color: `#${T.colors.line}` } },
    },
    valueAxis: {
      axisLabel: { fontSize: 15, color: "#475569" },
      splitLine: { lineStyle: { color: "#E2E8F0" } },
    },
  });
}

/** ChartCard's option assembly minus toolbox and in-chart title (caption moves
 *  to a styled slide text element). Returns the polished title for the caption. */
async function captureChartPng(
  chart: any,
): Promise<{ dataUrl: string; h: number; w: number; title: string } | null> {
  const optionRaw = resolveEchartsOption(chart);
  if (!optionRaw) return null;

  const optionTitle =
    optionRaw.title && typeof optionRaw.title === "object"
      ? (optionRaw.title as { text?: unknown }).text
      : undefined;
  const rawTitle = chart?.title || chart?.plan?.title || chart?.chart_plan?.title || optionTitle;
  const polished = polishTitle(typeof rawTitle === "string" ? rawTitle : undefined);
  const title = !rawTitle || polished === "Query result" ? "Chart" : polished;
  const horizontal = isHorizontalBar(optionRaw, chart);
  const height = computeCanvasHeight(chart, optionRaw, horizontal);

  const option = scrubNulls({ ...optionRaw, animation: false });
  delete (option as any).title; // title becomes a slide caption

  registerZeaPptTheme();

  // Hidden offscreen container (not display:none — echarts needs layout).
  const el = document.createElement("div");
  el.style.cssText = `position:fixed;left:-10000px;top:0;width:${T.capture.width}px;`;
  document.body.appendChild(el);
  const inst = echarts.init(el, "zea-ppt", {
    renderer: "canvas",
    width: T.capture.width,
    height,
  });
  try {
    await new Promise<void>((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        resolve();
      };
      inst.on("finished", finish);
      setTimeout(finish, 600); // fallback if "finished" never fires
      inst.setOption(option);
    });
    const dataUrl = inst.getDataURL({
      type: "png",
      pixelRatio: T.capture.pixelRatio,
      backgroundColor: T.capture.background,
    });
    return { dataUrl, w: T.capture.width, h: height, title };
  } finally {
    inst.dispose();
    el.remove();
  }
}

/** Cap answer segments: ~8 bullet lines / 500 chars, ellipsis on cut. */
function capSegments(segments: TextSegment[], maxChars = 500, maxLines = 8): TextSegment[] {
  const out: TextSegment[] = [];
  let chars = 0;
  let lines = 1;
  for (const seg of segments) {
    if (!seg.text) continue;
    const nl = (seg.text.match(/\n/g) || []).length;
    if (chars + seg.text.length <= maxChars && lines + nl <= maxLines) {
      out.push(seg);
      chars += seg.text.length;
      lines += nl;
      continue;
    }
    const allowChars = maxChars - chars;
    const allowLines = maxLines - lines;
    if (allowChars <= 0 || allowLines <= 0) {
      if (out.length) out.push({ text: "…" });
      break;
    }
    let cut = Math.min(seg.text.length, allowChars);
    const segLines = seg.text.split("\n");
    if (segLines.length > allowLines) {
      cut = Math.min(cut, segLines.slice(0, allowLines).join("\n").length);
    }
    out.push({ text: `${seg.text.slice(0, cut).trimEnd()}…`, superscript: seg.superscript });
    break;
  }
  return out.length ? out : segments.slice(0, 1);
}

function truncateCell(text: string, maxChars: number): string {
  const s = String(text ?? "");
  return s.length > maxChars ? `${s.slice(0, Math.max(1, maxChars - 1))}…` : s;
}

/** Column is numeric if ≥80% of sampled cells parse as numbers. */
function isNumericColumn(rows: unknown[], colIdx: number, key: string): boolean {
  let num = 0;
  let total = 0;
  for (let r = 0; r < Math.min(rows.length, 20); r++) {
    const row = rows[r];
    const cell = Array.isArray(row) ? row[colIdx] : (row as any)?.[key];
    const s = String(cell ?? "").trim();
    if (!s) continue;
    total += 1;
    if (NUM_CELL_RE.test(s) && Number.isFinite(Number(s.replace(/,/g, "").replace("%", "")))) {
      num += 1;
    }
  }
  return total > 0 && num / total >= 0.8;
}

/* ------------------------------------------------------------ deck build */

export async function exportConversationPptx(params: {
  sessionId: string;
  sessionTitle?: string;
}): Promise<PptExportResult> {
  // Fresh fetch — in-memory message state does not retain the export blobs.
  const messages = await getMessages(params.sessionId);
  const turns = collectPptTurns(messages);
  if (!turns.length) return { filename: "", slides: 0, charts: 0, tables: 0 };

  // Lazy-load so pptxgenjs stays out of the initial bundle.
  const PptxGenJS = (await import("pptxgenjs")).default;
  const pptx = new PptxGenJS();
  pptx.defineLayout({ name: "WIDE", width: T.geometry.slideW, height: T.geometry.slideH });
  pptx.layout = "WIDE";
  pptx.author = "DualSource Agent";
  pptx.company = "Airport Authority Hong Kong";
  pptx.title = params.sessionTitle || "DualSource Conversation";
  pptx.theme = { headFontFace: T.fonts.head, bodyFontFace: T.fonts.body };

  const date = new Date().toISOString().slice(0, 10);
  const shortTitle =
    (params.sessionTitle || "Conversation").length > 60
      ? `${params.sessionTitle!.slice(0, 59)}…`
      : params.sessionTitle || "Conversation";

  // ---- Slide masters (Level 1 design system) ----
  pptx.defineSlideMaster({
    title: MASTER_COVER,
    background: { color: "FFFFFF" },
    objects: [
      { rect: { x: 0, y: 0, w: T.geometry.slideW, h: T.geometry.accentH, fill: { color: T.colors.brand } } },
      { rect: { x: 0, y: T.geometry.slideH - T.geometry.accentH, w: T.geometry.slideW, h: T.geometry.accentH, fill: { color: T.colors.brand } } },
    ],
  });
  pptx.defineSlideMaster({
    title: MASTER_CONTENT,
    background: { color: "FFFFFF" },
    objects: [
      { rect: { x: 0, y: 0, w: T.geometry.slideW, h: T.geometry.accentH, fill: { color: T.colors.brand } } },
      // Footer rule
      {
        line: {
          x: T.geometry.marginX,
          y: T.geometry.footerY - 0.06,
          w: CONTENT_W,
          h: 0,
          line: { color: T.colors.line, width: 0.75 },
        },
      },
    ],
    slideNumber: {
      x: T.geometry.slideW - 0.85,
      y: T.geometry.footerY,
      w: 0.6,
      h: 0.25,
      fontFace: T.fonts.body,
      fontSize: T.type.footer,
      color: T.colors.muted,
    },
  });

  const addFooter = (slide: any) => {
    slide.addText(`Internal · ${shortTitle} · ${date}`, {
      fontFace: T.fonts.body,
      fontSize: T.type.footer,
      color: T.colors.muted,
      x: T.geometry.marginX,
      y: T.geometry.footerY,
      w: 8,
      h: 0.25,
      valign: "middle",
    });
  };

  const titleOpts = {
    fontFace: T.fonts.head,
    color: `#${T.colors.brand}`,
    bold: true,
    fontSize: T.type.slideTitle,
    x: T.geometry.marginX,
    y: 0.24,
    w: CONTENT_W,
    h: 0.55,
    valign: "top" as const,
  };

  const addFootnoteStrip = (slide: any, footnotes: PptFootnote[]) => {
    if (!footnotes.length) return;
    slide.addText(footnotes.map((f) => `${f.num}. ${f.label}`).join("    "), {
      fontFace: T.fonts.body,
      fontSize: T.type.footnote,
      color: T.colors.muted,
      x: T.geometry.marginX,
      y: T.geometry.footnotesY,
      w: CONTENT_W,
      h: 0.45,
      valign: "top",
    });
  };

  /** Centered chart image + caption; returns the y below the image. */
  const addChartBlock = (
    slide: any,
    png: { dataUrl: string; w: number; h: number; title: string },
    yTop: number,
    maxH: number,
    maxW: number,
  ) => {
    slide.addText(png.title, {
      fontFace: T.fonts.head,
      fontSize: T.type.caption,
      bold: true,
      color: `#${T.colors.brand}`,
      x: T.geometry.marginX,
      y: yTop,
      w: CONTENT_W,
      h: 0.3,
      align: "center",
      valign: "top",
    });
    let h = maxH;
    let w = (h * png.w) / png.h;
    if (w > maxW) {
      w = maxW;
      h = (w * png.h) / png.w;
    }
    slide.addImage({ data: png.dataUrl, x: (T.geometry.slideW - w) / 2, y: yTop + 0.34, w, h });
    return yTop + 0.34 + h;
  };

  /** Zebra, numeric right-aligned, horizontal-hairline native table. */
  const addTable = (slide: any, table: ExportTable, yTop: number, yBottom: number): void => {
    const columns = ((table as any).columns || []).map((c: unknown) => String(c ?? ""));
    const allRows = (table as any).rows || [];
    if (!columns.length || !allRows.length) return;

    const droppedCols = Math.max(0, columns.length - 8);
    const cols: string[] = columns.slice(0, 8);

    // Proportional colW from header/content weight (clamped), min 0.5".
    const weight = cols.map((c: string, i: number) => {
      let max = c.length;
      for (let r = 0; r < Math.min(allRows.length, 40); r++) {
        const row = allRows[r];
        const cell = Array.isArray(row) ? row[i] : (row as any)?.[cols[i]];
        max = Math.max(max, String(cell ?? "").length);
      }
      return Math.min(40, Math.max(4, max));
    });
    const totalWeight = weight.reduce((a: number, b: number) => a + b, 0);
    const colW = weight.map((wt: number) => Math.max(0.5, (CONTENT_W * wt) / totalWeight));
    const colWTotal = colW.reduce((a: number, b: number) => a + b, 0);
    const cellMax = colW.map((w: number) => Math.max(6, Math.floor(w * 11)));

    const rowCap = Math.min(15, Math.max(1, Math.floor((yBottom - yTop) / 0.22) - 1));
    const shownRows = allRows.slice(0, rowCap);
    const droppedRows = allRows.length - shownRows.length;

    const numericCols = cols.map((_, i) => isNumericColumn(allRows, i, cols[i]));

    const headerRow = cols.map((c, i) => ({
      text: truncateCell(c, cellMax[i]),
      options: {
        bold: true,
        fontSize: T.type.tableHead,
        fontFace: T.fonts.head,
        color: T.colors.brand,
        fill: { color: T.colors.headFill },
        align: "right" as const,
        // Strong rule under the header (only horizontal borders on the table).
        border: [
          { type: "none" },
          { type: "none" },
          { type: "solid", pt: 1, color: T.colors.brand },
          { type: "none" },
        ],
      },
    }));

    const dataRows = shownRows.map((row: unknown, r: number) => {
      const cells: unknown[] = Array.isArray(row)
        ? (row as unknown[])
        : cols.map((c: string) => (row as any)?.[c]);
      return cells.map((cell: unknown, i: number) => ({
        text: truncateCell(String(cell ?? ""), cellMax[i]),
        options: {
          fontSize: T.type.tableBody,
          fontFace: T.fonts.body,
          color: T.colors.text,
          ...(numericCols[i] ? { align: "right" as const } : {}),
          ...(r % 2 === 1 ? { fill: { color: T.colors.zebra } } : {}),
        },
      }));
    });

    const rows = [headerRow, ...dataRows];
    if (droppedRows > 0 || droppedCols > 0) {
      const notes: string[] = [];
      if (droppedRows > 0) notes.push(`… ${droppedRows} more rows`);
      if (droppedCols > 0) notes.push(`… +${droppedCols} more columns`);
      rows.push([
        {
          text: notes.join("   ·   "),
          options: {
            fontSize: 8.5,
            italic: true,
            fontFace: T.fonts.body,
            color: T.colors.muted,
            colspan: cols.length,
          },
        },
      ]);
    }

    slide.addTable(rows, {
      x: T.geometry.marginX,
      y: yTop,
      w: colWTotal,
      colW,
      autoPage: false,
      // Horizontal hairlines only.
      border: [
        { type: "none" },
        { type: "none" },
        { type: "solid", pt: 0.5, color: T.colors.line },
        { type: "none" },
      ],
      margin: 0.04,
      valign: "middle",
    });
  };

  /* ---- Slide 1: cover ---- */
  {
    const slide = pptx.addSlide({ masterName: MASTER_COVER });
    slide.addText("Internal · DualSource Agent", {
      fontFace: T.fonts.head,
      fontSize: T.type.coverSubtitle,
      color: T.colors.muted,
      x: T.geometry.marginX,
      y: 2.35,
      w: CONTENT_W,
      h: 0.5,
      align: "center",
    });
    slide.addText(params.sessionTitle || "Conversation Export", {
      fontFace: T.fonts.head,
      fontSize: T.type.coverTitle,
      bold: true,
      color: T.colors.brand,
      x: 0.8,
      y: 2.9,
      w: T.geometry.slideW - 1.6,
      h: 1.3,
      align: "center",
      valign: "top",
    });
    slide.addText(`Generated by DualSource Agent · ${date}`, {
      fontFace: T.fonts.body,
      fontSize: 12,
      color: T.colors.muted,
      x: T.geometry.marginX,
      y: 4.35,
      w: CONTENT_W,
      h: 0.4,
      align: "center",
    });
  }

  let chartCount = 0;
  let tableCount = 0;
  let slideCount = 1;

  /* ---- Agenda slide (only when there's more than one turn) ---- */
  if (turns.length > 1) {
    const slide = pptx.addSlide({ masterName: MASTER_CONTENT });
    slideCount += 1;
    slide.addText("Agenda", titleOpts);
    slide.addText(
      turns
        .map((t) => {
          const q = t.question.length > 90 ? `${t.question.slice(0, 89)}…` : t.question;
          return [
            { text: `${t.index}.  `, options: { bold: true, color: T.colors.brandLight } },
            { text: q, options: {} },
            { text: "", options: { breakLine: true } },
          ];
        })
        .flat(),
      {
        fontFace: T.fonts.body,
        fontSize: T.type.agendaItem,
        color: T.colors.text,
        x: T.geometry.marginX,
        y: 1.15,
        w: CONTENT_W,
        h: 5.2,
        valign: "top",
        lineSpacingMultiple: 1.4,
      },
    );
    addFooter(slide);
  }

  for (const turn of turns) {
    const plain = answerToPlainText(turn.answer, turn.sources);
    const hasText = plain.segments.some((s) => s.text.trim());
    const charts = orderedCharts(turn.charts, turn.answer);

    // Sequential captures to bound memory.
    const pngs: { dataUrl: string; w: number; h: number; title: string }[] = [];
    for (const c of charts) {
      const png = await captureChartPng(c);
      if (png) pngs.push(png);
    }

    const primaryChart = pngs[0] || null;
    const extraCharts = pngs.slice(1);
    const primaryTable = turn.tables[0] || null;
    const extraTables = turn.tables.slice(1);

    if (!hasText && !primaryChart && !primaryTable && !extraCharts.length && !extraTables.length) {
      continue;
    }

    const slide = pptx.addSlide({ masterName: MASTER_CONTENT });
    slideCount += 1;
    slide.addText(
      turn.question.length > 90 ? `${turn.question.slice(0, 89)}…` : turn.question,
      titleOpts,
    );

    let yCursor = 0.95;
    if (hasText) {
      const segs = capSegments(plain.segments);
      const lineCount = segs.reduce((n, s) => n + (s.text.match(/\n/g) || []).length + 1, 0);
      const textH = Math.min(1.35, 0.3 + lineCount * 0.22);
      // Answer card: soft background so text reads as a distinct zone.
      slide.addText(
        segs.map((s) => ({ text: s.text, options: { superscript: !!s.superscript } })),
        {
          fontFace: T.fonts.body,
          fontSize: T.type.answer,
          color: T.colors.text,
          x: T.geometry.marginX,
          y: yCursor,
          w: CONTENT_W,
          h: textH + 0.14,
          valign: "top",
          lineSpacingMultiple: 1.15,
          fill: { color: T.colors.cardFill },
          margin: 0.08,
        },
      );
      yCursor += textH + 0.26;
    }

    const hasBoth = !!primaryChart && !!primaryTable;
    if (primaryChart) {
      const maxH = hasBoth ? 2.6 : Math.max(2.0, T.geometry.contentBottom - yCursor);
      const maxW = hasBoth ? 8.2 : 11.2;
      yCursor = addChartBlock(slide, primaryChart, yCursor, maxH, maxW) + 0.15;
      chartCount += 1;
    }
    if (primaryTable) {
      addTable(slide, primaryTable, yCursor, T.geometry.contentBottom);
      tableCount += 1;
    }
    addFootnoteStrip(slide, plain.footnotes);
    addFooter(slide);

    // Appendix slides — extra charts/tables are never silently dropped.
    extraCharts.forEach((png, i) => {
      const s = pptx.addSlide({ masterName: MASTER_CONTENT });
      slideCount += 1;
      s.addText(`Turn ${turn.index} — Chart ${i + 2} (cont.)`, titleOpts);
      addChartBlock(s, png, 1.05, T.geometry.contentBottom - 1.4, 11.2);
      chartCount += 1;
      addFooter(s);
    });
    extraTables.forEach((t, i) => {
      const s = pptx.addSlide({ masterName: MASTER_CONTENT });
      slideCount += 1;
      s.addText(`Turn ${turn.index} — Table ${i + 2} (cont.)`, titleOpts);
      addTable(s, t, 1.05, T.geometry.contentBottom);
      tableCount += 1;
      addFooter(s);
    });
  }

  /* ---- Closing slide: sources ---- */
  {
    const allFootnotes: PptFootnote[] = [];
    const seen = new Set<string>();
    for (const turn of turns) {
      const { footnotes } = answerToPlainText(turn.answer, turn.sources);
      for (const f of footnotes) {
        if (!seen.has(f.key)) {
          seen.add(f.key);
          allFootnotes.push(f);
        }
      }
    }
    if (allFootnotes.length) {
      const slide = pptx.addSlide({ masterName: MASTER_CONTENT });
      slideCount += 1;
      slide.addText("Sources", { ...titleOpts, fontSize: 20 });
      const lines = allFootnotes.map((f) => [
        { text: `${f.num}. `, options: { bold: true, color: T.colors.brand } },
        { text: `${f.label}${f.url ? `  (${f.url})` : ""}`, options: {} },
        { text: "", options: { breakLine: true } },
      ]);
      slide.addText(lines.flat(), {
        fontFace: T.fonts.body,
        fontSize: 11,
        color: T.colors.text,
        x: T.geometry.marginX,
        y: 1.1,
        w: CONTENT_W,
        h: 5.4,
        valign: "top",
        lineSpacingMultiple: 1.3,
      });
      addFooter(slide);
    }
  }

  const filename = `dualsource-${slugify(params.sessionTitle)}-${date}.pptx`;
  const blob = (await pptx.write({ outputType: "blob" })) as Blob;
  saveBlob(
    new Blob([blob], {
      type: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }),
    filename,
  );
  return { filename, slides: slideCount, charts: chartCount, tables: tableCount };
}
