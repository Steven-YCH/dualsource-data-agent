import { getMessages } from "./api";

/**
 * Conversation → multi-sheet XLSX export, fully client-side (exceljs).
 * Data-only sheets: bold header row + data rows, no cover page.
 *
 * Row fidelity matches the backend Excel path (`_pick_export_context`):
 * `export_tables_json` (full ≤500 rows) preferred over `tables_json` (≤20 display).
 */

export type ExportTable = {
  title?: string;
  columns?: unknown[];
  rows?: unknown[];
  [key: string]: unknown;
};

export type ExportResult = {
  filename: string;
  sheets: number;
  rows: number;
};

const XLSX_MIME =
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

/** Excel-forbidden sheet chars: : \ / ? * [ ] — names cap at 31 chars. */
function sanitizeSheetName(raw: string): string {
  const cleaned = String(raw || "")
    .replace(/[:\\/?*[\]]/g, " ")
    .replace(/['"<>]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return (cleaned || "Table").slice(0, 31);
}

function uniqueSheetName(base: string, used: Set<string>): string {
  let name = base;
  let n = 2;
  while (used.has(name.toLowerCase())) {
    const suffix = ` (${n})`;
    name = `${base.slice(0, 31 - suffix.length)}${suffix}`;
    n += 1;
  }
  used.add(name.toLowerCase());
  return name;
}

export function slugify(raw: string | null | undefined): string {
  return (
    String(raw || "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 40) || "chat"
  );
}

/** Full-fidelity tables first, then display tables. Assistant turns only, in order. */
export function collectExportTables(messages: unknown): ExportTable[] {
  const out: ExportTable[] = [];
  const list = Array.isArray(messages) ? messages : [];
  for (const m of list) {
    const msg = m as Record<string, unknown>;
    if (msg?.role !== "assistant") continue;
    const full = Array.isArray(msg.export_tables_json)
      ? (msg.export_tables_json as ExportTable[])
      : null;
    const display = Array.isArray(msg.tables_json)
      ? (msg.tables_json as ExportTable[])
      : null;
    const tables = full?.length ? full : (display || []);
    for (const t of tables) {
      if (
        t &&
        Array.isArray(t.columns) &&
        (t.columns as unknown[]).length &&
        Array.isArray(t.rows) &&
        (t.rows as unknown[]).length
      ) {
        out.push(t);
      }
    }
  }
  return out;
}

/** Normalize a cell for exceljs: primitives/Date pass through, objects stringify. */
function cellValue(v: unknown): string | number | Date {
  if (v === null || v === undefined) return "";
  if (typeof v === "number" || typeof v === "string") return v;
  if (v instanceof Date) return v;
  return String(v);
}

/** Rows may be arrays or column-keyed objects (mirrors `normalizeTableRows`). */
function rowCells(row: unknown, columns: string[]): unknown[] {
  if (Array.isArray(row)) return row;
  if (row && typeof row === "object") {
    const rec = row as Record<string, unknown>;
    if (columns.length) return columns.map((c) => rec[c]);
    return Object.values(rec);
  }
  return [row];
}

export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function exportConversationXlsx(params: {
  sessionId: string;
  sessionTitle?: string;
}): Promise<ExportResult> {
  // Fresh fetch — in-memory message state does not retain export_tables_json.
  const messages = await getMessages(params.sessionId);
  const tables = collectExportTables(messages);
  if (!tables.length) return { filename: "", sheets: 0, rows: 0 };

  // Lazy-load so exceljs stays out of the initial bundle.
  const ExcelJS = await import("exceljs");
  const wb = new ExcelJS.Workbook();
  wb.creator = "Internal — DualSource Agent";

  const usedNames = new Set<string>();
  let rowCount = 0;

  tables.forEach((table, idx) => {
    const columns = (table.columns || []).map((c) => String(c ?? ""));
    const name = uniqueSheetName(
      sanitizeSheetName(String(table.title ?? `Table ${idx + 1}`)),
      usedNames,
    );
    const ws = wb.addWorksheet(name);
    ws.columns = columns.map((c) => ({ header: c, width: 10 }));
    ws.getRow(1).font = { bold: true };
    ws.views = [{ state: "frozen", ySplit: 1 }];

    const widths = columns.map((c) => Math.max(10, Math.min(c.length + 2, 40)));

    const rows = table.rows || [];
    rows.forEach((row, r) => {
      if (r >= 500) return; // keep the ≤500-row export cap
      const cells = rowCells(row, columns);
      ws.addRow(cells.map(cellValue));
      rowCount += 1;
      if (r < 200) {
        cells.forEach((cell, i) => {
          const len = cell === null || cell === undefined ? 0 : String(cell).length;
          widths[i] = Math.max(widths[i], Math.min(len + 2, 40));
        });
      }
    });
    ws.columns.forEach((col, i) => {
      col.width = widths[i];
    });
  });

  const buffer = await wb.xlsx.writeBuffer();
  const blob = new Blob([buffer], { type: XLSX_MIME });
  const date = new Date().toISOString().slice(0, 10);
  const filename = `dualsource-${slugify(params.sessionTitle)}-${date}.xlsx`;
  saveBlob(blob, filename);
  return { filename, sheets: tables.length, rows: rowCount };
}