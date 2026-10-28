/**
 * Shared ECharts option helpers — moved verbatim from ChartCard.tsx
 * (spec: docs/design/conversation-ppt-export.md, "Shared-code refactor").
 * Import from both ChartCard and the PPT export helper; do not copy-paste.
 */

// Canvas-height contract shared with the backend
// (app/backend/app/local_react/chart_plan.py). The backend sends an
// authoritative `height` hint per chart; these are only fallbacks.
export const CHART_HEIGHT_DEFAULT = 400;
export const CHART_HEIGHT_H_MIN = 320;
export const CHART_HEIGHT_H_BASE = 56;
export const CHART_HEIGHT_H_PER_ROW = 42;

/** Drop nulls that Python serializers sometimes leave in ECharts options. */
export function scrubNulls<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map((v) => scrubNulls(v)) as T;
  }
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if (v === null || v === undefined) continue;
      out[k] = scrubNulls(v);
    }
    return out as T;
  }
  return value;
}

export function hasRenderableEcharts(option: any): boolean {
  const series = Array.isArray(option?.series) ? option.series : [];
  if (!series.length) return false;
  let numericCount = 0;
  let nonZero = false;
  for (const s of series) {
    const data = Array.isArray(s?.data) ? s.data : [];
    for (const point of data) {
      let n: number | null = null;
      if (typeof point === "number") n = point;
      else if (point && typeof point === "object" && "value" in point) {
        const v = (point as { value: unknown }).value;
        n = typeof v === "number" ? v : Number(String(v).replace(/,/g, ""));
      } else if (typeof point === "string") {
        n = Number(point.replace(/,/g, ""));
      }
      if (n == null || !Number.isFinite(n)) continue;
      numericCount += 1;
      if (n !== 0) nonZero = true;
    }
  }
  return numericCount >= 2 && nonZero;
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const n = Number(value.replace(/,/g, "").trim());
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function humanizeField(name: string): string {
  return String(name || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}

type ChartPlan = {
  type?: string;
  x?: string;
  y?: string | string[];
  title?: string;
  series?: Array<{ name?: string; y?: string; data?: unknown[] }>;
  categories?: unknown[];
  data?: unknown[] | Record<string, unknown>[];
  rows?: unknown[][];
  columns?: string[];
};

/**
 * Build an ECharts option from an LLM chart plan `{type,x,y[],title}` plus
 * tabular data (columns/rows, row objects, or pre-split series).
 */
export function echartsFromPlan(chart: any): Record<string, unknown> | null {
  const plan: ChartPlan =
    (chart?.plan && typeof chart.plan === "object" ? chart.plan : null) ||
    (chart?.chart_plan && typeof chart.chart_plan === "object" ? chart.chart_plan : null) ||
    chart ||
    {};

  const chartType = String(plan.type || chart?.type || "line").toLowerCase();
  const title = String(plan.title || chart?.title || "Chart");
  const xField = String(plan.x || chart?.x || "").trim();
  const yFieldsRaw = plan.y ?? chart?.y;
  const yFields: string[] = Array.isArray(yFieldsRaw)
    ? yFieldsRaw.map(String)
    : yFieldsRaw
      ? [String(yFieldsRaw)]
      : [];

  // Prefer explicit categories + series arrays on the plan.
  if (Array.isArray(plan.categories) && Array.isArray(plan.series) && plan.series.length) {
    const categories = plan.categories.map((c) => String(c ?? ""));
    if (chartType === "pie") {
      const first = plan.series[0];
      const data = (Array.isArray(first?.data) ? first.data : []).map((v, i) => {
        if (v && typeof v === "object" && "value" in (v as object)) return v;
        return { name: categories[i] || `Item ${i + 1}`, value: toNumber(v) ?? 0 };
      });
      return {
        title: { text: title, left: "center" },
        tooltip: { trigger: "item" },
        series: [{ type: "pie", radius: "65%", data }],
      };
    }
    const series = plan.series.map((s) => ({
      name: s.name || humanizeField(s.y || "value"),
      type: chartType === "bar" ? "bar" : chartType === "area" ? "line" : chartType || "line",
      data: (Array.isArray(s.data) ? s.data : []).map((v) => toNumber(v) ?? 0),
      ...(chartType === "area" ? { areaStyle: {} } : {}),
      ...(chartType === "line" || chartType === "area" ? { smooth: true } : {}),
    }));
    return {
      title: { text: title, left: "center" },
      tooltip: { trigger: "axis" },
      legend: { top: 28 },
      grid: { left: 48, right: 24, top: 56, bottom: 48 },
      xAxis: {
        type: "category",
        data: categories,
        axisLabel: { rotate: categories.length > 8 ? 30 : 0 },
      },
      yAxis: { type: "value" },
      series,
    };
  }

  // Tabular: columns + rows, or array of row objects.
  const columns: string[] = Array.isArray(plan.columns)
    ? plan.columns.map(String)
    : Array.isArray(chart?.columns)
      ? chart.columns.map(String)
      : Array.isArray(chart?.table?.columns)
        ? chart.table.columns.map(String)
        : [];
  const rows: unknown[][] = Array.isArray(plan.rows)
    ? (plan.rows as unknown[][])
    : Array.isArray(chart?.rows)
      ? chart.rows
      : Array.isArray(chart?.table?.rows)
        ? chart.table.rows
        : [];

  let rowObjects: Record<string, unknown>[] = [];
  if (Array.isArray(plan.data) && plan.data.length && typeof plan.data[0] === "object" && !Array.isArray(plan.data[0])) {
    rowObjects = plan.data as Record<string, unknown>[];
  } else if (columns.length && rows.length) {
    rowObjects = rows.map((row) => {
      const obj: Record<string, unknown> = {};
      columns.forEach((c, i) => {
        obj[c] = Array.isArray(row) ? row[i] : undefined;
      });
      return obj;
    });
  }

  if (!rowObjects.length) return null;

  const keys = Object.keys(rowObjects[0] || {});
  const resolvedX =
    (xField && keys.includes(xField) ? xField : null) ||
    keys.find((k) => /month|date|year|period|label|name|category/i.test(k)) ||
    keys[0];
  const resolvedY =
    yFields.filter((y) => keys.includes(y)).length > 0
      ? yFields.filter((y) => keys.includes(y))
      : keys.filter((k) => k !== resolvedX && rowObjects.some((r) => toNumber(r[k]) != null));

  if (!resolvedX || !resolvedY.length) return null;

  const categories = rowObjects.map((r) => String(r[resolvedX] ?? ""));

  if (chartType === "pie") {
    const y0 = resolvedY[0];
    return {
      title: { text: title, left: "center" },
      tooltip: { trigger: "item" },
      series: [
        {
          type: "pie",
          radius: "65%",
          data: rowObjects.map((r) => ({
            name: String(r[resolvedX] ?? ""),
            value: toNumber(r[y0]) ?? 0,
          })),
        },
      ],
    };
  }

  const series = resolvedY.map((y) => ({
    name: humanizeField(y),
    type: chartType === "bar" ? "bar" : chartType === "area" ? "line" : "line",
    data: rowObjects.map((r) => toNumber(r[y]) ?? 0),
    ...(chartType === "area" ? { areaStyle: {} } : {}),
    ...(chartType !== "bar" ? { smooth: true } : {}),
  }));

  return {
    title: { text: title, left: "center" },
    tooltip: { trigger: "axis" },
    legend: series.length > 1 ? { top: 28 } : undefined,
    grid: { left: 48, right: 24, top: series.length > 1 ? 56 : 40, bottom: 48 },
    xAxis: {
      type: "category",
      data: categories,
      axisLabel: { rotate: categories.length > 8 ? 30 : 0 },
    },
    yAxis: { type: "value" },
    series,
  };
}

export function resolveEchartsOption(chart: any): Record<string, unknown> | null {
  // New agent contract uses `echarts`. Ignore legacy `chartjs` payloads.
  const optionRaw = chart?.echarts;
  if (optionRaw && typeof optionRaw === "object" && hasRenderableEcharts(optionRaw)) {
    return optionRaw;
  }
  // LLM chart plan path (local_react): type/x/y + data → ECharts
  const fromPlan = echartsFromPlan(chart);
  if (fromPlan && hasRenderableEcharts(fromPlan)) return fromPlan;
  // Last resort: accept a slightly incomplete echarts object if series exist
  if (optionRaw && typeof optionRaw === "object" && Array.isArray(optionRaw.series) && optionRaw.series.length) {
    return optionRaw;
  }
  return null;
}

export function isHorizontalBar(option: Record<string, unknown> | null, chart: any): boolean {
  if (!option) return false;
  if (String(chart?.orientation || "").toLowerCase() === "horizontal") return true;
  const x = option.xAxis as any;
  const y = option.yAxis as any;
  const xType = Array.isArray(x) ? x[0]?.type : x?.type;
  const yType = Array.isArray(y) ? y[0]?.type : y?.type;
  return String(xType) === "value" && String(yType) === "category";
}

/**
 * Canvas height for a chart: authoritative `chart.height` hint when present,
 * else the ChartCard fallback contract (horizontal bar scales with category
 * count; default 400). Mirrors the in-chat sizing exactly.
 */
export function computeCanvasHeight(
  chart: any,
  option: Record<string, unknown> | null,
  horizontal: boolean,
): number {
  if (typeof chart?.height === "number") return chart.height;
  if (!horizontal) return CHART_HEIGHT_DEFAULT;
  const axis = option ? ((option as any).yAxis ?? null) : null;
  const data = Array.isArray(axis) ? axis[0]?.data : axis?.data;
  const categoryCount = Array.isArray(data) ? data.length : 0;
  return Math.max(
    CHART_HEIGHT_H_MIN,
    CHART_HEIGHT_H_BASE + Math.max(categoryCount, 2) * CHART_HEIGHT_H_PER_ROW,
  );
}
