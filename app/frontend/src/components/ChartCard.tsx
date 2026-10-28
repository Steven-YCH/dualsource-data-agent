import ReactECharts from "echarts-for-react";
import { cn } from "@/lib/utils";
import {
  computeCanvasHeight,
  isHorizontalBar,
  resolveEchartsOption,
  scrubNulls,
} from "@/lib/chartOption";
import { polishTitle } from "./ChatElements";

export function ChartCard({ chart }: { chart: any }) {
  const optionRaw = resolveEchartsOption(chart);
  if (!optionRaw) return null;

  const optionTitle =
    optionRaw.title && typeof optionRaw.title === "object"
      ? (optionRaw.title as { text?: unknown }).text
      : undefined;
  const rawTitle =
    chart?.title || chart?.plan?.title || chart?.chart_plan?.title || optionTitle;
  const polished = polishTitle(typeof rawTitle === "string" ? rawTitle : undefined);
  const title = !rawTitle || polished === "Query result" ? "Chart" : polished;
  const type = String(
    chart?.type || chart?.plan?.type || chart?.chart_plan?.type || (optionRaw as any)?.series?.[0]?.type || "line",
  ).toLowerCase();
  const horizontal = isHorizontalBar(optionRaw, chart);
  // Backend sends an authoritative `height` hint on every chart (see
  // chart_plan.py sizing contract); fallbacks only apply to foreign payloads
  // (e.g. Serving path charts without the hint). Mirrors lib/chartOption.
  const canvasHeight = computeCanvasHeight(chart, optionRaw, horizontal);

  const option = scrubNulls({
    ...optionRaw,
    title: {
      ...(typeof optionRaw.title === "object" && optionRaw.title ? optionRaw.title : {}),
      text: title,
      left: "center",
      textStyle: {
        fontSize: 13,
        fontWeight: 600,
        color: "#0a3d73",
        ...((optionRaw as any)?.title?.textStyle || {}),
      },
    },
    toolbox: {
      right: 8,
      feature: {
        saveAsImage: { title: "Save" },
        ...(horizontal
          ? {}
          : { dataZoom: { title: { zoom: "Zoom", back: "Reset" } } }),
      },
      ...((optionRaw as any).toolbox || {}),
    },
  });

  return (
    <div
      className={cn(
        "mt-3 animate-[chart-fade-in_0.35s_ease-out] overflow-hidden rounded-md border border-line bg-tint-bg",
        horizontal && "bg-hbar-gradient",
      )}
    >
      <div className="flex items-center justify-between gap-2 border-b border-cell-line bg-head-gradient px-3 py-2.5">
        <div className="text-[13px] font-[650] text-brand-deep">{title}</div>
        <div className="text-[11px] text-muted-foreground">
          {horizontal ? "BAR · RANKED" : type.toUpperCase()}
        </div>
      </div>
      <div
        className={cn("w-full px-3 pt-2.5 pb-3.5", horizontal && "min-h-[320px]")}
        style={{ height: canvasHeight }}>
        <ReactECharts
          option={option}
          notMerge
          lazyUpdate
          style={{ width: "100%", height: "100%" }}
          opts={{ renderer: "canvas" }}
        />
      </div>
    </div>
  );
}

/**
 * Placeholder shown at a [[CHART:n]] marker while the chart is still building.
 * Reserves the exact box a ChartCard will occupy (400px canvas by default) so
 * text around the slot never reflows when the real chart swaps in.
 */
export function ChartSkeleton({
  index,
  height,
}: {
  index?: number;
  /** Override only when the final chart height is already known. */
  height?: number;
}) {
  return (
    <div
      className="mt-3 animate-[chart-fade-in_0.35s_ease-out] overflow-hidden rounded-md border border-line bg-tint-bg"
      aria-busy="true"
    >
      <div className="flex items-center justify-between gap-2 border-b border-cell-line bg-head-gradient px-3 py-2.5">
        <div className="text-[13px] font-[650] text-brand-deep">Preparing chart…</div>
        <div className="text-[11px] text-muted-foreground">
          {typeof index === "number" ? `CHART ${index + 1}` : "CHART"}
        </div>
      </div>
      {/* Reserves the EXACT slot a real ChartCard will occupy (default 400px
          canvas + same padding) so the swap causes zero layout shift. */}
      <div
        className="flex h-[400px] items-end justify-center gap-[18px] px-3 pt-2.5 pb-3.5"
        style={typeof height === "number" ? { height } : undefined}
      >
        {[40, 65, 52, 78, 60].map((barHeight, i) => (
          <div
            key={i}
            className="w-[42px] animate-[chart-shimmer_1.4s_ease-in-out_infinite] rounded-t-[6px] bg-shimmer-bar"
            style={{
              height: `${barHeight}%`,
              animationDelay: i > 0 ? `${i * 0.12}s` : undefined,
            }}
          />
        ))}
      </div>
    </div>
  );
}
