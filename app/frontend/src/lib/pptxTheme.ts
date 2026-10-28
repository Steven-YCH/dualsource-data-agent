/**
 * Internal PPT design tokens — single source of truth for the conversation export
 * deck (spec: docs/design/conversation-ppt-export.md, "Design system v2").
 * Keep this module pure (no pptxgenjs / echarts imports) so both the deck
 * builder and the chart-capture theme read from the same values.
 */

export const PPTX_THEME = {
  fonts: {
    head: "Microsoft YaHei",
    body: "Microsoft YaHei",
  },
  colors: {
    brand: "0A3D73",
    brandLight: "2E7CC1",
    accent: "1C8C7E",
    text: "1F2937",
    muted: "64748B",
    line: "CBD5E1", // hairline table borders / footer rule
    headFill: "EAF1F8", // table header
    zebra: "F5F8FB", // table odd-row fill
    cardFill: "F8FAFC", // answer-block background
  },
  geometry: {
    slideW: 13.333,
    slideH: 7.5,
    marginX: 0.55,
    accentH: 0.18,
    footnotesY: 6.72, // per-slide source footnote strip
    footerY: 7.22, // "Internal · title · date" line
    contentBottom: 6.68, // lowest y for tables/charts
  },
  type: {
    coverSubtitle: 13,
    coverTitle: 32,
    slideTitle: 17,
    answer: 11.5,
    caption: 12.5, // chart caption above the image
    agendaItem: 13,
    footnote: 8,
    tableHead: 9.5,
    tableBody: 9,
    footer: 7.5,
  },
  /** Chart-capture pixel geometry (echarts canvas) — see captureChartPng. */
  capture: {
    width: 1280,
    pixelRatio: 2,
    background: "#ffffff",
  },
} as const;

export const CONTENT_W = PPTX_THEME.geometry.slideW - PPTX_THEME.geometry.marginX * 2;

/** Master slide names — see defineZeaMasters() in exportPptx.ts. */
export const MASTER_COVER = "Internal_COVER";
export const MASTER_CONTENT = "Internal_CONTENT";
