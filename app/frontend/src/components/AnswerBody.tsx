import { Children, cloneElement, isValidElement, ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChartCard, ChartSkeleton } from "./ChartCard";
import { CiteCluster, SourceItem, domainFromUrl } from "./SourcesBar";
import {
  CHART_TOKEN_RE,
  normalizeAnswerMarkdown,
  resolveChart,
  stripChartTokens,
} from "@/lib/answerText";

type SourceLike = SourceItem;

function childrenToText(children: ReactNode): string {
  if (children == null || typeof children === "boolean") return "";
  if (typeof children === "string" || typeof children === "number") return String(children);
  if (Array.isArray(children)) return children.map(childrenToText).join("");
  if (isValidElement(children)) return childrenToText((children.props as any)?.children);
  return "";
}

function renderTextWithFallback(children: ReactNode): ReactNode {
  return Children.map(children, (child) => {
    if (typeof child === "string") {
      return child.replace(CHART_TOKEN_RE, "");
    }
    if (isValidElement(child) && child.props?.children != null) {
      return cloneElement(child, {
        children: renderTextWithFallback(child.props.children),
      } as any);
    }
    return child;
  });
}

function isWhitespaceNode(node: ReactNode): boolean {
  return typeof node === "string" && /^\s*$/.test(node);
}

function extractCiteSource(node: ReactNode, sources: SourceLike[]): SourceItem | null {
  if (!isValidElement(node)) return null;
  const props = node.props as any;
  const className = String(props?.className || "");
  if (!className.includes("cite-chip")) return null;

  if (className.includes("cite-internal")) {
    return { type: "internal", label: "Internal warehouse", cite_tag: "[Internal]" };
  }

  if (className.includes("cite-web") && props?.href) {
    const href = String(props.href);
    const label = childrenToText(props.children).trim();
    const matched = sources.find(
      (s) => s?.url && s.url.replace(/\/$/, "") === href.replace(/\/$/, ""),
    );
    return {
      type: matched?.type || "public",
      label: matched?.label || label || domainFromUrl(href),
      url: href,
      cite_tag: matched?.cite_tag,
    };
  }

  if (className.includes("cite-num")) {
    const num = childrenToText(props.children).replace(/[^\d]/g, "");
    const matched =
      sources.find((s) => (s.cite_tag || "").includes(`[${num}]`)) ||
      sources.filter((s) => s.url)[Number(num) - 1];
    if (matched) return matched;
    return { label: `Source ${num || "?"}`, cite_tag: `[[${num}]]` };
  }

  return null;
}

/** Collapse runs of adjacent citation chips into one cluster control. */
function clusterAdjacentCites(children: ReactNode, sources: SourceLike[]): ReactNode {
  const list = Children.toArray(children);
  if (list.length === 0) return children;

  const out: ReactNode[] = [];
  let i = 0;
  while (i < list.length) {
    const src = extractCiteSource(list[i], sources);
    if (!src) {
      const child = list[i];
      if (isValidElement(child) && child.props?.children != null) {
        out.push(
          cloneElement(child, {
            children: clusterAdjacentCites(child.props.children, sources),
          } as any),
        );
      } else {
        out.push(child);
      }
      i += 1;
      continue;
    }

    const group: SourceItem[] = [src];
    let j = i + 1;
    while (j < list.length) {
      if (isWhitespaceNode(list[j])) {
        j += 1;
        continue;
      }
      const next = extractCiteSource(list[j], sources);
      if (!next) break;
      group.push(next);
      j += 1;
    }
    out.push(<CiteCluster key={`cite-cluster-${i}`} items={group} compact />);
    i = j;
  }
  return out;
}

function MarkdownBlock({
  text,
  sources,
}: {
  text: string;
  sources?: SourceLike[];
}) {
  const cleaned = stripChartTokens(text);
  if (!cleaned) return null;
  const sourceList = Array.isArray(sources) ? sources : [];

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ href, children }) => {
          if (href === "#cite-internal") {
            return (
              <span className="cite-chip cite-internal" title="Internal warehouse">
                Internal
              </span>
            );
          }
          if (href?.startsWith("#cite-")) {
            return <sup className="cite-chip cite-num">{children}</sup>;
          }
          const label = childrenToText(children).trim();
          const matched = sourceList.find(
            (s) => s?.url && href && s.url.replace(/\/$/, "") === href.replace(/\/$/, ""),
          );
          const domain = domainFromUrl(href || "") || label || "source";
          const tip = matched?.label || label || domain;
          return (
            <a
              className="cite-chip cite-web"
              href={href}
              target="_blank"
              rel="noreferrer"
              title={tip}
            >
              {domain}
            </a>
          );
        },
        p: ({ children }) => (
          <p>{clusterAdjacentCites(renderTextWithFallback(children), sourceList)}</p>
        ),
        li: ({ children }) => (
          <li>{clusterAdjacentCites(renderTextWithFallback(children), sourceList)}</li>
        ),
        table: ({ children }) => (
          <div className="md-table-wrap">
            <table>{children}</table>
          </div>
        ),
        thead: ({ children }) => <thead>{children}</thead>,
        tbody: ({ children }) => <tbody>{children}</tbody>,
        tr: ({ children }) => <tr>{children}</tr>,
        th: ({ children }) => <th>{children}</th>,
        td: ({ children }) => <td>{children}</td>,
      }}
    >
      {cleaned}
    </ReactMarkdown>
  );
}

export function AnswerBody({
  markdown,
  charts,
  sources,
  streaming,
}: {
  markdown: string;
  charts?: any[];
  sources?: SourceLike[];
  /** True while answer tokens are still streaming. */
  streaming?: boolean;
}) {
  const chartList = Array.isArray(charts) ? charts : [];
  let normalized = normalizeAnswerMarkdown(markdown || "");
  if (streaming) {
    // A [[CHART:n]] marker streams in token-by-token. Hide the trailing partial
    // fragment (e.g. "[[CHART", "[[CHART:0]") so raw marker text never flashes
    // in the prose and then reflows when the marker completes. A complete
    // marker (ends with "]]" ) is intentionally NOT matched.
    normalized = normalized.replace(/\[\[\s*CHART(?:\s*:?\s*\d*)?\]?$/i, "");
  }
  const tokenRe = /\[\[CHART:(\d+)\]\]/gi;
  const nodes: ReactNode[] = [];
  const used = new Set<number>();
  let last = 0;
  let match: RegExpExecArray | null;
  let part = 0;

  while ((match = tokenRe.exec(normalized)) !== null) {
    const before = normalized.slice(last, match.index);
    if (before.trim()) {
      nodes.push(<MarkdownBlock key={`md-${part++}`} text={before} sources={sources} />);
    }
    const idx = Number(match[1]);
    // Repeated [[CHART:n]] markers must not paint the same chart twice.
    if (used.has(idx)) {
      last = match.index + match[0].length;
      continue;
    }
    const chart = resolveChart(chartList, idx);
    if (chart) {
      used.add(idx);
      nodes.push(
        <ChartCard key={chart?.id || chart?.title || `chart-${idx}`} chart={chart} />,
      );
    } else if (streaming) {
      // Marker arrived before the chart payload — hold its slot with a skeleton.
      used.add(idx);
      nodes.push(<ChartSkeleton key={`chart-skel-${idx}`} index={idx} />);
    }
    last = match.index + match[0].length;
  }

  const tail = normalized.slice(last);
  if (tail.trim()) {
    nodes.push(<MarkdownBlock key={`md-${part++}`} text={tail} sources={sources} />);
  }

  // Append unused charts (position=end, or inline markers missing) after markdown.
  // While streaming, markers may not have arrived yet — wait for them instead of
  // rendering charts at the end and then jumping them inline.
  if (!streaming) {
    chartList.forEach((c, i) => {
      const idx = Number.isFinite(Number(c?.index)) ? Number(c.index) : i;
      if (used.has(idx)) return;
      nodes.push(<ChartCard key={c?.id || c?.title || `chart-end-${idx}`} chart={c} />);
      used.add(idx);
    });
  }

  if (nodes.length === 0) {
    return (
      <MarkdownBlock text={stripChartTokens(normalized)} sources={sources} />
    );
  }
  return <>{nodes}</>;
}
