import { useEffect, useId, useRef, useState } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export type SourceItem = {
  type?: string;
  label?: string;
  url?: string;
  cite_tag?: string;
};

/** Show this many source chips before collapsing into "+N". GodUI / AI citation UX default. */
export const SOURCE_PREVIEW_COUNT = 3;

export function domainFromUrl(url: string): string {
  try {
    const host = new URL(url).hostname.toLowerCase();
    return host.replace(/^www\./, "");
  } catch {
    return "";
  }
}

function sourceKey(s: SourceItem, idx: number): string {
  return `${(s.url || "").replace(/\/$/, "")}|${s.label || ""}|${idx}`;
}

/* Citation chip look, replicated from app.css .cite-chip so chips rendered by
   AnswerBody (still legacy) and CiteCluster (utilities) are pixel-identical.
   `!` suffixes: app.css's global unlayered `button { font: inherit }` beats
   layered utilities, so font size/weight on button text needs important. */
const citeChipBase =
  "mx-[3px] inline-flex items-center rounded-full border px-[7px] py-px align-baseline text-[11px] font-[650] leading-[1.45] whitespace-nowrap";
/* Button-rendered chips (cluster trigger): font utilities need `!` because
   app.css's global unlayered `button { font: inherit }` beats layered ones. */
const citeChipBtn =
  "mx-[3px] inline-flex items-center rounded-full border px-[7px] py-px align-baseline text-[11px]! font-[650]! leading-[1.45] whitespace-nowrap";

export function CiteCluster({
  items,
  compact = false,
}: {
  items: SourceItem[];
  compact?: boolean;
}) {
  const unique: SourceItem[] = [];
  const seen = new Set<string>();
  for (const s of items) {
    const key = (s.url || s.label || "").replace(/\/$/, "").toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    unique.push(s);
  }
  if (unique.length === 0) return null;
  if (unique.length === 1) {
    const s = unique[0];
    const isZeap = s.type === "internal" || s.type === "hkia_internal" || /internal|zeap/i.test(s.label || "") || /internal|zeap|\[Internal\]/i.test(s.cite_tag || "");
    if (isZeap || !s.url) {
      return (
        <Tooltip>
          <TooltipTrigger asChild>
            <span
              className={`${citeChipBase} cursor-pointer border-zeap-line bg-zeap-bg text-zeap-fg`}
            >
              Internal
            </span>
          </TooltipTrigger>
          <TooltipContent>{s.label || "Internal warehouse"}</TooltipContent>
        </Tooltip>
      );
    }
    const domain = domainFromUrl(s.url) || s.label || "source";
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <a
            className={`${citeChipBase} cursor-pointer border-cite-line bg-cite-bg font-semibold text-cite-fg no-underline hover:bg-link-hover-bg hover:border-link-hover-line hover:text-link`}
            href={s.url}
            target="_blank"
            rel="noreferrer"
          >
            {domain}
          </a>
        </TooltipTrigger>
        <TooltipContent className="max-w-[280px] break-words">{s.label || domain}</TooltipContent>
      </Tooltip>
    );
  }

  return (
    <SourcePopover
      sources={unique}
      triggerLabel={compact ? `${unique.length}` : `${unique.length} sources`}
      triggerClassName={`${citeChipBtn} cursor-pointer border-cite-line bg-cite-bg text-cite-fg tabular-nums hover:bg-link-hover-bg hover:border-link-hover-line hover:text-link`}
      ariaLabel={`${unique.length} sources`}
    />
  );
}

function SourcePopover({
  sources,
  triggerLabel,
  triggerClassName,
  ariaLabel,
}: {
  sources: SourceItem[];
  triggerLabel: string;
  triggerClassName: string;
  ariaLabel: string;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLSpanElement>(null);
  const panelId = useId();
  const closeTimer = useRef<number | null>(null);

  const clearClose = () => {
    if (closeTimer.current != null) {
      window.clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
  };

  const scheduleClose = () => {
    clearClose();
    closeTimer.current = window.setTimeout(() => setOpen(false), 160);
  };

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  useEffect(() => () => clearClose(), []);

  return (
    <span
      className="relative inline-flex align-baseline"
      ref={rootRef}
      onMouseEnter={() => {
        clearClose();
        setOpen(true);
      }}
      onMouseLeave={scheduleClose}
    >
      <button
        type="button"
        className={triggerClassName}
        aria-expanded={open}
        aria-controls={panelId}
        aria-label={ariaLabel}
        onClick={() => setOpen((v) => !v)}
      >
        {triggerLabel}
      </button>
      {open && (
        <div
          id={panelId}
          className="absolute bottom-[calc(100%+8px)] left-0 z-40 max-h-[280px] w-[min(360px,78vw)] overflow-auto rounded-md border border-line-strong bg-white p-2.5 shadow-[0_14px_36px_rgba(16,35,61,0.14)]"
          role="dialog"
          aria-label="Sources"
          onMouseEnter={clearClose}
          onMouseLeave={scheduleClose}
        >
          <div className="mb-2 text-[11px] font-bold uppercase tracking-[0.02em] text-muted-foreground">
            {sources.length} sources
          </div>
          <ul className="m-0 flex list-none flex-col gap-1 p-0">
            {sources.map((s, idx) => {
              const domain = s.url ? domainFromUrl(s.url) : "";
              const title = s.label || domain || "Source";
              const isZeap =
                s.type === "internal" || s.type === "hkia_internal" ||
                /zeap/i.test(s.label || "") ||
                /zeap/i.test(s.cite_tag || "");
              if (!s.url || isZeap) {
                return (
                  <li key={sourceKey(s, idx)}>
                    <div className="flex flex-col gap-0.5 rounded-[10px] border border-transparent bg-ok-bg px-2.5 py-2">
                      <span className="text-xs font-bold text-zeap-fg">Internal</span>
                      <span className="line-clamp-2 text-[11px] leading-[1.35] text-ink-soft">{title}</span>
                    </div>
                  </li>
                );
              }
              return (
                <li key={sourceKey(s, idx)}>
                  <a
                    className="flex flex-col gap-0.5 rounded-[10px] border border-transparent px-2.5 py-2 no-underline hover:border-cite-line hover:bg-secondary"
                    href={s.url}
                    target="_blank"
                    rel="noreferrer"
                    title={title}
                  >
                    <span className="text-xs font-bold text-link">{domain || "source"}</span>
                    <span className="line-clamp-2 text-[11px] leading-[1.35] text-ink-soft">{title}</span>
                  </a>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </span>
  );
}

export function SourcesBar({ sources }: { sources: SourceItem[] }) {
  const web = (sources || []).filter((s) => s?.type !== "hkia_internal" && s?.url);
  const deduped: SourceItem[] = [];
  const seen = new Set<string>();
  for (const s of web) {
    const key = (s.url || "").replace(/\/$/, "").toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    deduped.push(s);
  }
  if (deduped.length === 0) return null;

  const visible = deduped.slice(0, SOURCE_PREVIEW_COUNT);
  const hidden = deduped.slice(SOURCE_PREVIEW_COUNT);

  return (
    <div
      className="mt-2 flex flex-wrap items-center gap-1.5 opacity-[0.92]"
      aria-label="Sources"
    >
      {visible.map((s, idx) => {
        const domain = domainFromUrl(s.url || "") || s.label || "source";
        return (
          <Tooltip key={sourceKey(s, idx)}>
            <TooltipTrigger asChild>
              <a
                className="inline-flex cursor-pointer items-center gap-1 rounded-full border border-chip-quiet-line bg-chip-quiet-bg px-2 py-[3px] text-[11px] font-medium text-chip-quiet-fg no-underline hover:border-chip-link-hover-line hover:bg-chip-link-hover-bg hover:text-brand-deep"
                href={s.url}
                target="_blank"
                rel="noreferrer"
              >
                {domain}
              </a>
            </TooltipTrigger>
            <TooltipContent className="max-w-[320px] break-words">
              {s.label || domain}
            </TooltipContent>
          </Tooltip>
        );
      })}
      {hidden.length > 0 && (
        <SourcePopover
          sources={deduped}
          triggerLabel={`+${hidden.length}`}
          triggerClassName="inline-flex cursor-pointer items-center rounded-full border border-dashed border-chip-quiet-line bg-chip-more-bg px-2 py-[3px] text-[11px]! font-medium! text-chip-more-fg hover:border-link-hover-line hover:bg-link-hover-bg hover:text-link aria-expanded:border-link-hover-line aria-expanded:bg-link-hover-bg aria-expanded:text-link"
          ariaLabel={`Show all ${deduped.length} sources`}
        />
      )}
    </div>
  );
}
