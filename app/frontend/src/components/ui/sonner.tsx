import type { CSSProperties } from "react";
import { Toaster as Sonner, type ToasterProps } from "sonner";

/**
 * Sonner toaster themed onto the repo tokens. `--popover` / `--border` come
 * from shadcn.css :root; `--radius-md` resolves to the legacy the hub radius
 * (14px) because unlayered app.css beats the Tailwind theme layer — intended.
 */
const Toaster = ({ ...props }: ToasterProps) => {
  return (
    <Sonner
      theme="light"
      className="toaster group"
      style={
        {
          "--normal-bg": "var(--popover)",
          "--normal-text": "var(--popover-foreground)",
          "--normal-border": "var(--border)",
          "--border-radius": "var(--radius-md)",
          fontFamily: "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif",
        } as CSSProperties
      }
      {...props}
    />
  );
};

export { Toaster };
