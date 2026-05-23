import React from "react";
import clsx from "clsx";

type Tone = "neutral" | "success" | "warn" | "danger" | "info" | "accent";

type Props = {
  tone?: Tone;
  dot?: boolean;
  pulse?: boolean;
  className?: string;
  children: React.ReactNode;
};

const toneClass: Record<Tone, string> = {
  neutral: "bg-[var(--color-surface-2)] text-[var(--color-text-muted)] border-[var(--color-border)]",
  success: "bg-[var(--color-accent)]/10 text-[var(--color-accent)] border-[var(--color-accent)]/30",
  warn: "bg-[var(--color-warn)]/10 text-[var(--color-warn)] border-[var(--color-warn)]/30",
  danger: "bg-[var(--color-danger)]/10 text-[var(--color-danger)] border-[var(--color-danger)]/30",
  info: "bg-[var(--color-info)]/10 text-[var(--color-info)] border-[var(--color-info)]/30",
  accent: "bg-[var(--color-accent-strong)] text-[var(--color-on-accent)] border-transparent"
};

const dotClass: Record<Tone, string> = {
  neutral: "bg-[var(--color-text-dim)]",
  success: "bg-[var(--color-accent)]",
  warn: "bg-[var(--color-warn)]",
  danger: "bg-[var(--color-danger)]",
  info: "bg-[var(--color-info)]",
  accent: "bg-[var(--color-on-accent)]"
};

export function Badge({ tone = "neutral", dot, pulse, className, children }: Props) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5",
        "text-[11px] font-medium leading-none whitespace-nowrap font-mono",
        toneClass[tone],
        className
      )}
    >
      {dot ? (
        <span className="relative flex w-2 h-2 shrink-0">
          {pulse ? (
            <span
              className={clsx("absolute inset-0 rounded-full opacity-70", dotClass[tone])}
              style={{ animation: "pulse-ring 1.8s cubic-bezier(0.4,0,0.2,1) infinite" }}
            />
          ) : null}
          <span className={clsx("relative inline-flex w-2 h-2 rounded-full", dotClass[tone])} />
        </span>
      ) : null}
      {children}
    </span>
  );
}
