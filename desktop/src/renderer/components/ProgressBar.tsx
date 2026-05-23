import React from "react";
import clsx from "clsx";

type Props = {
  value?: number | null;
  indeterminate?: boolean;
  tone?: "accent" | "warn" | "danger";
  className?: string;
};

const fillColor = {
  accent: "bg-[var(--color-accent-strong)]",
  warn: "bg-[var(--color-warn)]",
  danger: "bg-[var(--color-danger)]"
} as const;

export function ProgressBar({ value, indeterminate, tone = "accent", className }: Props) {
  const safe =
    typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : null;

  return (
    <div
      className={clsx(
        "relative h-1.5 w-full overflow-hidden rounded-full bg-[var(--color-surface-3)]",
        className
      )}
    >
      {indeterminate ? (
        <div className={clsx("absolute inset-y-0 w-1/3 animate-[slide_1.4s_ease-in-out_infinite]", fillColor[tone])}>
          <style>{`@keyframes slide { 0% { transform: translateX(-120%); } 100% { transform: translateX(420%); } }`}</style>
        </div>
      ) : (
        <div
          className={clsx("h-full transition-[width] duration-300 ease-out", fillColor[tone])}
          style={{ width: `${safe ?? 0}%` }}
        />
      )}
    </div>
  );
}
