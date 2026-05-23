import React from "react";
import clsx from "clsx";
import { motion } from "framer-motion";

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

const glowColor = {
  accent:
    "linear-gradient(90deg, transparent 0%, var(--color-scan-glow) 50%, transparent 100%)",
  warn: "linear-gradient(90deg, transparent 0%, rgba(251,191,36,0.55) 50%, transparent 100%)",
  danger: "linear-gradient(90deg, transparent 0%, rgba(248,113,113,0.55) 50%, transparent 100%)"
} as const;

export function ProgressBar({ value, indeterminate, tone = "accent", className }: Props) {
  const safe =
    typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : null;

  return (
    <div
      className={clsx(
        "relative h-2 w-full overflow-hidden rounded-full bg-[var(--color-surface-3)]",
        "shadow-[inset_0_1px_2px_rgba(0,0,0,0.25)]",
        className
      )}
    >
      {/* faint grid pattern reads as 'industrial' */}
      <div
        className="absolute inset-0 opacity-40 pointer-events-none"
        style={{
          backgroundImage:
            "repeating-linear-gradient(90deg, transparent 0 7px, rgba(255,255,255,0.05) 7px 8px)"
        }}
      />

      {indeterminate ? (
        <>
          {/* base glow strip */}
          <div className={clsx("absolute inset-y-0 left-0 right-0 opacity-30", fillColor[tone])} />
          {/* scanning beam */}
          <div
            className="absolute inset-y-0 w-1/3"
            style={{
              backgroundImage: glowColor[tone],
              animation: "scan-beam 1.6s cubic-bezier(0.4, 0, 0.2, 1) infinite"
            }}
          />
        </>
      ) : (
        <motion.div
          className={clsx("h-full rounded-full", fillColor[tone])}
          initial={false}
          animate={{ width: `${safe ?? 0}%` }}
          transition={{ type: "spring", stiffness: 120, damping: 22 }}
        >
          {/* leading edge highlight */}
          <div
            className="h-full w-8 ml-auto opacity-70"
            style={{
              backgroundImage:
                "linear-gradient(90deg, transparent, rgba(255,255,255,0.6))"
            }}
          />
        </motion.div>
      )}
    </div>
  );
}
