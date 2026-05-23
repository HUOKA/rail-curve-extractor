import React from "react";
import clsx from "clsx";
import { motion } from "framer-motion";
import { Check, X } from "lucide-react";

export type StepState = "pending" | "running" | "done" | "failed" | "skipped";

export type StepItem = {
  id: string;
  label: string;
  state: StepState;
  detail?: string;
};

type Props = {
  steps: StepItem[];
  className?: string;
};

const nodeColor: Record<StepState, string> = {
  pending: "bg-[var(--color-surface-3)] text-[var(--color-text-dim)] border-[var(--color-border)]",
  running: "bg-[var(--color-accent-soft)] text-[var(--color-accent)] border-[var(--color-accent)]/60",
  done: "bg-[var(--color-accent-strong)] text-[var(--color-on-accent)] border-[var(--color-accent-strong)]",
  failed: "bg-[var(--color-danger)]/15 text-[var(--color-danger)] border-[var(--color-danger)]/60",
  skipped: "bg-[var(--color-surface-3)] text-[var(--color-text-dim)] border-[var(--color-border)]"
};

export function Stepper({ steps, className }: Props) {
  const runningIndex = steps.findIndex((step) => step.state === "running");

  return (
    <ol className={clsx("flex items-stretch gap-0", className)}>
      {steps.map((step, index) => {
        const isLast = index === steps.length - 1;
        const next = steps[index + 1];

        // Connector visual after this node:
        // - if this step is done AND next exists: solid accent
        // - if this step is running: scan beam to convey activity
        // - else: muted border
        let connectorClass = "bg-[var(--color-border)]";
        let connectorOverlay: React.ReactNode = null;
        if (!isLast && next) {
          if (step.state === "done" && (next.state === "done" || next.state === "running")) {
            connectorClass = "bg-[var(--color-accent-strong)]";
          } else if (step.state === "running") {
            connectorClass = "bg-[var(--color-accent)]/20";
            connectorOverlay = (
              <div
                className="absolute inset-y-0 w-1/2"
                style={{
                  backgroundImage:
                    "linear-gradient(90deg, transparent, var(--color-scan-glow), transparent)",
                  animation: "scan-beam 1.4s cubic-bezier(0.4,0,0.2,1) infinite"
                }}
              />
            );
          } else if (step.state === "failed") {
            connectorClass = "bg-[var(--color-danger)]/30";
          }
        }

        return (
          <li key={step.id} className="flex-1 min-w-0 flex flex-col items-center">
            <div className="flex items-center w-full h-7">
              {/* left segment of the line */}
              <div className="flex-1 h-px relative overflow-hidden">
                {index === 0 ? null : (
                  <div
                    className={clsx(
                      "absolute inset-0",
                      steps[index - 1].state === "done" || step.state === "done" || step.state === "running"
                        ? "bg-[var(--color-accent-strong)]"
                        : "bg-[var(--color-border)]"
                    )}
                  />
                )}
              </div>

              {/* node */}
              <StepNode step={step} index={index} isCurrent={index === runningIndex} />

              {/* right segment */}
              <div className="flex-1 h-px relative overflow-hidden">
                {isLast ? null : (
                  <>
                    <div className={clsx("absolute inset-0", connectorClass)} />
                    {connectorOverlay}
                  </>
                )}
              </div>
            </div>

            <div className="mt-2 px-1 text-center min-w-0 max-w-full">
              <div
                className={clsx(
                  "text-[12px] font-medium truncate transition-colors duration-200",
                  step.state === "running"
                    ? "text-[var(--color-accent)]"
                    : step.state === "failed"
                      ? "text-[var(--color-danger)]"
                      : step.state === "done"
                        ? "text-[var(--color-text)]"
                        : "text-[var(--color-text-muted)]"
                )}
              >
                {step.label}
              </div>
              {step.detail ? (
                <div className="text-[11px] text-[var(--color-text-dim)] truncate font-mono">
                  {step.detail}
                </div>
              ) : null}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function StepNode({
  step,
  index,
  isCurrent
}: {
  step: StepItem;
  index: number;
  isCurrent: boolean;
}) {
  return (
    <div className="relative shrink-0">
      <div
        className={clsx(
          "relative w-7 h-7 rounded-full border flex items-center justify-center text-[11px] font-semibold transition-colors duration-200",
          nodeColor[step.state]
        )}
      >
        {step.state === "done" ? (
          <Check size={14} strokeWidth={2.6} />
        ) : step.state === "failed" ? (
          <X size={14} strokeWidth={2.6} />
        ) : step.state === "running" ? (
          <span className="relative flex w-2 h-2">
            <span
              className="absolute inset-0 rounded-full bg-[var(--color-accent)]"
              style={{ animation: "pulse-ring 1.6s cubic-bezier(0.4,0,0.2,1) infinite" }}
            />
            <span className="relative inline-flex w-2 h-2 rounded-full bg-[var(--color-accent)]" />
          </span>
        ) : (
          index + 1
        )}
      </div>

      {/* halo around currently-running node */}
      {isCurrent ? (
        <motion.div
          layoutId="step-halo"
          className="absolute inset-[-6px] rounded-full ring-2 ring-[var(--color-accent)]/30 pointer-events-none"
          transition={{ type: "spring", stiffness: 320, damping: 28 }}
        />
      ) : null}
    </div>
  );
}
