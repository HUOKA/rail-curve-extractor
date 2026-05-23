import React from "react";
import clsx from "clsx";

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

const stateColor: Record<StepState, string> = {
  pending: "bg-[var(--color-surface-3)] text-[var(--color-text-dim)] border-[var(--color-border)]",
  running: "bg-[var(--color-accent)]/15 text-[var(--color-accent)] border-[var(--color-accent)]/40",
  done: "bg-[var(--color-accent-strong)] text-[var(--color-on-accent)] border-[var(--color-accent-strong)]",
  failed: "bg-[var(--color-danger)]/15 text-[var(--color-danger)] border-[var(--color-danger)]/40",
  skipped: "bg-[var(--color-surface-3)] text-[var(--color-text-dim)] border-[var(--color-border)]"
};

const connectorColor: Record<StepState, string> = {
  pending: "bg-[var(--color-border)]",
  running: "bg-[var(--color-accent)]/40",
  done: "bg-[var(--color-accent-strong)]",
  failed: "bg-[var(--color-danger)]/40",
  skipped: "bg-[var(--color-border)]"
};

export function Stepper({ steps, className }: Props) {
  return (
    <ol className={clsx("flex items-stretch gap-0", className)}>
      {steps.map((step, index) => {
        const isLast = index === steps.length - 1;
        return (
          <li key={step.id} className="flex-1 min-w-0 flex items-start">
            <div className="flex flex-col items-center min-w-0 flex-1">
              <div className="flex items-center w-full">
                <div className="flex-1 flex items-center">
                  <div
                    className={clsx(
                      "h-px flex-1",
                      index === 0 ? "invisible" : connectorColor[step.state]
                    )}
                  />
                </div>
                <span
                  className={clsx(
                    "relative w-7 h-7 shrink-0 rounded-full border flex items-center justify-center text-xs font-semibold",
                    stateColor[step.state]
                  )}
                >
                  {step.state === "done" ? "✓" : step.state === "failed" ? "!" : step.state === "running" ? <RunningDot /> : index + 1}
                </span>
                <div className="flex-1 flex items-center">
                  <div
                    className={clsx(
                      "h-px flex-1",
                      isLast
                        ? "invisible"
                        : steps[index + 1].state === "pending" || step.state === "pending"
                          ? connectorColor.pending
                          : connectorColor[step.state]
                    )}
                  />
                </div>
              </div>
              <div className="mt-2 px-1 text-center min-w-0">
                <div
                  className={clsx(
                    "text-[12px] font-medium truncate",
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
                  <div className="text-[11px] text-[var(--color-text-dim)] truncate">{step.detail}</div>
                ) : null}
              </div>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function RunningDot() {
  return (
    <span className="relative flex w-2 h-2">
      <span className="absolute inset-0 rounded-full bg-[var(--color-accent)] animate-ping opacity-60" />
      <span className="relative inline-flex w-2 h-2 rounded-full bg-[var(--color-accent)]" />
    </span>
  );
}
