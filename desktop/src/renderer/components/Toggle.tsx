import React from "react";
import clsx from "clsx";

type Props = {
  checked: boolean;
  onChange: (next: boolean) => void;
  label?: string;
  hint?: string;
  disabled?: boolean;
};

export function Toggle({ checked, onChange, label, hint, disabled }: Props) {
  return (
    <label
      className={clsx(
        "flex items-center gap-3 cursor-pointer select-none",
        disabled && "cursor-not-allowed opacity-60"
      )}
    >
      <span
        role="switch"
        aria-checked={checked}
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === " " || event.key === "Enter") {
            event.preventDefault();
            if (!disabled) onChange(!checked);
          }
        }}
        onClick={() => {
          if (!disabled) onChange(!checked);
        }}
        className={clsx(
          "relative inline-flex h-5 w-9 shrink-0 rounded-full transition-colors duration-150",
          "border",
          checked
            ? "bg-[var(--color-accent-strong)] border-[var(--color-accent-strong)]"
            : "bg-[var(--color-surface-2)] border-[var(--color-border)]"
        )}
      >
        <span
          className={clsx(
            "absolute top-0.5 h-4 w-4 rounded-full bg-[var(--color-on-accent)] transition-transform duration-150 shadow",
            checked ? "translate-x-[18px]" : "translate-x-0.5"
          )}
        />
      </span>
      {label ? (
        <span className="flex flex-col gap-0.5">
          <span className="text-sm text-[var(--color-text)]">{label}</span>
          {hint ? <span className="text-[11px] text-[var(--color-text-dim)]">{hint}</span> : null}
        </span>
      ) : null}
    </label>
  );
}
