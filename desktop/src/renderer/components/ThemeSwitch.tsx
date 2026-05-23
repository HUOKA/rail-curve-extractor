import React from "react";
import clsx from "clsx";
import { ThemeMode } from "../lib/theme";

type Props = {
  mode: ThemeMode;
  onChange: (next: ThemeMode) => void;
};

export function ThemeSwitch({ mode, onChange }: Props) {
  return (
    <div
      role="group"
      aria-label="主题模式"
      className="inline-flex p-0.5 h-7 rounded-md border border-[var(--color-border)] bg-[var(--color-surface-2)]"
    >
      <ThemeButton active={mode === "light"} onClick={() => onChange("light")} label="浅色">
        <SunIcon />
      </ThemeButton>
      <ThemeButton active={mode === "system"} onClick={() => onChange("system")} label="跟随系统">
        <SystemIcon />
      </ThemeButton>
      <ThemeButton active={mode === "dark"} onClick={() => onChange("dark")} label="深色">
        <MoonIcon />
      </ThemeButton>
    </div>
  );
}

function ThemeButton({
  active,
  onClick,
  label,
  children
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      data-active={active}
      className={clsx(
        "inline-flex items-center justify-center w-7 h-6 rounded-[4px] transition-colors",
        active
          ? "bg-[var(--color-surface)] text-[var(--color-text)] shadow-sm"
          : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
      )}
    >
      {children}
      <span className="sr-only">{label}</span>
    </button>
  );
}

function SunIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M21 12.79A9 9 0 1 1 11.21 3a7 7 0 0 0 9.79 9.79z" />
    </svg>
  );
}

function SystemIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="3" y="4" width="18" height="13" rx="2" />
      <path d="M8 21h8M12 17v4" />
    </svg>
  );
}
