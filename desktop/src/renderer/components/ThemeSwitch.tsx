import React from "react";
import clsx from "clsx";
import { Sun, Moon, Monitor } from "lucide-react";
import { motion } from "framer-motion";
import { ThemeMode } from "../lib/theme";

type Props = {
  mode: ThemeMode;
  onChange: (next: ThemeMode) => void;
};

const OPTIONS: Array<{ value: ThemeMode; label: string; Icon: typeof Sun }> = [
  { value: "light", label: "浅色", Icon: Sun },
  { value: "system", label: "跟随系统", Icon: Monitor },
  { value: "dark", label: "深色", Icon: Moon }
];

export function ThemeSwitch({ mode, onChange }: Props) {
  return (
    <div
      role="group"
      aria-label="主题模式"
      className="inline-flex p-0.5 h-7 rounded-md border border-[var(--color-border)] bg-[var(--color-surface-2)] relative"
    >
      {OPTIONS.map((option) => {
        const active = mode === option.value;
        const { Icon } = option;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            title={option.label}
            data-active={active}
            className={clsx(
              "relative inline-flex items-center justify-center w-7 h-6 rounded-[4px] transition-colors z-10",
              active
                ? "text-[var(--color-text)]"
                : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
            )}
          >
            {active ? (
              <motion.span
                layoutId="theme-pill"
                className="absolute inset-0 rounded-[4px] bg-[var(--color-surface)] shadow-sm border border-[var(--color-border)]"
                transition={{ type: "spring", stiffness: 380, damping: 32 }}
              />
            ) : null}
            <Icon size={13} strokeWidth={2} className="relative" />
            <span className="sr-only">{option.label}</span>
          </button>
        );
      })}
    </div>
  );
}
