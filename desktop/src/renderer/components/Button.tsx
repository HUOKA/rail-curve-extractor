import React from "react";
import clsx from "clsx";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  icon?: React.ReactNode;
};

const variantClass: Record<Variant, string> = {
  primary:
    "bg-[var(--color-accent-strong)] text-[var(--color-on-accent)] hover:bg-[var(--color-accent)] disabled:bg-[var(--color-surface-3)] disabled:text-[var(--color-text-dim)]",
  secondary:
    "bg-[var(--color-surface-2)] text-[var(--color-text)] border border-[var(--color-border)] hover:bg-[var(--color-surface-3)] hover:border-[var(--color-border-strong)] disabled:bg-[var(--color-surface)] disabled:text-[var(--color-text-dim)]",
  ghost:
    "bg-transparent text-[var(--color-text-muted)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-text)] disabled:text-[var(--color-text-dim)]",
  danger:
    "bg-[var(--color-danger)]/10 text-[var(--color-danger)] border border-[var(--color-danger)]/30 hover:bg-[var(--color-danger)]/20"
};

const sizeClass: Record<Size, string> = {
  sm: "h-7 px-2.5 text-xs gap-1.5",
  md: "h-9 px-3.5 text-sm gap-2"
};

export function Button({
  variant = "secondary",
  size = "md",
  loading,
  icon,
  className,
  disabled,
  children,
  ...rest
}: Props) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={clsx(
        "inline-flex items-center justify-center rounded-md font-medium whitespace-nowrap",
        "transition-colors duration-150 select-none",
        "disabled:cursor-not-allowed",
        variantClass[variant],
        sizeClass[size],
        className
      )}
    >
      {loading ? <Spinner /> : icon}
      {children}
    </button>
  );
}

function Spinner() {
  return (
    <span
      className="inline-block w-3 h-3 rounded-full border-2 border-current border-r-transparent animate-spin"
      aria-hidden
    />
  );
}
