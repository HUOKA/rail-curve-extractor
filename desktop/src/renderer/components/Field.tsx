import React from "react";
import clsx from "clsx";

type FieldProps = {
  label: string;
  hint?: React.ReactNode;
  required?: boolean;
  className?: string;
  children: React.ReactNode;
};

export function Field({ label, hint, required, className, children }: FieldProps) {
  return (
    <label className={clsx("flex flex-col gap-1.5", className)}>
      <span className="flex items-center gap-1.5 text-xs font-medium text-[var(--color-text-muted)]">
        {label}
        {required ? <span className="text-[var(--color-warn)]">*</span> : null}
      </span>
      {children}
      {hint ? <span className="text-[11px] text-[var(--color-text-dim)] leading-relaxed">{hint}</span> : null}
    </label>
  );
}

export const inputClass = clsx(
  "h-9 w-full rounded-md px-3 text-sm",
  "bg-[var(--color-surface)] border border-[var(--color-border)]",
  "text-[var(--color-text)] placeholder:text-[var(--color-text-dim)]",
  "transition-colors duration-150",
  "hover:border-[var(--color-border-strong)]",
  "focus:outline-none focus:border-[var(--color-accent)] focus:ring-1 focus:ring-[var(--color-accent)]/30",
  "disabled:bg-[var(--color-surface-2)] disabled:text-[var(--color-text-dim)] disabled:cursor-not-allowed"
);

export const monoInputClass = clsx(inputClass, "font-mono text-[12.5px]");

type TextInputProps = React.InputHTMLAttributes<HTMLInputElement> & {
  mono?: boolean;
};

export function TextInput({ mono, className, ...rest }: TextInputProps) {
  return <input {...rest} className={clsx(mono ? monoInputClass : inputClass, className)} />;
}
