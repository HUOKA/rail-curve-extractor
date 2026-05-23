import React from "react";
import clsx from "clsx";

type CardProps = {
  title?: string;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  children: React.ReactNode;
};

export function Card({ title, subtitle, actions, className, bodyClassName, children }: CardProps) {
  return (
    <section
      className={clsx(
        "flex flex-col rounded-lg border border-[var(--color-border)]",
        "bg-[var(--color-surface)]",
        className
      )}
    >
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 px-4 py-3 border-b border-[var(--color-border)]">
          <div className="min-w-0">
            {title ? (
              <h2 className="text-sm font-semibold text-[var(--color-text)] tracking-tight">{title}</h2>
            ) : null}
            {subtitle ? (
              <p className="mt-0.5 text-xs text-[var(--color-text-muted)] leading-relaxed">{subtitle}</p>
            ) : null}
          </div>
          {actions ? <div className="flex items-center gap-1.5 shrink-0">{actions}</div> : null}
        </header>
      )}
      <div className={clsx("p-4", bodyClassName)}>{children}</div>
    </section>
  );
}
