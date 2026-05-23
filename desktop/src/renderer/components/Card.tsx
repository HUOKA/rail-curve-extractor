import React from "react";
import clsx from "clsx";

type CardProps = {
  title?: string;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  interactive?: boolean;
  children: React.ReactNode;
};

export function Card({
  title,
  subtitle,
  actions,
  className,
  bodyClassName,
  interactive,
  children
}: CardProps) {
  return (
    <section
      className={clsx(
        "flex flex-col rounded-lg border border-[var(--color-border)]",
        "bg-[var(--color-surface)] shadow-card",
        "transition-shadow duration-200",
        interactive && "hover:shadow-card-hover",
        className
      )}
    >
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 px-4 py-2.5 border-b border-[var(--color-border)]">
          <div className="min-w-0">
            {title ? (
              <h2 className="flex items-center gap-2 text-[13px] font-semibold text-[var(--color-text)] tracking-tight uppercase">
                <span className="inline-block w-1 h-3 rounded-sm bg-[var(--color-accent-strong)] shrink-0" />
                {title}
              </h2>
            ) : null}
            {subtitle ? (
              <p className="mt-1 text-xs text-[var(--color-text-muted)] leading-relaxed">{subtitle}</p>
            ) : null}
          </div>
          {actions ? <div className="flex items-center gap-1.5 shrink-0">{actions}</div> : null}
        </header>
      )}
      <div className={clsx("p-4", bodyClassName)}>{children}</div>
    </section>
  );
}
