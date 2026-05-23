import React, { useEffect, useState } from "react";
import { animate, useMotionValue, useTransform, motion } from "framer-motion";

type Props = {
  value: number | null | undefined;
  fractionDigits?: number;
  suffix?: string;
  emptyText?: string;
  className?: string;
};

/** Smoothly interpolate to a numeric value. Used for percent counters. */
export function AnimatedNumber({
  value,
  fractionDigits = 1,
  suffix = "",
  emptyText = "—",
  className
}: Props) {
  const motionValue = useMotionValue(typeof value === "number" && Number.isFinite(value) ? value : 0);
  const display = useTransform(motionValue, (latest) =>
    `${latest.toFixed(fractionDigits)}${suffix}`
  );
  const [isEmpty, setIsEmpty] = useState<boolean>(
    !(typeof value === "number" && Number.isFinite(value))
  );

  useEffect(() => {
    if (typeof value === "number" && Number.isFinite(value)) {
      setIsEmpty(false);
      const controls = animate(motionValue, value, {
        type: "tween",
        ease: "easeOut",
        duration: 0.45
      });
      return () => controls.stop();
    }
    setIsEmpty(true);
  }, [value, motionValue]);

  if (isEmpty) {
    return <span className={className}>{emptyText}</span>;
  }
  return <motion.span className={className}>{display}</motion.span>;
}
