"use client";

import { archColor, isDashedArch } from "@/lib/colors";
import { usePrefersDark } from "@/lib/usePrefersDark";

/** Small colored badge for an architecture label (R/RV/IS/OPT/OPT2/ML/ML2/RL),
 * consistent everywhere it appears (history, leaderboard, table cards).
 * ML2 shares ML's color (see lib/colors.ts's ARCH_BASE_SLOT) so it gets a
 * dashed border on top of its own text label to stay visually distinct. */
export function ArchBadge({ archLabel, prefix = "", className = "" }: { archLabel: string; prefix?: string; className?: string }) {
  const dark = usePrefersDark();
  const color = archColor(archLabel, dark ? "dark" : "light");
  const dashed = isDashedArch(archLabel);
  return (
    <span
      className={`text-xs px-xs rounded ${className}`}
      style={{
        background: `${color}26`,
        color,
        border: dashed ? `1px dashed ${color}` : undefined,
      }}
    >
      {prefix}
      {archLabel}
    </span>
  );
}
