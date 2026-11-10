// Hex values mirrored from dataviz's references/palette.md (validated
// categorical order + chart chrome). Kept as plain hex here because chart
// libraries (Recharts) need literal color strings, not CSS custom
// properties, for stroke/fill props.

import { engineLabelForAgent } from "./types";

export const palette = {
  light: {
    surface: "#fcfcfb",
    ink: "#0b0b0b",
    inkSecondary: "#52514e",
    inkMuted: "#898781",
    grid: "#e1e0d9",
    baseline: "#c3c2b7",
    series1: "#2a78d6", // blue    -- R
    series2: "#eb6834", // orange  -- RV
    series3: "#1baf7a", // aqua    -- ML / sale price (different screens, no shared legend)
    series4: "#eda100", // yellow  -- RL / goods cost (different screens, no shared legend)
    series5: "#e87ba4", // magenta -- OPT2
    series6: "#008300", // green   -- IS
    series7: "#4a3aa7", // violet  -- OPT
    series8: "#e34948", // red     -- sole overflow slot (2nd+ instance of same architecture)
    good: "#0ca30c",
    warning: "#fab219", // mirrors tailwind.config.ts's status.warning -- single value, no light/dark split
    critical: "#d03b3b",
    winner: "#ffcc33", // gold -- the Gale-Shapley match actually kept, layered on top of "good" (see NegotiationGraph.tsx)
  },
  dark: {
    surface: "#1a1a19",
    ink: "#ffffff",
    inkSecondary: "#c3c2b7",
    inkMuted: "#898781",
    grid: "#2c2c2a",
    baseline: "#383835",
    series1: "#3987e5",
    series2: "#d95926",
    series3: "#199e70",
    series4: "#c98500",
    series5: "#d55181",
    series6: "#008300",
    series7: "#9085e9",
    series8: "#e66767",
    good: "#0ca30c",
    warning: "#fab219",
    critical: "#e66767",
    winner: "#ffd766",
  },
} as const;

export type PaletteMode = keyof typeof palette;
type SeriesKey = "series1" | "series2" | "series3" | "series4" | "series5" | "series6" | "series7" | "series8";

// One base hue per architecture, assigned in a fixed order so the same
// architecture always reads as the same color everywhere in the dashboard
// (charts, badges, legends). The validated categorical palette is exactly
// 8 slots (see the dataviz skill's references/palette.md) -- an 8th
// architecture is deliberately NOT an invented 9th hue (that's an explicit
// anti-pattern). ML and RL reuse series3/series4, promoted from
// "non-architecture use" -- neither ever shares a legend with the
// sale-price/goods-cost lines that also use those slots (different
// screens), and every badge/button using these colors always carries the
// architecture's text label too, so identity is never color-alone. ML2 and
// RL2 each deliberately reuse their "parent" architecture's own slot
// (series3, series4) rather than claiming the last remaining slot
// (series8, reserved for same-architecture overflow): ML2 IS ML plus a
// learned accept decision, and RL2 IS RL retrained for competitive
// awareness -- neither is an unrelated architecture, so sharing the hue is
// the honest choice -- `isDashedArch` below marks both for a dashed
// stroke/border wherever that distinction matters visually (see
// PriceTrajectoryChart.tsx, ArchBadge.tsx), on top of the text label every
// use site already carries.
const ARCH_BASE_SLOT: Record<string, SeriesKey> = {
  R: "series1",
  RV: "series2",
  IS: "series6",
  OPT: "series7",
  OPT2: "series5",
  ML: "series3",
  ML2: "series3",
  RL: "series4",
  RL2: "series4",
};
const OVERFLOW_SLOTS: SeriesKey[] = ["series8"];

/** True for any architecture that shares its base hue with another one and
 * needs a dashed stroke/border to stay visually distinct (ML2 sharing ML's
 * series3, RL2 sharing RL's series4 -- see ARCH_BASE_SLOT's comment above). */
export function isDashedArch(archLabel: string): boolean {
  return archLabel === "ML2" || archLabel === "RL2";
}

/** Base color for one architecture label ("R"|"RV"|"IS"|"OPT"|"OPT2"|"ML"|"ML2"|"RL"),
 * for small UI elements (badges) that don't have the full carrier-id list handy. */
export function archColor(archLabel: string, mode: PaletteMode): string {
  const c = palette[mode];
  // Anything that is not a reference architecture is a community agent: it gets the
  // red overflow slot so "your agent" stands out against the nine baselines.
  return c[ARCH_BASE_SLOT[archLabel] ?? "series8"];
}

/**
 * Deterministic per-agent line color: tied to (architecture, instance
 * number within that architecture), not to array position -- so the same
 * carrier or shipper always gets the same color across different rounds,
 * and a 2nd instance of the same architecture is visually distinct from
 * the 1st rather than silently sharing its color. Role-agnostic: pass
 * either a carrier-id list or a shipper-id list, they share one convention.
 */
export function agentColor(agentIds: string[], agentId: string, mode: PaletteMode): string {
  const c = palette[mode];
  const arch = engineLabelForAgent(agentId);
  const sameArchBefore = agentIds.slice(0, agentIds.indexOf(agentId)).filter((id) => engineLabelForAgent(id) === arch).length;
  if (sameArchBefore === 0) return archColor(arch, mode);
  return c[OVERFLOW_SLOTS[(sameArchBefore - 1) % OVERFLOW_SLOTS.length]];
}
