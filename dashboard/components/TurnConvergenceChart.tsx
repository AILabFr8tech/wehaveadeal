"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { palette } from "@/lib/colors";
import { num } from "@/lib/format";
import { usePrefersDark } from "@/lib/usePrefersDark";
import type { TableResult } from "@/lib/types";

const TURNS_MAX = 12; // 6 per side, WHAD spec §3

export function TurnConvergenceChart({ tables }: { tables: TableResult[] }) {
  const dark = usePrefersDark();
  const c = dark ? palette.dark : palette.light;

  // Average offer per side at each global turn, across all 12 tables of this
  // negotiation. Unlike the round-level carrier chart, a null here just
  // means "not this side's turn" (shippers only offer on odd turns, carriers
  // only on even ones) -- structural, not missing information -- so we
  // connect across it (connectNulls) instead of leaving a gap.
  const data = useMemo(() => {
    const rows = [];
    for (let turn = 1; turn <= TURNS_MAX; turn++) {
      const actor = turn % 2 === 1 ? "shipper" : "carrier";
      const amounts = tables.flatMap((t) =>
        t.history.filter((r) => r.turn === turn && r.actor === actor && r.action === "offer" && r.amount != null).map((r) => r.amount as number)
      );
      const avg = amounts.length > 0 ? amounts.reduce((a, b) => a + b, 0) / amounts.length : null;
      rows.push({
        turn: turn,
        shipper: actor === "shipper" ? avg : null,
        carrier: actor === "carrier" ? avg : null,
        n_mesas: amounts.length,
      });
    }
    return rows;
  }, [tables]);

  return (
    <div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis
            dataKey="turn"
            stroke={c.baseline}
            tick={{ fill: c.inkMuted, fontSize: 11 }}
            label={{ value: "turn", position: "insideBottom", offset: -2, fill: c.inkMuted, fontSize: 11 }}
            allowDecimals={false}
          />
          <YAxis stroke={c.baseline} tick={{ fill: c.inkMuted, fontSize: 11 }} width={64} tickFormatter={(v) => `$${v}`} />
          <Tooltip
            contentStyle={{
              background: c.surface,
              border: `1px solid ${c.grid}`,
              borderRadius: 4,
              fontSize: 12,
              fontFamily: "ui-monospace, monospace",
            }}
            labelFormatter={(v) => `turn ${v}`}
            formatter={(value: any, name: any, item: any) => {
              if (value === null || value === undefined) return ["—", name];
              const n = item?.payload?.n_mesas;
              return [`$${num(Math.round(value as number))} (mean of ${n} tables)`, name];
            }}
          />
          <Legend wrapperStyle={{ fontSize: 12, fontFamily: "ui-monospace, monospace" }} />
          <Line
            type="monotone"
            dataKey="shipper"
            name="mean shipper offer"
            stroke={c.series3}
            strokeWidth={2}
            dot={{ r: 4 }}
            connectNulls
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="carrier"
            name="mean carrier counter-offer"
            stroke={c.series4}
            strokeWidth={2}
            dot={{ r: 4 }}
            connectNulls
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="text-xs text-ink-muted font-mono mt-xs">
        mean of the offers/counter-offers of the 12 tables at each turn · tables that have already closed stop contributing, so
        the number of tables behind the mean shrinks as turns go by
      </p>
    </div>
  );
}
