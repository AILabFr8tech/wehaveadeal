"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { agentColor, isDashedArch, palette, type PaletteMode } from "@/lib/colors";
import { num } from "@/lib/format";
import { usePrefersDark } from "@/lib/usePrefersDark";
import { engineLabelForAgent } from "@/lib/types";
import type { NegotiationResult } from "@/lib/types";

type View = "shipper" | "carrier" | "zopa";

function ChartTooltip({ active, payload, label }: any) {
  const dark = usePrefersDark();
  const c = dark ? palette.dark : palette.light;
  if (!active || !payload?.length) return null;
  const rows = payload.filter((p: any) => !String(p.dataKey).endsWith("__sin"));
  if (rows.length === 0) return null;
  return (
    <div
      style={{
        background: c.surface,
        border: `1px solid ${c.grid}`,
        borderRadius: 4,
        padding: "8px 10px",
        fontSize: 12,
        fontFamily: "ui-monospace, monospace",
      }}
    >
      <div style={{ color: c.inkMuted, marginBottom: 4 }}>negotiation {label}</div>
      {rows.map((p: any) => (
        <div key={p.dataKey} style={{ color: p.color ?? c.ink }}>
          {p.name}: {p.value === null || p.value === undefined ? "unmatched" : `$${num(Math.round(p.value as number))}`}
        </div>
      ))}
    </div>
  );
}

export function PriceTrajectoryChart({
  negotiations,
  roundFile,
  carrierIds,
}: {
  negotiations: NegotiationResult[];
  roundFile: string;
  carrierIds: string[];
}) {
  const router = useRouter();
  const dark = usePrefersDark();
  const mode: PaletteMode = dark ? "dark" : "light";
  const c = palette[mode];
  const [view, setView] = useState<View>("shipper");
  const [seleccionados, setSeleccionados] = useState<Set<string>>(new Set(carrierIds));

  const shipperData = useMemo(
    () =>
      negotiations.map((n) => ({
        n: n.negotiation_index,
        sale_price: Math.round(n.sale_price),
        goods_cost: Math.round(n.goods_cost),
      })),
    [negotiations]
  );

  // Per-carrier "agreed rate": the rate that carrier actually got matched
  // at in each negotiation (from carrier_outcomes, computed by the matching
  // module). null when not matched -- Recharts renders that as a gap rather
  // than a fake interpolation. The parallel `${id}__sin` field is 0 exactly
  // when the carrier WASN'T matched, rendered as a small marker pinned to
  // the y=0 baseline: it turns "empty space" into an explicit "this
  // negotiation happened, this carrier just didn't win it" signal, without
  // inventing a price for the gap.
  const carrierData = useMemo(
    () =>
      negotiations.map((n) => {
        const row: Record<string, number | null> = { n: n.negotiation_index };
        for (const carrierId of carrierIds) {
          const outcome = n.carrier_outcomes.find((o) => o.carrier_id === carrierId);
          row[carrierId] = outcome?.matched ? Math.round(outcome.rate ?? 0) : null;
          row[`${carrierId}__sin`] = outcome?.matched ? null : 0;
        }
        return row;
      }),
    [negotiations, carrierIds]
  );

  // ZOPA (zone of possible agreement): the client's shipper_ceiling (RP; exact
  // -- sale_price and goods_cost are shared by all 4 shippers in a
  // negotiation, so RP is a single number) against the OUTER range of that
  // day's carrier bands (min of every route's floor to max of every route's
  // ceiling, since each of the 4 shippers was assigned a different
  // route/band that day). If the RP line dips below the band, no legal deal
  // existed with ANY carrier that day -- this is exactly what triggers
  // ShipperR's deliberate walk-away.
  //
  // The band is drawn as two dashed boundary Lines (band_low/band_high)
  // rather than a filled Area: recharts@2.15's <Area> silently renders
  // nothing in this app (no console error/warning either, in dev or prod --
  // likely a React 19 incompatibility, since recharts 2.x predates React 19
  // and Area may rely on function-component defaultProps, which React 19
  // stopped applying). <Line> is unaffected, so boundary lines are the
  // reliable choice here.
  const zopaData = useMemo(
    () =>
      negotiations.map((n) => {
        const shipper_ceiling = Math.round(n.sale_price - n.goods_cost);
        const bands = Object.values(n.routes).map((r) => ({
          low: r.refprice_mean - 0.5 * r.refprice_sigma,
          high: r.refprice_mean + 0.5 * r.refprice_sigma,
        }));
        const band_low = Math.round(Math.min(...bands.map((b) => b.low)));
        const band_high = Math.round(Math.max(...bands.map((b) => b.high)));
        return { n: n.negotiation_index, shipper_ceiling, band_low, band_high };
      }),
    [negotiations]
  );

  function toggleCarrier(id: string) {
    setSeleccionados((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const data = view === "shipper" ? shipperData : view === "carrier" ? carrierData : zopaData;

  return (
    <div>
      <div className="flex items-center justify-between flex-wrap gap-md mb-md">
        <div className="inline-flex rounded border border-line-light dark:border-line-dark overflow-hidden font-mono text-xs">
          {(
            [
              ["shipper", "shipper side"],
              ["carrier", "carrier side"],
              ["zopa", "zone of agreement (ZOPA)"],
            ] as [View, string][]
          ).map(([key, label], i) => (
            <button
              key={key}
              onClick={() => setView(key)}
              className={
                "px-md py-xs " +
                (i > 0 ? "border-l border-line-light dark:border-line-dark " : "") +
                (view === key
                  ? "bg-series-1-light/15 text-series-1-light dark:bg-series-1-dark/20 dark:text-series-1-dark"
                  : "text-ink-muted hover:bg-line-light/30 dark:hover:bg-line-dark/30")
              }
            >
              {label}
            </button>
          ))}
        </div>

        {view === "carrier" && (
          <div className="flex items-center gap-md flex-wrap font-mono text-xs">
            {carrierIds.map((id) => {
              const color = agentColor(carrierIds, id, mode);
              const checked = seleccionados.has(id);
              return (
                <label key={id} className="flex items-center gap-xs cursor-pointer select-none">
                  <input type="checkbox" checked={checked} onChange={() => toggleCarrier(id)} style={{ accentColor: color }} />
                  <span className="inline-block w-2 h-2 rounded-full" style={{ background: color, opacity: checked ? 1 : 0.3 }} />
                  <span className={checked ? "" : "text-ink-muted line-through"}>
                    {id} ({engineLabelForAgent(id)})
                  </span>
                </label>
              );
            })}
          </div>
        )}
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart
          data={data}
          margin={{ top: 8, right: 12, bottom: 0, left: 0 }}
          onClick={(e) => {
            const idx = e?.activePayload?.[0]?.payload?.n;
            if (idx !== undefined) router.push(`/round/${roundFile}/negotiation/${idx}`);
          }}
        >
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis
            dataKey="n"
            stroke={c.baseline}
            tick={{ fill: c.inkMuted, fontSize: 11 }}
            label={{ value: "negotiation #", position: "insideBottom", offset: -2, fill: c.inkMuted, fontSize: 11 }}
          />
          <YAxis
            stroke={c.baseline}
            tick={{ fill: c.inkMuted, fontSize: 11 }}
            width={64}
            domain={[0, "auto"]}
            tickFormatter={(v) => `$${v}`}
          />
          <Tooltip content={<ChartTooltip />} />
          <Legend wrapperStyle={{ fontSize: 12, fontFamily: "ui-monospace, monospace" }} />

          {view === "shipper" && [
            <Line
              key="sale_price"
              type="monotone"
              dataKey="sale_price"
              name="sale price"
              stroke={c.series3}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
            />,
            <Line
              key="goods_cost"
              type="monotone"
              dataKey="goods_cost"
              name="goods cost"
              stroke={c.series4}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
            />,
          ]}

          {view === "carrier" &&
            carrierIds
              .filter((id) => seleccionados.has(id))
              .flatMap((id) => {
                const color = agentColor(carrierIds, id, mode);
                const dashed = isDashedArch(engineLabelForAgent(id));
                return [
                  <Line
                    key={id}
                    type="monotone"
                    dataKey={id}
                    name={`${id} — agreed rate`}
                    stroke={color}
                    strokeWidth={2}
                    strokeDasharray={dashed ? "6 3" : undefined}
                    dot={false}
                    activeDot={{ r: 4 }}
                    connectNulls={false}
                    isAnimationActive={false}
                  />,
                  <Scatter
                    key={`${id}__sin`}
                    dataKey={`${id}__sin`}
                    name={`${id} — unmatched`}
                    legendType="none"
                    isAnimationActive={false}
                    shape={(props: any) => (
                      <circle cx={props.cx} cy={props.cy} r={3} fill={color} fillOpacity={0.4} />
                    )}
                  />,
                ];
              })}

          {view === "zopa" && [
            <Line
              key="band_high"
              type="monotone"
              dataKey="band_high"
              name="carrier band — ceiling"
              stroke={c.series1}
              strokeWidth={1.5}
              strokeDasharray="4 3"
              dot={false}
              activeDot={{ r: 3 }}
              isAnimationActive={false}
            />,
            <Line
              key="band_low"
              type="monotone"
              dataKey="band_low"
              name="carrier band — floor"
              stroke={c.series1}
              strokeWidth={1.5}
              strokeDasharray="4 3"
              dot={false}
              activeDot={{ r: 3 }}
              isAnimationActive={false}
            />,
            <Line
              key="shipper_ceiling"
              type="monotone"
              dataKey="shipper_ceiling"
              name="shipper RP (ceiling)"
              stroke={c.series3}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
            />,
          ]}
        </ComposedChart>
      </ResponsiveContainer>
      <p className="text-xs text-ink-muted font-mono mt-xs">
        {view === "shipper" &&
          "sale price and goods cost set the shipper's RP (its ceiling) · a mean-reverting process, not a naive random walk"}
        {view === "carrier" &&
          "rate actually agreed by each carrier when matched · a point on the baseline = that negotiation happened but the carrier did not win it (no unexplained gaps, and no invented price)"}
        {view === "zopa" &&
          "band = range between the lowest and highest route bands of that day · if the shipper RP falls below the band, no legal deal was possible with any carrier"}
        {" · "}click a point to open that negotiation
      </p>
    </div>
  );
}
