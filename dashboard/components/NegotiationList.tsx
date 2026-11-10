"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type { NegotiationResult } from "@/lib/types";
import { num } from "@/lib/format";

export function NegotiationList({
  negotiations,
  roundFile,
}: {
  negotiations: NegotiationResult[];
  roundFile: string;
}) {
  const [filter, setFilter] = useState("");

  const rows = useMemo(() => {
    return negotiations
      .map((n) => ({
        idx: n.negotiation_index,
        sale_price: n.sale_price,
        goods_cost: n.goods_cost,
        matches: n.matching.matches.length,
        unmatched: n.matching.unmatched_shippers.length,
        closes: n.tables.filter((t) => t.closed).length,
      }))
      .filter((r) => (filter.trim() === "" ? true : r.idx.toString().includes(filter.trim())));
  }, [negotiations, filter]);

  return (
    <div>
      <input
        className="w-full mb-sm bg-surface-light dark:bg-surface-dark border border-line-light dark:border-line-dark rounded px-sm py-xs font-mono text-xs placeholder:text-ink-muted"
        placeholder="filter by negotiation number…"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
      />
      <div className="max-h-[420px] overflow-y-auto border border-line-light dark:border-line-dark rounded">
        <table className="w-full text-xs font-mono tabular">
          <thead className="sticky top-0 bg-surface-light dark:bg-surface-dark">
            <tr className="text-left text-ink-muted border-b border-line-light dark:border-line-dark">
              <th className="py-xs px-sm font-normal">#</th>
              <th className="py-xs px-sm font-normal text-right">sale price</th>
              <th className="py-xs px-sm font-normal text-right">goods cost</th>
              <th className="py-xs px-sm font-normal text-right">closed</th>
              <th className="py-xs px-sm font-normal text-right">matched</th>
              <th className="py-xs px-sm font-normal text-right">no transport</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.idx} className="border-b border-line-light dark:border-line-dark last:border-0 hover:bg-line-light/40 dark:hover:bg-line-dark/40">
                <td className="py-xs px-sm">
                  <Link className="text-series-1-light dark:text-series-1-dark hover:underline" href={`/round/${roundFile}/negotiation/${r.idx}`}>
                    {r.idx}
                  </Link>
                </td>
                <td className="py-xs px-sm text-right">${num(Math.round(r.sale_price))}</td>
                <td className="py-xs px-sm text-right">${num(Math.round(r.goods_cost))}</td>
                <td className="py-xs px-sm text-right">{r.closes}/12</td>
                <td className="py-xs px-sm text-right">{r.matches}/3</td>
                <td className="py-xs px-sm text-right text-status-critical">{r.unmatched}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
