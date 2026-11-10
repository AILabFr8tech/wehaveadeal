import type { CarrierStanding, ShipperStanding } from "@/lib/types";
import { engineLabelForAgent } from "@/lib/types";
import { ArchBadge } from "@/components/ArchBadge";

import { num as fmt } from "@/lib/format";

export function ShipperLeaderboard({
  standings,
  winner,
}: {
  standings: Record<string, ShipperStanding>;
  winner: string | null;
}) {
  const rows = Object.entries(standings).sort((a, b) => b[1].points - a[1].points);
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-xs font-mono text-ink-muted border-b border-line-light dark:border-line-dark">
          <th className="py-xs pr-md font-normal">shipper</th>
          <th className="py-xs pr-md font-normal">engine</th>
          <th className="py-xs pr-md font-normal text-right">points</th>
          <th className="py-xs pr-md font-normal text-right">beneficio</th>
          <th className="py-xs pr-md font-normal text-right">closed</th>
          <th className="py-xs pr-md font-normal text-right">matched</th>
        </tr>
      </thead>
      <tbody className="font-mono tabular">
        {rows.map(([id, s]) => {
          const engine = engineLabelForAgent(id);
          return (
            <tr key={id} className="border-b border-line-light dark:border-line-dark last:border-0">
              <td className="py-xs pr-md">
                {id}
                {id === winner && <span className="ml-xs text-status-good">★ round winner</span>}
                {s.got_bonus && <span className="ml-xs text-ink-muted text-xs">(+bono)</span>}
              </td>
              <td className="py-xs pr-md">
                <ArchBadge archLabel={engine} prefix="Shipper" />
              </td>
              <td className="py-xs pr-md text-right">{fmt(s.points, 1)}</td>
              <td className="py-xs pr-md text-right">${fmt(s.profit)}</td>
              <td className="py-xs pr-md text-right">{s.closes}</td>
              <td className="py-xs pr-md text-right">{s.matches}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export function CarrierLeaderboard({
  standings,
  winner,
}: {
  standings: Record<string, CarrierStanding>;
  winner: string | null;
}) {
  const rows = Object.entries(standings).sort((a, b) => b[1].margin - a[1].margin);
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-xs font-mono text-ink-muted border-b border-line-light dark:border-line-dark">
          <th className="py-xs pr-md font-normal">carrier</th>
          <th className="py-xs pr-md font-normal">engine</th>
          <th className="py-xs pr-md font-normal text-right">margin</th>
          <th className="py-xs pr-md font-normal text-right">occupancy</th>
          <th className="py-xs pr-md font-normal text-right">matched</th>
        </tr>
      </thead>
      <tbody className="font-mono tabular">
        {rows.map(([id, s]) => {
          const engine = engineLabelForAgent(id);
          return (
            <tr key={id} className="border-b border-line-light dark:border-line-dark last:border-0">
              <td className="py-xs pr-md">
                {id}
                {id === winner && <span className="ml-xs text-status-good">★ round winner</span>}
              </td>
              <td className="py-xs pr-md">
                <ArchBadge archLabel={engine} prefix="Carrier" />
              </td>
              <td className="py-xs pr-md text-right">${fmt(s.margin)}</td>
              <td className="py-xs pr-md text-right">{fmt(s.occupancy * 100, 0)}%</td>
              <td className="py-xs pr-md text-right">{s.matches}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
