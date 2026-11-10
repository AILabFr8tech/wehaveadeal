import type { MatchingResult, TableResult } from "@/lib/types";
import { money } from "@/lib/format";

function rateFor(tables: TableResult[], shipperId: string, carrierId: string): number | null {
  const t = tables.find((t) => t.shipper_id === shipperId && t.carrier_id === carrierId);
  return t?.closed ? t.rate : null;
}

export function MatchingPanel({
  matching,
  tables,
  shipperIds,
  carrierIds,
}: {
  matching: MatchingResult;
  tables: TableResult[];
  shipperIds: string[];
  carrierIds: string[];
}) {
  const matchedCarrierOf: Record<string, string> = {};
  const matchedShipperOf: Record<string, string> = {};
  for (const m of matching.matches) {
    matchedCarrierOf[m.shipper_id] = m.carrier_id;
    matchedShipperOf[m.carrier_id] = m.shipper_id;
  }
  const unmatchedCarriers = carrierIds.filter((c) => !matchedShipperOf[c]);
  const diff = shipperIds.length - carrierIds.length;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-lg">
      <div>
        <h3 className="font-mono text-xs text-ink-muted mb-sm">
          shipper preferences (ascending rate — cheapest first)
        </h3>
        <div className="space-y-xs">
          {shipperIds.map((s) => {
            const prefs = matching.shipper_preferences[s] ?? [];
            const matchedTo = matchedCarrierOf[s];
            return (
              <div key={s} className="font-mono text-sm flex items-baseline gap-sm flex-wrap">
                <span className="w-24 shrink-0 text-ink-secondary-light dark:text-ink-secondary-dark">{s}</span>
                {prefs.length === 0 && <span className="text-ink-muted text-xs">no closed deals</span>}
                {prefs.map((c, i) => {
                  const rate = rateFor(tables, s, c);
                  const isMatch = c === matchedTo;
                  return (
                    <span
                      key={c}
                      className={
                        "text-xs px-xs rounded tabular " +
                        (isMatch
                          ? "bg-status-good/15 text-status-good font-semibold"
                          : "text-ink-muted")
                      }
                    >
                      {i + 1}. {c} ({money(rate)})
                    </span>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>

      <div>
        <h3 className="font-mono text-xs text-ink-muted mb-sm">
          carrier preferences (descending rate — highest first)
        </h3>
        <div className="space-y-xs">
          {carrierIds.map((c) => {
            const prefs = matching.carrier_preferences[c] ?? [];
            const matchedTo = matchedShipperOf[c];
            return (
              <div key={c} className="font-mono text-sm flex items-baseline gap-sm flex-wrap">
                <span className="w-28 shrink-0 text-ink-secondary-light dark:text-ink-secondary-dark">{c}</span>
                {prefs.length === 0 && <span className="text-ink-muted text-xs">no closed deals</span>}
                {prefs.map((s, i) => {
                  const rate = rateFor(tables, s, c);
                  const isMatch = s === matchedTo;
                  return (
                    <span
                      key={s}
                      className={
                        "text-xs px-xs rounded tabular " +
                        (isMatch
                          ? "bg-status-good/15 text-status-good font-semibold"
                          : "text-ink-muted")
                      }
                    >
                      {i + 1}. {s} ({money(rate)})
                    </span>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>

      <div className="md:col-span-2 border-t border-line-light dark:border-line-dark pt-md">
        <h3 className="font-mono text-xs text-ink-muted mb-sm">result</h3>
        <div className="flex flex-wrap gap-sm">
          {matching.matches.map((m) => (
            <span
              key={m.shipper_id}
              className="font-mono text-xs px-sm py-xs rounded bg-status-good/10 text-status-good border border-status-good/30"
            >
              {m.shipper_id} ↔ {m.carrier_id} · {money(m.rate)} · {m.turns_to_close} turns
            </span>
          ))}
          {matching.unmatched_shippers.map((s) => (
            <span
              key={s}
              className="font-mono text-xs px-sm py-xs rounded bg-status-critical/10 text-status-critical border border-status-critical/30"
            >
              {s} · no transport
            </span>
          ))}
          {unmatchedCarriers.map((c) => (
            <span
              key={c}
              className="font-mono text-xs px-sm py-xs rounded bg-status-warning/10 text-status-warning border border-status-warning/30"
            >
              {c} · no shipper
            </span>
          ))}
        </div>
        <p className="text-xs text-ink-muted mt-sm max-w-2xl">
          {diff > 0 &&
            `With ${shipperIds.length} shippers and ${carrierIds.length} carriers, at least ${diff} shipper${diff === 1 ? "" : "s"} always end${diff === 1 ? "s" : ""} up without transport. `}
          {diff < 0 &&
            `With ${shipperIds.length} shippers and ${carrierIds.length} carriers, at least ${-diff} carrier${-diff === 1 ? "" : "s"} always end${-diff === 1 ? "s" : ""} up without a shipper. `}
          {diff === 0 &&
            `With ${shipperIds.length} shippers and ${carrierIds.length} carriers, both sides could in theory be fully matched. `}
          A shipper can lose its slot even after closing the cheapest rate of the whole negotiation, if its
          carrier had a higher offer from another shipper (the matching paradox).
        </p>
      </div>
    </div>
  );
}
