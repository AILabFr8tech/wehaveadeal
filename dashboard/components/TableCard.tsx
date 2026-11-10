import type { TableResult } from "@/lib/types";
import { engineLabelForAgent } from "@/lib/types";
import { ArchBadge } from "@/components/ArchBadge";
import { money } from "@/lib/format";

export function TableCard({ table }: { table: TableResult }) {
  const shipperEngine = engineLabelForAgent(table.shipper_id);
  const carrierEngine = engineLabelForAgent(table.carrier_id);
  return (
    <div className="border border-line-light dark:border-line-dark rounded-lg p-md">
      <div className="flex items-center justify-between mb-sm">
        <span className="font-mono text-sm">
          {table.shipper_id} <ArchBadge className="ml-xs" archLabel={shipperEngine} />{" "}
          <span className="text-ink-muted">↔</span> {table.carrier_id}{" "}
          <ArchBadge className="ml-xs" archLabel={carrierEngine} />
        </span>
        {table.closed ? (
          <span className="font-mono text-xs px-sm py-xs rounded bg-status-good/10 text-status-good">
            closed · {money(table.rate)} · turn {table.turns_to_close}
          </span>
        ) : table.impasse ? (
          <span className="font-mono text-xs px-sm py-xs rounded bg-status-critical/10 text-status-critical">
            impasse
          </span>
        ) : (
          <span className="font-mono text-xs px-sm py-xs rounded border border-line-light dark:border-line-dark text-ink-muted">
            negotiating…
          </span>
        )}
      </div>
      <div className="space-y-[2px]">
        {table.history.map((rec, i) => {
          const isShipper = rec.actor === "shipper";
          return (
            <div key={i} className={"flex " + (isShipper ? "justify-start" : "justify-end")}>
              <div
                className={
                  "font-mono text-xs tabular px-sm py-[3px] rounded max-w-[75%] " +
                  (isShipper
                    ? "bg-series-3-light/10 dark:bg-series-3-dark/10 text-ink-primary-light dark:text-ink-primary-dark"
                    : "bg-series-4-light/10 dark:bg-series-4-dark/10 text-ink-primary-light dark:text-ink-primary-dark")
                }
              >
                <span className="text-ink-muted">t{rec.turn}</span>{" "}
                {rec.action === "offer" && money(rec.amount)}
                {rec.action === "accept" && <span className="text-status-good">accepts</span>}
                {rec.action === "reject" && <span className="text-status-critical">se retira</span>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
