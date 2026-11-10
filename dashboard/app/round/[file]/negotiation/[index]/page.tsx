import Link from "next/link";
import { loadRound } from "@/lib/data";
import { TableCard } from "@/components/TableCard";
import { MatchingPanel } from "@/components/MatchingPanel";
import { TurnConvergenceChart } from "@/components/TurnConvergenceChart";
import { num } from "@/lib/format";

export default async function NegotiationPage({
  params,
}: {
  params: Promise<{ file: string; index: string }>;
}) {
  const { file, index } = await params;
  const round = loadRound(file);
  const idx = parseInt(index, 10);
  const neg = round.negotiations.find((n) => n.negotiation_index === idx);

  if (!neg) {
    return <div className="font-mono text-sm">Negotiation {index} not found in {file}.</div>;
  }

  const prev = idx > 0 ? idx - 1 : null;
  const next = idx < round.negotiations.length - 1 ? idx + 1 : null;

  return (
    <div className="space-y-2xl">
      <div className="flex items-start justify-between flex-wrap gap-md">
        <div>
          <Link href={`/round/${file}`} className="font-mono text-xs text-ink-muted hover:underline">
            ← Round {round.round_index}
          </Link>
          <h1 className="text-2xl font-semibold mt-xs">Negotiation {idx}</h1>
          <p className="font-mono text-xs text-ink-muted mt-xs">
            sale price ${num(Math.round(neg.sale_price))} · goods cost $
            {num(Math.round(neg.goods_cost))} · seed {neg.seed}
          </p>
        </div>
        <div className="font-mono text-xs flex gap-md">
          {prev !== null && (
            <Link className="text-series-1-light dark:text-series-1-dark hover:underline" href={`/round/${file}/negotiation/${prev}`}>
              ← negotiation {prev}
            </Link>
          )}
          {next !== null && (
            <Link className="text-series-1-light dark:text-series-1-dark hover:underline" href={`/round/${file}/negotiation/${next}`}>
              negotiation {next} →
            </Link>
          )}
        </div>
      </div>

      <section>
        <h2 className="font-mono text-xs text-ink-muted uppercase tracking-wide mb-sm">
          Gale–Shapley matching
        </h2>
        <div className="border border-line-light dark:border-line-dark rounded-lg p-md">
          <MatchingPanel
            matching={neg.matching}
            tables={neg.tables}
            shipperIds={round.shipper_ids}
            carrierIds={round.carrier_ids}
          />
        </div>
      </section>

      <section>
        <h2 className="font-mono text-xs text-ink-muted uppercase tracking-wide mb-sm">
          12 bilateral tables — full turn-by-turn history
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-md">
          {neg.tables.map((t) => (
            <TableCard key={`${t.shipper_id}-${t.carrier_id}`} table={t} />
          ))}
        </div>

        <div className="mt-lg border border-line-light dark:border-line-dark rounded-lg p-md">
          <h3 className="font-mono text-xs text-ink-muted uppercase tracking-wide mb-sm">
            Turn-by-turn convergence (mean over the 12 tables)
          </h3>
          <TurnConvergenceChart tables={neg.tables} />
        </div>
      </section>
    </div>
  );
}
