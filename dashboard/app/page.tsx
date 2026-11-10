import Link from "next/link";
import { listRoundSummaries } from "@/lib/data";
import { engineLabelForAgent } from "@/lib/types";
import { ArchBadge } from "@/components/ArchBadge";

// Always re-read /data at request time -- otherwise a production build
// would bake in whatever rounds existed at build time and never show new
// ones generated afterwards by `python -m wehaveadeal.simulate --out data` or `wehaveadeal benchmark <agent> --save-rounds data`.
export const dynamic = "force-dynamic";

export default function HistoryPage() {
  const rounds = listRoundSummaries();

  if (rounds.length === 0) {
    return (
      <div className="font-mono text-sm text-ink-secondary-light dark:text-ink-secondary-dark">
        No saved rounds in <code>/data</code>. Run{" "}
        <code className="text-series-1-light dark:text-series-1-dark">
          python -m wehaveadeal.simulate --out data --seed 42 --negotiations 100
        </code>{" "}
        from the repository root first.
      </div>
    );
  }

  return (
    <div className="space-y-2xl">
      <div>
        <h1 className="text-2xl font-semibold">Round history</h1>
        <p className="font-mono text-xs text-ink-muted mt-xs">
          {rounds.length} rounds · the mix of shipper and carrier engines can differ between rounds
          ("engines" column)
        </p>
      </div>

      <div className="border border-line-light dark:border-line-dark rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs font-mono text-ink-muted border-b border-line-light dark:border-line-dark bg-surface-light dark:bg-surface-dark">
              <th className="py-sm px-md font-normal">round</th>
              <th className="py-sm px-md font-normal">seed</th>
              <th className="py-sm px-md font-normal">winner — shipper</th>
              <th className="py-sm px-md font-normal">winner — carrier</th>
              <th className="py-sm px-md font-normal">engines</th>
              <th className="py-sm px-md font-normal text-right">negotiations</th>
              <th className="py-sm px-md font-normal"></th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {rounds.map((r) => (
              <tr
                key={r.file}
                className="border-b border-line-light dark:border-line-dark last:border-0 hover:bg-line-light/30 dark:hover:bg-line-dark/30"
              >
                <td className="py-sm px-md">
                  <Link href={`/round/${r.file}`} className="text-series-1-light dark:text-series-1-dark hover:underline font-semibold">
                    Round {r.round_index}
                  </Link>
                </td>
                <td className="py-sm px-md tabular text-ink-muted">{r.seed}</td>
                <td className="py-sm px-md">
                  <span className="text-status-good">★</span> {r.round_winner_shipper}
                </td>
                <td className="py-sm px-md">
                  <span className="text-status-good">★</span> {r.round_winner_carrier}{" "}
                  {r.round_winner_carrier && (
                    <ArchBadge className="ml-xs" archLabel={engineLabelForAgent(r.round_winner_carrier)} />
                  )}
                </td>
                <td className="py-sm px-md">
                  <div className="flex gap-xs flex-wrap items-center">
                    {Array.from(new Set(r.shipper_ids.map(engineLabelForAgent))).map((arch) => (
                      <ArchBadge key={`s-${arch}`} archLabel={arch} />
                    ))}
                    <span className="text-ink-muted text-xs">×</span>
                    {Array.from(new Set(r.carrier_ids.map(engineLabelForAgent))).map((arch) => (
                      <ArchBadge key={`c-${arch}`} archLabel={arch} />
                    ))}
                  </div>
                </td>
                <td className="py-sm px-md text-right tabular">{r.n_negotiations}</td>
                <td className="py-sm px-md text-right">
                  <Link href={`/round/${r.file}`} className="text-xs text-ink-muted hover:underline">
                    details →
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="border border-status-critical/30 bg-status-critical/5 rounded-lg p-md">
        <h2 className="font-mono text-xs text-status-critical uppercase tracking-wide mb-xs">
          ● Live negotiation
        </h2>
        <p className="text-sm text-ink-secondary-light dark:text-ink-secondary-dark max-w-2xl">
          Besides the history, you can watch a new negotiation unfold turn by turn in real time,
          served by the engine's live API (FastAPI).
        </p>
        <Link
          href="/live"
          className="inline-block mt-sm font-mono text-xs px-md py-sm rounded bg-status-critical/10 text-status-critical border border-status-critical/30 hover:bg-status-critical/20"
        >
          ● start a live negotiation →
        </Link>
      </div>
    </div>
  );
}
