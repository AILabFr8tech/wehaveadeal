import Link from "next/link";
import { listRoundFiles, loadRound } from "@/lib/data";
import { RoundPicker } from "@/components/RoundPicker";
import { PriceTrajectoryChart } from "@/components/PriceTrajectoryChart";
import { CarrierLeaderboard, ShipperLeaderboard } from "@/components/Leaderboard";
import { NegotiationList } from "@/components/NegotiationList";
import { engineLabelForAgent } from "@/lib/types";

const MODE_LABEL: Record<string, string> = { frozen: "frozen", learning: "learning" };

export default async function RoundPage({ params }: { params: Promise<{ file: string }> }) {
  const { file } = await params;
  const round = loadRound(file);
  const files = listRoundFiles();

  return (
    <div className="space-y-2xl">
      <div className="flex items-start justify-between flex-wrap gap-md">
        <div>
          <Link href="/" className="font-mono text-xs text-ink-muted hover:underline">
            ← History
          </Link>
          <h1 className="text-2xl font-semibold mt-xs">Round {round.round_index}</h1>
          <p className="font-mono text-xs text-ink-muted mt-xs">
            seed {round.seed} · mode {MODE_LABEL[round.mode] ?? round.mode} · {round.negotiations.length} negotiations ·{" "}
            {round.shipper_ids.length} shippers × {round.carrier_ids.length} carriers
          </p>
        </div>
        <RoundPicker files={files} current={file} />
      </div>

      <section>
        <h2 className="font-mono text-xs text-ink-muted uppercase tracking-wide mb-sm">
          Market and negotiation — price trajectory per negotiation
        </h2>
        <div className="border border-line-light dark:border-line-dark rounded-lg p-md">
          <PriceTrajectoryChart negotiations={round.negotiations} roundFile={file} carrierIds={round.carrier_ids} />
        </div>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-lg">
        <div className="border border-line-light dark:border-line-dark rounded-lg p-md">
          <h2 className="font-mono text-xs text-ink-muted uppercase tracking-wide mb-sm">
            Shipper standings — {Array.from(new Set(round.shipper_ids.map(engineLabelForAgent))).join(" vs ")}
          </h2>
          <ShipperLeaderboard standings={round.shipper_standings} winner={round.round_winner_shipper} />
        </div>
        <div className="border border-line-light dark:border-line-dark rounded-lg p-md">
          <h2 className="font-mono text-xs text-ink-muted uppercase tracking-wide mb-sm">
            Carrier standings — {Array.from(new Set(round.carrier_ids.map(engineLabelForAgent))).join(" vs ")}
          </h2>
          <CarrierLeaderboard standings={round.carrier_standings} winner={round.round_winner_carrier} />
        </div>
      </section>

      <section>
        <h2 className="font-mono text-xs text-ink-muted uppercase tracking-wide mb-sm">
          Negotiations — drill into any of the {round.negotiations.length}
        </h2>
        <NegotiationList negotiations={round.negotiations} roundFile={file} />
      </section>
    </div>
  );
}
