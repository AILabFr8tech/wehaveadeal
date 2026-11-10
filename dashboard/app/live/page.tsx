"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { TableCard } from "@/components/TableCard";
import { MatchingPanel } from "@/components/MatchingPanel";
import { NegotiationGraph } from "@/components/NegotiationGraph";
import { API_BASE } from "@/lib/config";
import { archColor, isDashedArch } from "@/lib/colors";
import { num } from "@/lib/format";
import { usePrefersDark } from "@/lib/usePrefersDark";
import type { LiveEvent } from "@/lib/live";
import type { NegotiationResult, TableResult, TurnRecord } from "@/lib/types";

type Status = "idle" | "connecting" | "live" | "finished" | "error";
type View = "cards" | "graph";

const ARCH_OPTIONS: { kind: string; label: string }[] = [
  { kind: "r", label: "R" },
  { kind: "rv", label: "RV" },
  { kind: "is", label: "IS" },
  { kind: "opt", label: "OPT" },
  { kind: "opt2", label: "OPT2" },
  { kind: "ml", label: "ML" },
  { kind: "ml2", label: "ML2" },
  { kind: "rl", label: "RL" },
  { kind: "rl2", label: "RL2" },
];
const DEFAULT_SHIPPERS = 4;
const DEFAULT_CARRIERS = 3;
const MIN_SHIPPERS = 1;
const MAX_SHIPPERS = 8;
const MIN_CARRIERS = 1;
const MAX_CARRIERS = 8;

// Repeats `kinds` round-robin until it has exactly `n` entries, so a
// handful of selected architectures can fill any carrier-count slider
// value, e.g. [is, opt] at n=5 -> [is, opt, is, opt, is].
function cycleToLength(kinds: string[], n: number): string[] {
  if (kinds.length === 0) return [];
  return Array.from({ length: n }, (_, i) => kinds[i % kinds.length]);
}

function emptyTable(shipperId: string, carrierId: string): TableResult {
  return {
    shipper_id: shipperId,
    carrier_id: carrierId,
    closed: false,
    impasse: false,
    rate: null,
    turns_to_close: null,
    history: [],
  };
}

const STATUS_LABEL: Record<Status, string> = {
  idle: "idle",
  connecting: "connecting…",
  live: "● live",
  finished: "finished",
  error: "error",
};

export default function EnVivoPage() {
  const dark = usePrefersDark();
  const mode = dark ? "dark" : "light";
  const [selectedShipperKinds, setSelectedShipperKinds] = useState<string[]>([]);
  const [selectedKinds, setSelectedKinds] = useState<string[]>([]);
  const [nShippers, setNShippers] = useState(DEFAULT_SHIPPERS);
  const [nCarriers, setNCarriers] = useState(DEFAULT_CARRIERS);
  const [status, setStatus] = useState<Status>("idle");
  const [seed, setSeed] = useState<number | null>(null);
  const [salePrice, setSalePrice] = useState<number | null>(null);
  const [goodsCost, setGoodsCost] = useState<number | null>(null);
  const [shipperIds, setShipperIds] = useState<string[]>([]);
  const [carrierIds, setCarrierIds] = useState<string[]>([]);
  const [currentTurn, setCurrentTurn] = useState(0);
  const [tables, setTables] = useState<Record<string, TableResult>>({});
  const [result, setResult] = useState<NegotiationResult | null>(null);
  const [view, setView] = useState<View>("cards");
  // Whether a trained model exists for the "ml"/"ml2"/"rl"/"rl2" kinds on
  // each role -- fetched once so those buttons can be disabled with an
  // explanation instead of letting someone pick one and find out
  // mid-negotiation (the API itself also rejects it up front with a 400,
  // but EventSource can't surface that response body -- see wehaveadeal/api.py's
  // _check_trained_kinds_available).
  type TrainedAvailability = { ml: boolean; ml2: boolean; rl: boolean; rl2: boolean };
  const [engineAvailable, setEngineAvailable] = useState<{ carrier: TrainedAvailability; shipper: TrainedAvailability } | null>(
    null
  );

  // Community agents discovered by the backend in its agents/ folder
  // (wehaveadeal/api.py::engine_status -> "community"), offered next to the baselines.
  type CommunityOption = { kind: string; label: string };
  const [community, setCommunity] = useState<{ carrier: CommunityOption[]; shipper: CommunityOption[] }>({
    carrier: [],
    shipper: [],
  });

  const esRef = useRef<EventSource | null>(null);
  // Per-table last-seen offers, kept only to reconstruct the accepted rate
  // when an "accept" event arrives (the accept event itself carries no
  // amount, since Move.amount is only set for OFFER -- see wehaveadeal/models.py).
  const lastOffers = useRef<Record<string, { shipper: number | null; carrier: number | null }>>({});

  function toggleShipperKind(kind: string) {
    if (status === "connecting" || status === "live") return;
    setSelectedShipperKinds((prev) => {
      if (prev.includes(kind)) return prev.filter((k) => k !== kind);
      if (prev.length >= nShippers) return prev;
      return [...prev, kind];
    });
  }

  function toggleKind(kind: string) {
    if (status === "connecting" || status === "live") return;
    setSelectedKinds((prev) => {
      if (prev.includes(kind)) return prev.filter((k) => k !== kind);
      if (prev.length >= nCarriers) return prev;
      return [...prev, kind];
    });
  }

  // If a count slider drops below its side's current selection, trim it --
  // selecting more distinct architectures than there are slots to fill
  // would just mean some choices get silently dropped by cycleToLength.
  useEffect(() => {
    setSelectedShipperKinds((prev) => (prev.length > nShippers ? prev.slice(0, nShippers) : prev));
  }, [nShippers]);
  useEffect(() => {
    setSelectedKinds((prev) => (prev.length > nCarriers ? prev.slice(0, nCarriers) : prev));
  }, [nCarriers]);

  function start(
    kindsOverride?: string[],
    nShippersOverride?: number,
    nCarriersOverride?: number,
    shipperKindsOverride?: string[]
  ) {
    const picked = kindsOverride ?? selectedKinds;
    const pickedShippers = shipperKindsOverride ?? selectedShipperKinds;
    const shippersCount = nShippersOverride ?? nShippers;
    const carriersCount = nCarriersOverride ?? nCarriers;
    if (picked.length === 0 || pickedShippers.length === 0) return;
    const kinds = cycleToLength(picked, carriersCount);
    const shipperKinds = cycleToLength(pickedShippers, shippersCount);
    esRef.current?.close();
    setStatus("connecting");
    setResult(null);
    setTables({});
    setCurrentTurn(0);
    lastOffers.current = {};

    const carriersParam = kinds.join(",");
    const shippersParam = shipperKinds.join(",");
    const es = new EventSource(
      `${API_BASE}/api/live/negotiation?delay_ms=350&carriers=${carriersParam}&shippers=${shippersParam}`
    );
    esRef.current = es;

    es.onmessage = (ev) => {
      const data = JSON.parse(ev.data) as LiveEvent;

      if (data.kind === "setup") {
        setSeed(data.seed);
        setSalePrice(data.sale_price);
        setGoodsCost(data.goods_cost);
        setShipperIds(data.shipper_ids);
        setCarrierIds(data.carrier_ids);

        const initial: Record<string, TableResult> = {};
        const offers: Record<string, { shipper: number | null; carrier: number | null }> = {};
        for (const s of data.shipper_ids) {
          for (const c of data.carrier_ids) {
            const id = `${s}::${c}`;
            initial[id] = emptyTable(s, c);
            offers[id] = { shipper: null, carrier: null };
          }
        }
        setTables(initial);
        lastOffers.current = offers;
        setStatus("live");
        return;
      }

      if (data.kind === "turn") {
        setCurrentTurn(data.turn);
        setTables((prev) => {
          const t = prev[data.table_id];
          if (!t) return prev;
          const rec: TurnRecord = {
            turn: data.turn,
            actor: data.actor,
            action: data.action,
            amount: data.amount,
          };
          const history = [...t.history, rec];
          let { closed, impasse, rate, turns_to_close } = t;
          const offers = lastOffers.current[data.table_id];

          if (data.action === "offer") {
            if (data.actor === "shipper") offers.shipper = data.amount;
            else offers.carrier = data.amount;
          } else if (data.action === "accept") {
            closed = true;
            turns_to_close = data.turn;
            rate = data.actor === "shipper" ? offers.carrier : offers.shipper;
          } else if (data.action === "reject") {
            impasse = true;
          }

          return { ...prev, [data.table_id]: { ...t, history, closed, impasse, rate, turns_to_close } };
        });
        return;
      }

      if (data.kind === "result") {
        // The final result is authoritative: it also marks tables that ran out of
        // turns without a deal as impasses (the turn events only flag explicit rejects).
        const finalTables: TableResult[] = data.negotiation.tables ?? [];
        setTables((prev) => {
          const next = { ...prev };
          for (const ft of finalTables) {
            const id = `${ft.shipper_id}::${ft.carrier_id}`;
            if (next[id]) {
              next[id] = { ...next[id], closed: ft.closed, impasse: ft.impasse, rate: ft.rate, turns_to_close: ft.turns_to_close };
            }
          }
          return next;
        });
        setResult(data.negotiation);
        setStatus("finished");
        es.close();
      }
    };

    es.onerror = () => {
      setStatus("error");
      es.close();
    };
  }

  useEffect(() => {
    fetch(`${API_BASE}/api/engine-status`)
      .then((r) => r.json())
      .then((data) => {
        setEngineAvailable({ carrier: data.carrier, shipper: data.shipper });
        if (data.community) {
          const toOptions = (xs: { kind: string; label: string }[]) =>
            xs.map(({ kind, label }) => ({ kind, label: label.toUpperCase() }));
          setCommunity({ carrier: toOptions(data.community.carrier ?? []), shipper: toOptions(data.community.shipper ?? []) });
        }
      })
      .catch(() =>
        setEngineAvailable({
          carrier: { ml: false, ml2: false, rl: false, rl2: false },
          shipper: { ml: false, ml2: false, rl: false, rl2: false },
        })
      );
  }, []);

  useEffect(() => {
    // ?auto=1 auto-starts on load with a default 3-way mix -- handy for
    // bookmarking/testing, not required for normal use (picking 3 engines
    // and clicking the button does the same thing).
    if (new URLSearchParams(window.location.search).get("auto") === "1") {
      const defaults = ["is", "opt", "opt2"];
      const shipperDefaults = ["r"];
      const autoShippers = DEFAULT_SHIPPERS;
      const autoCarriers = DEFAULT_CARRIERS;
      setNShippers(autoShippers);
      setNCarriers(autoCarriers);
      setSelectedKinds(defaults);
      setSelectedShipperKinds(shipperDefaults);
      start(defaults, autoShippers, autoCarriers, shipperDefaults);
    }
    return () => esRef.current?.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const orderedTables = shipperIds.flatMap((s) => carrierIds.map((c) => tables[`${s}::${c}`])).filter(Boolean);
  const isRunning = status === "connecting" || status === "live";

  return (
    <div className="space-y-2xl">
      <div>
        <Link href="/" className="font-mono text-xs text-ink-muted hover:underline">
          ← History
        </Link>
        <h1 className="text-2xl font-semibold mt-xs">Live negotiation</h1>
        <p className="font-mono text-xs text-ink-muted mt-xs max-w-2xl">
          Each new negotiation is generated and streamed turn by turn from the engine's live API
          (FastAPI, <code>wehaveadeal/api.py</code>), not from files saved in <code>/data</code>.
        </p>
      </div>

      <div className="border border-line-light dark:border-line-dark rounded-lg p-md space-y-md">
        <div>
          <h3 className="font-mono text-xs text-ink-muted uppercase tracking-wide mb-sm">market scenario</h3>
          <div className="flex items-center gap-xl flex-wrap">
            <label className="font-mono text-xs text-ink-muted flex items-center gap-sm">
              shippers
              <input
                type="range"
                min={MIN_SHIPPERS}
                max={MAX_SHIPPERS}
                value={nShippers}
                disabled={isRunning}
                onChange={(e) => setNShippers(Number(e.target.value))}
                className="range-input disabled:opacity-40"
              />
              <span className="tabular text-ink-primary-light dark:text-ink-primary-dark w-[1.5em] text-right">
                {nShippers}
              </span>
            </label>
            <label className="font-mono text-xs text-ink-muted flex items-center gap-sm">
              carriers
              <input
                type="range"
                min={MIN_CARRIERS}
                max={MAX_CARRIERS}
                value={nCarriers}
                disabled={isRunning}
                onChange={(e) => setNCarriers(Number(e.target.value))}
                className="range-input disabled:opacity-40"
              />
              <span className="tabular text-ink-primary-light dark:text-ink-primary-dark w-[1.5em] text-right">
                {nCarriers}
              </span>
            </label>
            <span className="font-mono text-xs text-ink-muted">
              ({nShippers * nCarriers} bilateral tables)
            </span>
          </div>
        </div>

        <div className="border-t border-line-light dark:border-line-dark pt-md">
          <h3 className="font-mono text-xs text-ink-muted uppercase tracking-wide mb-sm">
            shipper engines — pick 1 to {nShippers} ({selectedShipperKinds.length} picked
            {selectedShipperKinds.length > 0 && selectedShipperKinds.length < nShippers ? " · repeated in rotation" : ""}
            )
          </h3>
          <div className="flex items-center gap-sm flex-wrap">
            {[...ARCH_OPTIONS, ...community.shipper].map(({ kind, label }) => {
              const isSelected = selectedShipperKinds.includes(kind);
              const color = archColor(label, mode);
              const dashed = isDashedArch(label);
              const isUnavailable =
                (kind === "ml" || kind === "ml2" || kind === "rl" || kind === "rl2") &&
                engineAvailable !== null &&
                !engineAvailable.shipper[kind as "ml" | "ml2" | "rl" | "rl2"];
              const isLocked = !isSelected && selectedShipperKinds.length >= nShippers;
              return (
                <button
                  key={kind}
                  type="button"
                  onClick={() => toggleShipperKind(kind)}
                  disabled={isLocked || isRunning || isUnavailable}
                  title={isUnavailable ? "No trained model yet -- see wehaveadeal/ml/ or wehaveadeal/rl/" : undefined}
                  className={
                    "font-mono text-xs px-md py-sm rounded border disabled:cursor-not-allowed " +
                    (isSelected
                      ? "font-semibold"
                      : "border-line-light dark:border-line-dark text-ink-muted " +
                        (isLocked || isRunning || isUnavailable ? "opacity-30" : "opacity-70 hover:opacity-100"))
                  }
                  style={
                    isSelected
                      ? {
                          borderColor: color,
                          borderStyle: dashed ? "dashed" : undefined,
                          background: `${color}26`,
                          color,
                        }
                      : undefined
                  }
                >
                  {label}
                </button>
              );
            })}
          </div>
        </div>

        <div className="border-t border-line-light dark:border-line-dark pt-md">
          <h3 className="font-mono text-xs text-ink-muted uppercase tracking-wide mb-sm">
            carrier engines — pick 1 to {nCarriers} ({selectedKinds.length} picked
            {selectedKinds.length > 0 && selectedKinds.length < nCarriers ? " · repeated in rotation" : ""})
          </h3>
          <div className="flex items-center gap-sm flex-wrap">
            {[...ARCH_OPTIONS, ...community.carrier].map(({ kind, label }) => {
              const isSelected = selectedKinds.includes(kind);
              const color = archColor(label, mode);
              const dashed = isDashedArch(label);
              const isUnavailable =
                (kind === "ml" || kind === "ml2" || kind === "rl" || kind === "rl2") &&
                engineAvailable !== null &&
                !engineAvailable.carrier[kind as "ml" | "ml2" | "rl" | "rl2"];
              const isLocked = !isSelected && selectedKinds.length >= nCarriers;
              return (
                <button
                  key={kind}
                  type="button"
                  onClick={() => toggleKind(kind)}
                  disabled={isLocked || isRunning || isUnavailable}
                  title={isUnavailable ? "No trained model yet -- see wehaveadeal/ml/ or wehaveadeal/rl/" : undefined}
                  className={
                    "font-mono text-xs px-md py-sm rounded border disabled:cursor-not-allowed " +
                    (isSelected
                      ? "font-semibold"
                      : "border-line-light dark:border-line-dark text-ink-muted " +
                        (isLocked || isRunning || isUnavailable ? "opacity-30" : "opacity-70 hover:opacity-100"))
                  }
                  style={
                    isSelected
                      ? {
                          borderColor: color,
                          borderStyle: dashed ? "dashed" : undefined,
                          background: `${color}26`,
                          color,
                        }
                      : undefined
                  }
                >
                  {label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-md flex-wrap">
        <button
          onClick={() => start()}
          disabled={isRunning || selectedKinds.length === 0 || selectedShipperKinds.length === 0}
          className="font-mono text-xs px-md py-sm rounded bg-status-critical/10 text-status-critical border border-status-critical/30 hover:bg-status-critical/20 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {status === "finished" || status === "error" ? "↻ run a new negotiation" : "▶ start live negotiation"}
        </button>
        <span className="font-mono text-xs text-ink-muted">status: {STATUS_LABEL[status]}</span>
        {status === "live" && <span className="font-mono text-xs text-ink-muted">turn {currentTurn}/12</span>}
        {seed !== null && (
          <span className="font-mono text-xs text-ink-muted">
            seed {seed}
            {salePrice !== null && <> · sale price ${num(Math.round(salePrice))}</>}
            {goodsCost !== null && <> · goods cost ${num(Math.round(goodsCost))}</>}
          </span>
        )}
      </div>

      {status === "error" && (
        <div className="border border-status-critical/30 bg-status-critical/5 rounded-lg p-md text-sm">
          Could not connect to the live API at <code>{API_BASE}</code>. Start it with:
          <pre className="font-mono text-xs mt-sm bg-surface-light dark:bg-surface-dark p-sm rounded overflow-x-auto">
            wehaveadeal serve
          </pre>
        </div>
      )}

      {orderedTables.length > 0 && (
        <section>
          <div className="flex items-center justify-between flex-wrap gap-md mb-sm">
            <h2 className="font-mono text-xs text-ink-muted uppercase tracking-wide">
              {orderedTables.length} bilateral tables — building turn by turn
            </h2>
            <div className="inline-flex rounded border border-line-light dark:border-line-dark overflow-hidden font-mono text-xs">
              {(
                [
                  ["cards", "cards"],
                  ["graph", "graph"],
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
          </div>

          {view === "cards" ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-md">
              {orderedTables.map((t) => (
                <TableCard key={`${t.shipper_id}-${t.carrier_id}`} table={t} />
              ))}
            </div>
          ) : (
            <NegotiationGraph
              shipperIds={shipperIds}
              carrierIds={carrierIds}
              tables={tables}
              matching={result?.matching ?? null}
              mode={mode}
            />
          )}
        </section>
      )}

      {result && (
        <section>
          <h2 className="font-mono text-xs text-ink-muted uppercase tracking-wide mb-sm">
            Gale–Shapley matching — final result
          </h2>
          <div className="border border-line-light dark:border-line-dark rounded-lg p-md">
            <MatchingPanel
              matching={result.matching}
              tables={result.tables}
              shipperIds={shipperIds}
              carrierIds={carrierIds}
            />
          </div>
        </section>
      )}
    </div>
  );
}
