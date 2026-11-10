import fs from "fs";
import path from "path";
import type { RoundResult } from "./types";

// Data lives in /data at the repo root (sibling of /dashboard and /engine),
// written by `python -m wehaveadeal.simulate --out data` or `wehaveadeal benchmark <agent> --save-rounds data`. No database, no network call --
// this is a local file read, same spirit as the rest of Phase 1.
const DATA_DIR = path.join(process.cwd(), "..", "data");

export function listRoundFiles(): string[] {
  if (!fs.existsSync(DATA_DIR)) return [];
  return fs
    .readdirSync(DATA_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort()
    .reverse(); // newest-looking filename first (round_XXX_seed_YYY.json)
}

const cache = new Map<string, RoundResult>();

export function loadRound(file: string): RoundResult {
  if (cache.has(file)) return cache.get(file)!;
  const full = path.join(DATA_DIR, file);
  if (!full.startsWith(DATA_DIR)) throw new Error("invalid round file path");
  const raw = fs.readFileSync(full, "utf-8");
  const parsed = JSON.parse(raw) as RoundResult;
  cache.set(file, parsed);
  return parsed;
}

export interface RoundSummary {
  file: string;
  round_index: number;
  seed: number;
  n_negotiations: number;
  shipper_ids: string[];
  carrier_ids: string[];
  round_winner_shipper: string | null;
  round_winner_carrier: string | null;
}

/** Lightweight per-round summary for the history ("History") listing. */
export function listRoundSummaries(): RoundSummary[] {
  return listRoundFiles().map((file) => {
    const r = loadRound(file);
    return {
      file,
      round_index: r.round_index,
      seed: r.seed,
      n_negotiations: r.negotiations.length,
      shipper_ids: r.shipper_ids,
      carrier_ids: r.carrier_ids,
      round_winner_shipper: r.round_winner_shipper,
      round_winner_carrier: r.round_winner_carrier,
    };
  });
}
