"""RL2 vs RL2Fixed ablation: re-runs the exact same 3-condition tournament
protocol as notebooks/validation_exhaustive.ipynb (same base seeds, same N,
same scarcity-pair rotation), with CarrierRL2/ShipperRL2 replaced by the
retrained CarrierRL2Fixed/ShipperRL2Fixed (wehaveadeal/rl/league_env.py's
portfolio-visibility fix, see that module's docstring). Only one architecture
is swapped; all 8 other architectures are unchanged, so any difference
between this run's "rl2" numbers and the original study's is attributable to
the fix, not to a different market draw -- reusing the original's exact base
seeds makes the two runs' market conditions bit-for-bit identical.

Writes its own output files (data/validation/ablation_*.csv) rather than
overwriting the main study's raw_results.csv/summary_stats.csv/pairwise_tests.csv.

Usage:
    python scripts/run_rl2fixed_ablation.py --role carrier
    python scripts/run_rl2fixed_ablation.py --role shipper
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))  # repo root: the wehaveadeal package

import pandas as pd

from wehaveadeal.agents import (
    CarrierIS, CarrierML, CarrierML2, CarrierOPT, CarrierOPT2, CarrierR, CarrierRL, CarrierRL2, CarrierRL2Fixed, CarrierRV,
    ShipperIS, ShipperML, ShipperML2, ShipperOPT, ShipperOPT2, ShipperR, ShipperRL, ShipperRL2, ShipperRL2Fixed, ShipperRV,
)
from wehaveadeal.simulate import CARRIER_KINDS, SHIPPER_KINDS, run_round

assert CARRIER_KINDS == SHIPPER_KINDS
ALL_KINDS = list(CARRIER_KINDS)  # the original 9: rl2's slot is swapped for rl2fixed at build time only
ARCH_LABELS = {"r": "R", "rv": "RV", "is": "IS", "opt": "OPT", "opt2": "OPT2",
               "ml": "ML", "ml2": "ML2", "rl": "RL", "rl2": "RL2Fixed"}

CARRIER_BUILDERS = {
    "r": CarrierR, "rv": CarrierRV, "is": CarrierIS, "opt": CarrierOPT, "opt2": CarrierOPT2,
    "ml": CarrierML, "ml2": CarrierML2, "rl": CarrierRL, "rl2": CarrierRL2Fixed,
}
SHIPPER_BUILDERS = {
    "r": ShipperR, "rv": ShipperRV, "is": ShipperIS, "opt": ShipperOPT, "opt2": ShipperOPT2,
    "ml": ShipperML, "ml2": ShipperML2, "rl": ShipperRL, "rl2": ShipperRL2Fixed,
}

DATA_DIR = PROJECT_ROOT / "data" / "validation"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def build_roster(kinds, role: str):
    builders = CARRIER_BUILDERS if role == "carrier" else SHIPPER_BUILDERS
    engines, id_to_kind = {}, {}
    for kind in kinds:
        agent_id = f"{role}-{kind}-1"
        engines[agent_id] = builders[kind](agent_id=agent_id)
        id_to_kind[agent_id] = kind
    return engines, id_to_kind


def extract_rows(round_result, config: str, excluded_label: str, carrier_id_to_kind, shipper_id_to_kind):
    rows = []
    for neg in round_result.negotiations:
        day_id = f"{config}|{excluded_label}|{round_result.seed}|{neg.negotiation_index}"
        for co in neg.carrier_outcomes:
            rows.append({
                "config": config, "excluded": excluded_label, "day_id": day_id,
                "role": "carrier", "kind": carrier_id_to_kind[co.carrier_id],
                "matched": co.matched, "outcome": co.margin,
            })
        for so in neg.shipper_outcomes:
            rows.append({
                "config": config, "excluded": excluded_label, "day_id": day_id,
                "role": "shipper", "kind": shipper_id_to_kind[so.shipper_id],
                "matched": so.matched, "outcome": so.profit,
            })
    return rows


def run_balanced(seed_base: int, n_negotiations: int):
    carriers, c_map = build_roster(ALL_KINDS, "carrier")
    shippers, s_map = build_roster(ALL_KINDS, "shipper")
    result = run_round(round_index=0, seed=seed_base, shipper_engines=shippers,
                        carrier_engines=carriers, n_negotiations=n_negotiations)
    return extract_rows(result, "balanced", "none", c_map, s_map)


def run_scarce(seed_base: int, scarce_role: str, n_per_pair: int):
    rows = []
    pairs = list(itertools.combinations(ALL_KINDS, 2))
    for i, pair in enumerate(pairs):
        kept_scarce = [k for k in ALL_KINDS if k not in pair]
        if scarce_role == "carrier":
            carriers, c_map = build_roster(kept_scarce, "carrier")
            shippers, s_map = build_roster(ALL_KINDS, "shipper")
        else:
            shippers, s_map = build_roster(kept_scarce, "shipper")
            carriers, c_map = build_roster(ALL_KINDS, "carrier")
        result = run_round(round_index=i, seed=seed_base + i, shipper_engines=shippers,
                            carrier_engines=carriers, n_negotiations=n_per_pair)
        config_name = f"{scarce_role}_scarce"
        rows += extract_rows(result, config_name, "+".join(pair), c_map, s_map)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-balanced", type=int, default=400)
    parser.add_argument("--n-per-pair-scarce", type=int, default=12)
    args = parser.parse_args()

    raw_path = DATA_DIR / "ablation_rl2fixed_raw_results.csv"
    print("Running RL2Fixed ablation tournament (same protocol/seeds as validation_exhaustive.ipynb)...")
    all_rows = []
    all_rows += run_balanced(seed_base=10_000_000, n_negotiations=args.n_balanced)
    print(f"  balanced done: {len(all_rows)} rows so far")
    all_rows += run_scarce(seed_base=20_000_000, scarce_role="carrier", n_per_pair=args.n_per_pair_scarce)
    print(f"  carrier_scarce done: {len(all_rows)} rows so far")
    all_rows += run_scarce(seed_base=30_000_000, scarce_role="shipper", n_per_pair=args.n_per_pair_scarce)
    print(f"  shipper_scarce done: {len(all_rows)} rows so far")

    raw = pd.DataFrame(all_rows)
    raw["arch"] = raw["kind"].map(ARCH_LABELS)
    raw.to_csv(raw_path, index=False)
    print(f"Wrote {raw_path} ({len(raw)} rows)")


if __name__ == "__main__":
    main()
