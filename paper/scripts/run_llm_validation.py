"""Runs the LLM-vs-engines validation study: for each (model, role)
combination, embeds one LLM-backed agent (CarrierLLM/ShipperLLM) in a
small, real competitive market -- alongside a few hand-designed/learned
rivals on its own side -- against a fixed representative roster on the
other side, for `--days` independent negotiations.

Built to run UNATTENDED on a remote server for hours (see
paper/README.md): every negotiation is its own `run_round`
call, and results are appended to a JSONL file and flushed to disk
IMMEDIATELY after each one -- if the process dies partway (Ollama hangs,
the machine reboots, you Ctrl+C it), re-running the exact same command
picks up where it left off instead of starting over or double-counting.
A single failed negotiation (a model that never produces a parseable
response, a network blip) is logged and skipped rather than aborting the
whole run.

Output, per (model, role):
  data/llm_validation/<model>_<role>_outcomes.jsonl  -- one row per agent
      per negotiation (LLM AND its rivals, so the rivals' results in this
      same small market are directly comparable, not thrown away)
  data/llm_validation/<model>_<role>_calls.jsonl      -- one row per model
      call (prompt-response latency, parse errors, raw text) -- the
      structured-output-compliance and latency numbers the paper wants

Usage:
    python scripts/run_llm_validation.py --days 40
    python scripts/run_llm_validation.py --models llama3.1:8b --roles carrier --days 5   # quick pilot
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import zlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))  # repo root: the wehaveadeal package

from wehaveadeal.agents import CarrierLLM, ShipperLLM
from wehaveadeal.llm.roster import DEFAULT_MODELS, family_of, tier_of
from wehaveadeal.simulate import build_carrier, build_shipper, run_round

RIVAL_KINDS = ["r", "opt2", "ml2"]  # same-side rivals -- spans weak baseline / best hand-designed / best learned
COUNTERPART_KINDS = ["r", "rv", "opt2", "is"]  # fixed opposite-side roster, hand-designed only (no latency, no LLM)

OUT_DIR = PROJECT_ROOT / "data" / "llm_validation"


def sanitize(name: str) -> str:
    return name.replace(":", "-").replace("/", "-")


def build_llm_engine(role: str, agent_id: str, model: str, host: str, timeout_s: float, max_retries: int, think, log_sink: list):
    cls = CarrierLLM if role == "carrier" else ShipperLLM
    return cls(agent_id=agent_id, model=model, host=host, timeout_s=timeout_s, max_retries=max_retries, think=think, log_sink=log_sink)


def build_roster(role: str, model: str, host: str, timeout_s: float, max_retries: int, think, log_sink: list):
    """Returns (own_side_engines, other_side_engines, own_role, other_role).
    own_side = LLM (1) + RIVAL_KINDS (hand-designed/learned); other_side =
    the fixed COUNTERPART_KINDS roster."""
    other_role = "shipper" if role == "carrier" else "carrier"
    other_builder = build_shipper if role == "carrier" else build_carrier
    own_builder = build_carrier if role == "carrier" else build_shipper

    llm_id = f"{role}-llm-1"
    own = {llm_id: build_llm_engine(role, llm_id, model, host, timeout_s, max_retries, think, log_sink)}
    for kind in RIVAL_KINDS:
        agent_id = f"{role}-{kind}-1"
        own[agent_id] = own_builder(kind, agent_id)

    counterpart = {}
    for kind in COUNTERPART_KINDS:
        agent_id = f"{other_role}-{kind}-1"
        counterpart[agent_id] = other_builder(kind, agent_id)

    return own, counterpart, llm_id


def load_completed_days(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["day_index"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_one_combo(role: str, model: str, days: int, base_seed: int, host: str, timeout_s: float, max_retries: int, think):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{sanitize(model)}_{role}"
    outcomes_path = OUT_DIR / f"{tag}_outcomes.jsonl"
    calls_path = OUT_DIR / f"{tag}_calls.jsonl"

    completed = load_completed_days(outcomes_path)
    print(f"[{tag}] {len(completed)}/{days} days already done, resuming" if completed else f"[{tag}] starting fresh")

    with outcomes_path.open("a", encoding="utf-8") as outcomes_f, calls_path.open("a", encoding="utf-8") as calls_f:
        for day_index in range(days):
            if day_index in completed:
                continue
            log_sink: list = []
            seed = base_seed + day_index
            t0 = time.monotonic()
            try:
                own, counterpart, llm_id = build_roster(role, model, host, timeout_s, max_retries, think, log_sink)
                if role == "carrier":
                    result = run_round(round_index=day_index, seed=seed, shipper_engines=counterpart, carrier_engines=own, n_negotiations=1)
                else:
                    result = run_round(round_index=day_index, seed=seed, shipper_engines=own, carrier_engines=counterpart, n_negotiations=1)
                neg = result.negotiations[0]
                outcomes = neg.carrier_outcomes if role == "carrier" else neg.shipper_outcomes
                for o in outcomes:
                    agent_id = o.carrier_id if role == "carrier" else o.shipper_id
                    is_llm = agent_id == llm_id
                    row = {
                        "model": model, "tier": tier_of(model), "family": family_of(model),
                        "role": role, "day_index": day_index, "seed": seed,
                        "agent_id": agent_id, "is_llm": is_llm, "matched": o.matched,
                        "outcome": o.margin if role == "carrier" else o.profit,
                        "elapsed_s": time.monotonic() - t0,
                    }
                    outcomes_f.write(json.dumps(row) + "\n")
                outcomes_f.flush()
            except Exception as exc:  # noqa: BLE001 -- one bad day must not kill an hours-long remote run
                print(f"[{tag}] day {day_index} FAILED: {exc}")
                traceback.print_exc()
                outcomes_f.write(json.dumps({
                    "model": model, "tier": tier_of(model), "family": family_of(model),
                    "role": role, "day_index": day_index, "seed": seed,
                    "agent_id": None, "is_llm": None, "matched": None, "outcome": None,
                    "elapsed_s": time.monotonic() - t0, "error": str(exc),
                }) + "\n")
                outcomes_f.flush()

            for entry in log_sink:
                entry["tier"] = tier_of(entry.get("model", model))
                entry["family"] = family_of(entry.get("model", model))
                calls_f.write(json.dumps(entry) + "\n")
            calls_f.flush()

            if day_index % 5 == 0 or day_index == days - 1:
                print(f"[{tag}] day {day_index + 1}/{days} done ({time.monotonic() - t0:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-vs-engines validation study (Ollama-backed)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Ollama model tags, must already be `ollama pull`-ed")
    parser.add_argument("--roles", nargs="+", default=["carrier", "shipper"], choices=["carrier", "shipper"])
    parser.add_argument("--days", type=int, default=40, help="independent negotiations per (model, role) combo")
    parser.add_argument("--base-seed", type=int, default=90_000_000)
    parser.add_argument("--host", type=str, default="http://localhost:11434")
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--think", type=str, default=None,
        help="passed through to Ollama's `think` field -- 'false'/'true' (case-insensitive) are parsed into "
             "real JSON booleans (Ollama's API expects a boolean or 'low'/'medium'/'high', not the literal "
             "string 'false'); any other value (e.g. 'low') is passed through unchanged; omit for model default.",
    )
    args = parser.parse_args()

    think: object = args.think
    if isinstance(think, str) and think.lower() in ("false", "true"):
        think = think.lower() == "true"

    print(f"models={args.models} roles={args.roles} days={args.days} host={args.host} think={think!r}")
    print(f"own-side rivals={RIVAL_KINDS}  fixed counterpart roster={COUNTERPART_KINDS}")

    for role in args.roles:
        for model in args.models:
            # Distinct base seed per (role, model) so no two combos ever draw
            # the same sequence of market days. Must be reproducible across
            # process restarts (this script is explicitly designed to resume
            # after a crash/reboot, see module docstring) -- Python's builtin
            # hash() on a tuple of strings is NOT that: it is salted per
            # process by PYTHONHASHSEED randomization, so two runs of the
            # exact same command used to get two different seeds for the same
            # combo. Discovered after the fact from genuinely divergent
            # outcome rows sharing the same day_index in two output files
            # (confirmed via dedup); zlib.crc32 on a fixed UTF-8 encoding is
            # deterministic across processes and interpreter versions.
            combo_key = f"{role}:{model}".encode("utf-8")
            combo_seed = args.base_seed + zlib.crc32(combo_key) % 1_000_000
            run_one_combo(role, model, args.days, combo_seed, args.host, args.timeout_s, args.max_retries, think)

    print("Done. Analyze results with notebooks/analyze_llm_validation.ipynb")


if __name__ == "__main__":
    main()
