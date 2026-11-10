"""Self-play data collection for imitation learning.

Wraps a chosen "teacher" architecture with a recorder and runs it through
`wehaveadeal.simulate.run_round` against a roster of opponents that cycles
through every other architecture, so the teacher gets observed making
decisions against a genuine variety of counterpart behavior rather than
just mirrors of itself. One row is written per OFFER *or* ACCEPT decision
the teacher made: its `wehaveadeal.ml.features.extract_features`, a `label`
(the position on the [0,1] band of whichever price the row is about -- the
new counter for an OFFER row, the price just accepted for an ACCEPT row),
and two extra columns that only matter for `CarrierML2`/`ShipperML2`
(see `wehaveadeal/agents/carrier_ml2.py`):

  `accepted`         -- 1 if the teacher accepted here, 0 if it countered.
  `accept_eligible`  -- 1 if the opponent's current offer was even legal to
                        accept at this decision point (accepting an
                        illegal offer was never on the table, so such rows
                        aren't real "accept vs. continue" examples and the
                        accept-classifier's training script filters them
                        out; the offer-regressor ignores these two columns
                        entirely and keeps working exactly as before).

Deliberately does not touch `wehaveadeal/simulate.py`: `RecordingEngine` wraps
the existing `DecisionEngine` interface from the outside -- exactly what
that interface was built to allow (see `wehaveadeal/agents/base.py`) -- so every
legality/matching/scoring code path this data depends on stays exactly as
already tested, with zero new surface area in the simulation core itself.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from wehaveadeal.agents.base import DecisionEngine
from wehaveadeal.ml.features import FEATURE_NAMES, extract_features, role_band
from wehaveadeal.models import Action, Move, TableState
from wehaveadeal.simulate import CARRIER_KINDS, SHIPPER_KINDS, build_carrier, build_shipper, run_round
from wehaveadeal.value_functions import value_bar


class RecordingEngine(DecisionEngine):
    """Transparent wrapper: delegates every decision to `inner` unchanged,
    and additionally appends (state, move) to `sink` for every OFFER or
    ACCEPT it makes (a REJECT is never a real "what would you offer / would
    you accept" example -- it's the turn-limit or route-infeasibility
    safety net every engine already has unconditionally, not a learned
    decision -- so those aren't recorded). Behaviorally invisible to the
    rest of the engine -- matching, scoring, and legality validation see
    exactly the same moves `inner` would have produced on its own."""

    def __init__(self, inner: DecisionEngine, sink: List[Tuple[TableState, Move]]) -> None:
        super().__init__(inner.agent_id, frozen=inner.frozen)
        self.engine_name = inner.engine_name
        self.version = inner.version
        self._inner = inner
        self._sink = sink

    def next_move(self, state: TableState) -> Move:
        move = self._inner.next_move(state)
        if move.action in (Action.OFFER, Action.ACCEPT):
            self._sink.append((state, move))
        return move


def collect_rows(
    teacher_kind: str,
    role: str,
    seeds: Iterable[int],
    opponent_kinds: Sequence[str],
    n_negotiations: int = 30,
    n_teacher_agents: int = 3,
) -> List[dict]:
    """One round per seed: `n_teacher_agents` copies of `teacher_kind` on
    `role`'s side (recorded), one of each `opponent_kinds` on the other
    side (not recorded) -- so every seed's round already pits the teacher
    against the full opponent roster at once."""
    rows: List[dict] = []
    for seed in seeds:
        sink: List[Tuple[TableState, Move]] = []
        teacher_kinds = tuple([teacher_kind] * n_teacher_agents)

        def _build_side(kinds: Sequence[str], builder, prefix: str, record: bool) -> Dict[str, DecisionEngine]:
            counts: Dict[str, int] = {}
            engines: Dict[str, DecisionEngine] = {}
            for kind in kinds:
                counts[kind] = counts.get(kind, 0) + 1
                agent_id = f"{prefix}-{kind}-{counts[kind]}"
                inner = builder(kind, agent_id)
                engines[agent_id] = RecordingEngine(inner, sink) if record else inner
            return engines

        if role == "carrier":
            carriers = _build_side(teacher_kinds, build_carrier, "carrier", record=True)
            shippers = _build_side(opponent_kinds, build_shipper, "shipper", record=False)
        else:
            shippers = _build_side(teacher_kinds, build_shipper, "shipper", record=True)
            carriers = _build_side(opponent_kinds, build_carrier, "carrier", record=False)

        run_round(round_index=0, seed=seed, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=n_negotiations)

        for state, move in sink:
            worst, best = role_band(state)
            feats = extract_features(state)
            higher_is_better = state.role == "carrier"
            opp_offer = state.opp_last_offer
            accept_eligible = opp_offer is not None and (
                (opp_offer >= worst) if higher_is_better else (opp_offer <= worst)
            )
            if move.action == Action.ACCEPT:
                # The row is "about" the price just accepted -- there is no
                # alternative offer to measure, since the teacher chose not
                # to make one.
                distance = value_bar(opp_offer, worst, best)
                accepted = 1
            else:
                distance = value_bar(move.amount, worst, best)
                accepted = 0
            rows.append(
                {
                    **feats,
                    "label": distance,
                    "accepted": accepted,
                    "accept_eligible": int(accept_eligible),
                    "role": role,
                    "teacher_kind": teacher_kind,
                    "seed": seed,
                }
            )
    return rows


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(FEATURE_NAMES) + ["label", "accepted", "accept_eligible", "role", "teacher_kind", "seed"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate imitation-learning training data via self-play")
    parser.add_argument("--role", choices=["carrier", "shipper"], required=True)
    parser.add_argument("--teacher", required=True, help=f"carrier kinds: {CARRIER_KINDS}; shipper kinds: {SHIPPER_KINDS}")
    parser.add_argument("--seeds", type=int, default=200, help="number of distinct self-play seeds (rounds)")
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--negotiations", type=int, default=30, help="negotiations per seed")
    parser.add_argument("--teacher-agents", type=int, default=3)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    # Learned kinds are excluded from the opponent pool: at data-collection
    # time there is (at best) no trained model yet for them, and even once
    # one exists, imitation data should reflect the established
    # hand-designed architectures' behavior, not create a circular
    # dependency on another learned engine.
    learned_kinds = ("ml", "ml2", "rl")
    opponent_kinds = [k for k in (SHIPPER_KINDS if args.role == "carrier" else CARRIER_KINDS) if k not in learned_kinds]
    seeds = range(args.base_seed, args.base_seed + args.seeds)
    rows = collect_rows(
        teacher_kind=args.teacher,
        role=args.role,
        seeds=seeds,
        opponent_kinds=opponent_kinds,
        n_negotiations=args.negotiations,
        n_teacher_agents=args.teacher_agents,
    )
    out = Path(args.out) if args.out else Path("data/ml") / f"{args.role}_{args.teacher}_imitation.csv"
    write_csv(rows, out)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
