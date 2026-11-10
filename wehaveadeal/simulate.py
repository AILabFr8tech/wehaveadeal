"""Simulation runner: Tournament -> Round -> Negotiation -> Table.

Runs all 12 tables of a negotiation in turn-synchronized lockstep (all
shippers act on odd global turns, all carriers on even ones) so that each
agent's own-portfolio visibility (`TableState.my_portfolio`) reflects the
true concurrent state of its other tables at each decision point -- matching
WHAD spec's parallel bilateral tables (Phase 1).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from wehaveadeal.agents.base import DecisionEngine
from wehaveadeal.market import MarketState
from wehaveadeal.matching import gale_shapley
from wehaveadeal.models import (
    Action,
    CarrierNegotiationOutcome,
    MatchingResult,
    Move,
    NegotiationResult,
    PortfolioTableSnapshot,
    RoundResult,
    RouteInfo,
    ShipperNegotiationOutcome,
    TableResult,
    TableState,
    TurnRecord,
)
from wehaveadeal.scoring import (
    DEFAULT_ROUND_BONUS,
    apply_round_bonus,
    build_carrier_standings,
    build_shipper_standings,
    carrier_outcome_margin,
    determine_round_winner_carrier,
    determine_round_winner_shipper,
    score_shippers,
)

TURNS_MAX = 12  # 6 per side (WHAD spec §3)


class IllegalMoveError(ValueError):
    pass


def _validate_move(move: Move, state: TableState) -> None:
    """Hard-band / monotonic-concession legality check (WHAD spec §6.2, Strategy
    spec §10.0: "an illegal action counts as a forfeit and is unacceptable in
    any engine"). Any future engine (IS/PPO/LLM) is checked the same way."""
    if move.action != Action.OFFER:
        return
    amount = move.amount
    tol = 1e-6
    if state.role == "shipper":
        rp = state.sale_price - state.goods_cost
        if amount > rp + tol:
            raise IllegalMoveError(f"shipper offer {amount} exceeds RP {rp}")
        if state.my_last_offer is not None and amount < state.my_last_offer - tol:
            raise IllegalMoveError("shipper concession is non-monotonic (offer decreased)")
    else:
        rp_t = state.refprice_mean - 0.5 * state.sigma
        target_t = state.refprice_mean + 0.5 * state.sigma
        if amount < rp_t - tol or amount > target_t + tol:
            raise IllegalMoveError(f"carrier offer {amount} outside hard band [{rp_t}, {target_t}]")
        if state.my_last_offer is not None and amount > state.my_last_offer + tol:
            raise IllegalMoveError("carrier concession is non-monotonic (counter increased)")


@dataclass
class _TableRunner:
    shipper_id: str
    carrier_id: str
    sigma: float
    history: List[TurnRecord] = field(default_factory=list)
    shipper_turns_used: int = 0
    carrier_turns_used: int = 0
    last_shipper_offer: Optional[float] = None
    last_carrier_offer: Optional[float] = None
    closed: bool = False
    impasse: bool = False
    rate: Optional[float] = None
    turns_to_close: Optional[int] = None

    @property
    def table_id(self) -> str:
        return f"{self.shipper_id}::{self.carrier_id}"

    def is_done(self) -> bool:
        return self.closed or self.impasse

    def apply(self, global_turn: int, actor_role: str, move: Move) -> None:
        self.history.append(
            TurnRecord(turn=global_turn, actor=actor_role, action=move.action, amount=move.amount)
        )
        if move.action == Action.ACCEPT:
            accepted_rate = self.last_carrier_offer if actor_role == "shipper" else self.last_shipper_offer
            if accepted_rate is None:
                raise IllegalMoveError(f"{actor_role} accepted with nothing on the table")
            self.closed = True
            self.rate = accepted_rate
            self.turns_to_close = global_turn
        elif move.action == Action.REJECT:
            self.impasse = True
        elif move.action == Action.OFFER:
            if actor_role == "shipper":
                self.last_shipper_offer = move.amount
                self.shipper_turns_used += 1
            else:
                self.last_carrier_offer = move.amount
                self.carrier_turns_used += 1

    def to_result(self) -> TableResult:
        return TableResult(
            shipper_id=self.shipper_id,
            carrier_id=self.carrier_id,
            closed=self.closed,
            impasse=self.impasse or not self.closed,
            rate=self.rate,
            turns_to_close=self.turns_to_close,
            history=list(self.history),
        )


def _portfolio_snapshot(
    siblings: List[_TableRunner], viewer_role: str, exclude: _TableRunner
) -> List[PortfolioTableSnapshot]:
    snaps = []
    for r in siblings:
        if r is exclude:
            continue
        if viewer_role == "carrier":
            counterpart_id, latest = r.shipper_id, r.last_shipper_offer
        else:
            counterpart_id, latest = r.carrier_id, r.last_carrier_offer
        snaps.append(
            PortfolioTableSnapshot(
                counterpart_id=counterpart_id,
                is_closed=r.closed,
                is_impasse=r.impasse,
                closed_rate=r.rate if r.closed else None,
                latest_offer_on_table=latest,
            )
        )
    return snaps


def _score_shipper_outcomes(
    shipper_ids: List[str], matching: MatchingResult, sale_price: float, goods_cost: float
) -> List[ShipperNegotiationOutcome]:
    matched_map = {m.shipper_id: m for m in matching.matches}
    outcomes = []
    for s in shipper_ids:
        m = matched_map.get(s)
        if m is None:
            outcomes.append(ShipperNegotiationOutcome(shipper_id=s, matched=False, profit=0.0))
            continue
        profit = sale_price - (m.rate + goods_cost)
        if profit < 0:
            # No-ship option (WHAD spec, Phase 3): decline, profit -> 0, counts as
            # "no transport" for scoring; carrier is released, not reassigned.
            outcomes.append(
                ShipperNegotiationOutcome(shipper_id=s, matched=False, profit=0.0, declined_to_ship=True)
            )
        else:
            outcomes.append(
                ShipperNegotiationOutcome(
                    shipper_id=s, matched=True, carrier_id=m.carrier_id, rate=m.rate, profit=profit
                )
            )
    return outcomes


def _score_carrier_outcomes(
    carrier_ids: List[str], shipper_outcomes: List[ShipperNegotiationOutcome], routes: Dict[str, RouteInfo]
) -> List[CarrierNegotiationOutcome]:
    matched_by_carrier = {o.carrier_id: o for o in shipper_outcomes if o.matched}
    outcomes = []
    for c in carrier_ids:
        o = matched_by_carrier.get(c)
        if o is None:
            outcomes.append(CarrierNegotiationOutcome(carrier_id=c, matched=False, margin=0.0))
        else:
            margin = carrier_outcome_margin(o.rate, routes[o.shipper_id].refprice_mean)
            outcomes.append(
                CarrierNegotiationOutcome(
                    carrier_id=c, matched=True, shipper_id=o.shipper_id, rate=o.rate, margin=margin
                )
            )
    return outcomes


def stream_negotiation(
    negotiation_index: int,
    seed: int,
    sale_price: float,
    goods_cost: float,
    shipper_engines: Dict[str, DecisionEngine],
    carrier_engines: Dict[str, DecisionEngine],
    routes: Dict[str, RouteInfo],
    refprice_est: Dict[str, float],
):
    """Generator form of one negotiation's 12 tables: yields one event dict
    per applied turn (`{"kind": "turn", ...}`), then a final
    `{"kind": "result", "negotiation": NegotiationResult}` event.

    This is the single implementation of the turn loop -- both the batch
    `run_negotiation` (below) and the live SSE API (wehaveadeal/api.py) drive this
    same generator, so there is exactly one place the WHAD spec, Phase 1 protocol is
    encoded, whether played back instantly or streamed turn-by-turn.
    """
    shipper_ids = list(shipper_engines.keys())
    carrier_ids = list(carrier_engines.keys())

    runners: List[_TableRunner] = []
    by_shipper: Dict[str, List[_TableRunner]] = {s: [] for s in shipper_ids}
    by_carrier: Dict[str, List[_TableRunner]] = {c: [] for c in carrier_ids}
    for s in shipper_ids:
        for c in carrier_ids:
            r = _TableRunner(shipper_id=s, carrier_id=c, sigma=routes[s].refprice_sigma)
            runners.append(r)
            by_shipper[s].append(r)
            by_carrier[c].append(r)

    for global_turn in range(1, TURNS_MAX + 1):
        acting_role = "shipper" if global_turn % 2 == 1 else "carrier"
        for r in runners:
            if r.is_done():
                continue
            route = routes[r.shipper_id]
            if acting_role == "shipper":
                state = TableState(
                    role="shipper",
                    table_id=r.table_id,
                    negotiation_seed=seed,
                    turn=global_turn,
                    turns_max=TURNS_MAX,
                    my_turns_used=r.shipper_turns_used,
                    opp_turns_used=r.carrier_turns_used,
                    my_last_offer=r.last_shipper_offer,
                    opp_last_offer=r.last_carrier_offer,
                    history=list(r.history),
                    sigma=route.refprice_sigma,
                    sale_price=sale_price,
                    goods_cost=goods_cost,
                    refprice_est=refprice_est[r.shipper_id],
                    my_portfolio=_portfolio_snapshot(by_shipper[r.shipper_id], "shipper", r),
                )
                move = shipper_engines[r.shipper_id].next_move(state)
            else:
                state = TableState(
                    role="carrier",
                    table_id=r.table_id,
                    negotiation_seed=seed,
                    turn=global_turn,
                    turns_max=TURNS_MAX,
                    my_turns_used=r.carrier_turns_used,
                    opp_turns_used=r.shipper_turns_used,
                    my_last_offer=r.last_carrier_offer,
                    opp_last_offer=r.last_shipper_offer,
                    history=list(r.history),
                    sigma=route.refprice_sigma,
                    refprice_mean=route.refprice_mean,
                    my_portfolio=_portfolio_snapshot(by_carrier[r.carrier_id], "carrier", r),
                )
                move = carrier_engines[r.carrier_id].next_move(state)

            _validate_move(move, state)
            r.apply(global_turn, acting_role, move)

            yield {
                "kind": "turn",
                "table_id": r.table_id,
                "shipper_id": r.shipper_id,
                "carrier_id": r.carrier_id,
                "turn": global_turn,
                "actor": acting_role,
                "action": move.action.value,
                "amount": move.amount,
                "closed": r.closed,
                "impasse": r.impasse,
            }

    tables = [r.to_result() for r in runners]
    matching = gale_shapley(tables, shipper_ids, seed=seed)

    shipper_outcomes = _score_shipper_outcomes(shipper_ids, matching, sale_price, goods_cost)
    score_shippers(shipper_outcomes)
    carrier_outcomes = _score_carrier_outcomes(carrier_ids, shipper_outcomes, routes)

    result = NegotiationResult(
        negotiation_index=negotiation_index,
        seed=seed,
        sale_price=sale_price,
        goods_cost=goods_cost,
        routes={s: routes[s] for s in shipper_ids},
        tables=tables,
        matching=matching,
        shipper_outcomes=shipper_outcomes,
        carrier_outcomes=carrier_outcomes,
    )
    yield {"kind": "result", "negotiation": result}


def run_negotiation(
    negotiation_index: int,
    seed: int,
    sale_price: float,
    goods_cost: float,
    shipper_engines: Dict[str, DecisionEngine],
    carrier_engines: Dict[str, DecisionEngine],
    routes: Dict[str, RouteInfo],
    refprice_est: Dict[str, float],
) -> NegotiationResult:
    """Batch form: drive stream_negotiation to completion and return only
    the final NegotiationResult."""
    result: Optional[NegotiationResult] = None
    for event in stream_negotiation(
        negotiation_index, seed, sale_price, goods_cost, shipper_engines, carrier_engines, routes, refprice_est
    ):
        if event["kind"] == "result":
            result = event["negotiation"]
    assert result is not None
    return result


def run_round(
    round_index: int,
    seed: int,
    shipper_engines: Dict[str, DecisionEngine],
    carrier_engines: Dict[str, DecisionEngine],
    n_negotiations: int = 100,
    mode: str = "frozen",
    bonus: float = DEFAULT_ROUND_BONUS,
) -> RoundResult:
    market = MarketState(seed, n_negotiations)
    shipper_ids = list(shipper_engines.keys())
    carrier_ids = list(carrier_engines.keys())

    negotiations = []
    for n in range(n_negotiations):
        neg_rng = market.negotiation_rng(n)
        routes: Dict[str, RouteInfo] = {}
        refprice_est: Dict[str, float] = {}
        for s in shipper_ids:
            route = market.assign_route(neg_rng)
            routes[s] = route
            eps = market.sample_shipper_epsilon(route.refprice_sigma, neg_rng)
            refprice_est[s] = route.refprice_mean + eps

        neg_result = run_negotiation(
            negotiation_index=n,
            seed=market.negotiation_seed_int(n),
            sale_price=market.sale_prices[n],
            goods_cost=market.goods_costs[n],
            shipper_engines=shipper_engines,
            carrier_engines=carrier_engines,
            routes=routes,
            refprice_est=refprice_est,
        )
        negotiations.append(neg_result)

    shipper_standings = build_shipper_standings(negotiations, shipper_ids)
    apply_round_bonus(shipper_standings, bonus=bonus)
    carrier_standings = build_carrier_standings(negotiations, carrier_ids)

    return RoundResult(
        round_index=round_index,
        seed=seed,
        mode=mode,
        shipper_ids=shipper_ids,
        carrier_ids=carrier_ids,
        negotiations=negotiations,
        shipper_standings=shipper_standings,
        carrier_standings=carrier_standings,
        round_winner_shipper=determine_round_winner_shipper(shipper_standings),
        round_winner_carrier=determine_round_winner_carrier(carrier_standings),
    )


def run_tournament(
    n_rounds: int,
    base_seed: int,
    shipper_engines: Dict[str, DecisionEngine],
    carrier_engines: Dict[str, DecisionEngine],
    n_negotiations_per_round: int = 100,
    mode: str = "frozen",
) -> List[RoundResult]:
    """Thin loop over run_round with per-round seeds derived from base_seed.

    Phase 1 does not implement the full statistical protocol (common random
    numbers, duplicated/permuted tournaments, bootstrap CIs) -- see
    the paper (docs/PROTOCOL.md) -- this just gives TournamentRunner's shape.
    """
    import numpy as np

    seeds = np.random.SeedSequence(base_seed).generate_state(n_rounds)
    return [
        run_round(
            round_index=i,
            seed=int(seeds[i]),
            shipper_engines=shipper_engines,
            carrier_engines=carrier_engines,
            n_negotiations=n_negotiations_per_round,
            mode=mode,
        )
        for i in range(n_rounds)
    ]


def write_round_json(result: RoundResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"round_{result.round_index:03d}_seed_{result.seed}.json"
    path.write_text(result.model_dump_json(indent=2))
    return path


CARRIER_KINDS = ("r", "rv", "is", "opt", "opt2", "ml", "ml2", "rl", "rl2")
SHIPPER_KINDS = ("r", "rv", "is", "opt", "opt2", "ml", "ml2", "rl", "rl2")


def build_carrier(kind: str, agent_id: str) -> DecisionEngine:
    """Factory over the nine carrier architectures, keyed by the short
    name used in carrier IDs (`carrier-<kind>-<n>`) and in the CLI's
    `--carriers` flag. Kept separate from `default_agents` so callers can
    assemble any carrier mix of any size. `"ml"` (`CarrierML`) needs a
    trained model at `ml/carrier_ml_model.joblib` -- see
    `wehaveadeal/ml/collect.py` and `wehaveadeal/ml/train.py`; `"ml2"` (`CarrierML2`)
    needs both `ml/carrier_ml2_offer_model.joblib` and
    `ml/carrier_ml2_accept_model.joblib` -- see
    `wehaveadeal/ml/train_accept.py`; `"rl"` (`CarrierRL`) needs one at
    `rl/carrier_rl_model.zip` -- see `wehaveadeal/rl/env.py` and
    `wehaveadeal/rl/train.py`; `"rl2"` (`CarrierRL2`) needs one at
    `rl/carrier_rl2_model.zip` -- see `wehaveadeal/rl/league_env.py` and
    `wehaveadeal/rl/train_league.py`. All four raise a clear error at decision
    time if their file(s) don't exist yet."""
    from wehaveadeal.agents import CarrierIS, CarrierML, CarrierML2, CarrierOPT, CarrierOPT2, CarrierR, CarrierRL, CarrierRL2, CarrierRV

    builders = {
        "r": CarrierR,
        "rv": CarrierRV,
        "is": CarrierIS,
        "opt": CarrierOPT,
        "opt2": CarrierOPT2,
        "ml": CarrierML,
        "ml2": CarrierML2,
        "rl": CarrierRL,
        "rl2": CarrierRL2,
    }
    if kind not in builders:
        raise ValueError(f"unknown carrier kind {kind!r}; expected one of {CARRIER_KINDS}")
    return builders[kind](agent_id=agent_id)


def build_shipper(kind: str, agent_id: str) -> DecisionEngine:
    """Factory over the nine shipper architectures, mirroring
    `build_carrier` exactly -- same nine short names, keyed the same way,
    for carrier IDs (`shipper-<kind>-<n>`) and the CLI's `--shippers` flag."""
    from wehaveadeal.agents import ShipperIS, ShipperML, ShipperML2, ShipperOPT, ShipperOPT2, ShipperR, ShipperRL, ShipperRL2, ShipperRV

    builders = {
        "r": ShipperR,
        "rv": ShipperRV,
        "is": ShipperIS,
        "opt": ShipperOPT,
        "opt2": ShipperOPT2,
        "ml": ShipperML,
        "ml2": ShipperML2,
        "rl": ShipperRL,
        "rl2": ShipperRL2,
    }
    if kind not in builders:
        raise ValueError(f"unknown shipper kind {kind!r}; expected one of {SHIPPER_KINDS}")
    return builders[kind](agent_id=agent_id)


def default_agents(
    carrier_kinds: tuple[str, ...] = ("r", "r", "rv"), shipper_kinds: tuple[str, ...] = ("r", "r", "r", "r")
) -> tuple[Dict[str, DecisionEngine], Dict[str, DecisionEngine]]:
    """Shippers built from `shipper_kinds` against carriers built from
    `carrier_kinds`, one per entry on each side (default: the original
    4-ShipperR / R-R-RV 3-carrier mix, preserved so existing historical
    rounds stay reproducible). The original brief's 4-vs-3 shipper/carrier
    ratio isn't load-bearing anywhere else in the engine (matching, scoring
    and every agent are generic over shipper/carrier counts) -- both sides
    are exposed as kind-mix knobs so the live view can explore other
    scarcity ratios and other architecture mixes on either side, e.g. a
    2-entry `shipper_kinds` against a 6-entry `carrier_kinds` for an
    oversupplied carrier market, or the reverse for a scarce one."""
    if len(carrier_kinds) < 1:
        raise ValueError("at least 1 carrier is required")
    if len(shipper_kinds) < 1:
        raise ValueError("at least 1 shipper is required")

    counts: Dict[str, int] = {}
    shippers: Dict[str, DecisionEngine] = {}
    for kind in shipper_kinds:
        counts[kind] = counts.get(kind, 0) + 1
        agent_id = f"shipper-{kind}-{counts[kind]}"
        shippers[agent_id] = build_shipper(kind, agent_id)

    counts = {}
    carriers: Dict[str, DecisionEngine] = {}
    for kind in carrier_kinds:
        counts[kind] = counts.get(kind, 0) + 1
        agent_id = f"carrier-{kind}-{counts[kind]}"
        carriers[agent_id] = build_carrier(kind, agent_id)
    return shippers, carriers


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a We Have a Deal Round and write its log to data/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negotiations", type=int, default=100)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument(
        "--carriers",
        type=str,
        default="r,r,rv",
        help=f"comma-separated list of carrier kinds, from {CARRIER_KINDS} (default: r,r,rv)",
    )
    parser.add_argument(
        "--shippers",
        type=str,
        default="r,r,r,r",
        help=f"comma-separated list of shipper kinds, from {SHIPPER_KINDS} (default: r,r,r,r)",
    )
    args = parser.parse_args()

    carrier_kinds = tuple(k.strip() for k in args.carriers.split(","))
    shipper_kinds = tuple(k.strip() for k in args.shippers.split(","))
    shippers, carriers = default_agents(carrier_kinds, shipper_kinds)
    result = run_round(
        round_index=args.round_index,
        seed=args.seed,
        shipper_engines=shippers,
        carrier_engines=carriers,
        n_negotiations=args.negotiations,
    )
    path = write_round_json(result, Path(args.out))
    print(f"Wrote {path}")
    print(f"Round winner (shipper): {result.round_winner_shipper}")
    print(f"Round winner (carrier): {result.round_winner_carrier}")
    print(json.dumps(result.shipper_standings, indent=2))
    print(json.dumps(result.carrier_standings, indent=2))


if __name__ == "__main__":
    main()
