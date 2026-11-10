"""Per-negotiation shipper scoring and round-level standings aggregation.

Rule (REVIEW_NOTES.md "Scoring"): points +6/+3/+1/0 by profit rank, but only
among shippers who both CLOSED and were MATCHED (and did not decline to
ship). Unmatched shippers, shippers with zero closes, and shippers who
declined a negative-profit match all score 0 and occupy no ranked slot.
Ties share the mean of the point values for the positions they jointly
occupy.
"""
from __future__ import annotations

from typing import Dict, List

from wehaveadeal.models import CarrierNegotiationOutcome, NegotiationResult, ShipperNegotiationOutcome

POINTS_SCHEDULE = (6.0, 3.0, 1.0)
DEFAULT_ROUND_BONUS = 200.0


def _point_for_rank(rank: int) -> float:
    """rank is 1-indexed; ranks beyond the schedule length score 0."""
    if rank <= len(POINTS_SCHEDULE):
        return POINTS_SCHEDULE[rank - 1]
    return 0.0


def score_shippers(outcomes: List[ShipperNegotiationOutcome]) -> List[ShipperNegotiationOutcome]:
    """Mutates and returns `outcomes` with `.points` set per the rule above."""
    for o in outcomes:
        o.points = 0.0

    eligible = [o for o in outcomes if o.matched and not o.declined_to_ship]
    eligible.sort(key=lambda o: o.profit, reverse=True)

    i = 0
    n = len(eligible)
    while i < n:
        j = i
        while j < n and eligible[j].profit == eligible[i].profit:
            j += 1
        tie_group = eligible[i:j]
        ranks = range(i + 1, j + 1)  # 1-indexed positions this tie group occupies
        shared_points = sum(_point_for_rank(r) for r in ranks) / len(tie_group)
        for o in tie_group:
            o.points = shared_points
        i = j

    return outcomes


def build_shipper_standings(negotiations: List[NegotiationResult], shipper_ids: List[str]) -> Dict[str, dict]:
    standings = {s: {"points": 0.0, "profit": 0.0, "closes": 0, "matches": 0} for s in shipper_ids}
    for neg in negotiations:
        for o in neg.shipper_outcomes:
            st = standings[o.shipper_id]
            st["points"] += o.points
            st["profit"] += o.profit
            if o.matched:
                st["matches"] += 1
        for t in neg.tables:
            if t.closed:
                standings[t.shipper_id]["closes"] += 1
    return standings


def build_carrier_standings(negotiations: List[NegotiationResult], carrier_ids: List[str]) -> Dict[str, dict]:
    standings = {c: {"margin": 0.0, "matches": 0, "negotiations": len(negotiations)} for c in carrier_ids}
    for neg in negotiations:
        for o in neg.carrier_outcomes:
            st = standings[o.carrier_id]
            st["margin"] += o.margin
            if o.matched:
                st["matches"] += 1
    for st in standings.values():
        st["occupancy"] = st["matches"] / st["negotiations"] if st["negotiations"] else 0.0
    return standings


def apply_round_bonus(
    shipper_standings: Dict[str, dict], bonus: float = DEFAULT_ROUND_BONUS
) -> Dict[str, dict]:
    """+bonus points to the shipper with the single highest accumulated profit."""
    if not shipper_standings:
        return shipper_standings
    leader = max(shipper_standings.items(), key=lambda kv: kv[1]["profit"])[0]
    shipper_standings[leader]["points"] += bonus
    shipper_standings[leader]["got_bonus"] = True
    return shipper_standings


def determine_round_winner_shipper(shipper_standings: Dict[str, dict]) -> str:
    return max(shipper_standings.items(), key=lambda kv: (kv[1]["points"], kv[1]["profit"]))[0]


def determine_round_winner_carrier(carrier_standings: Dict[str, dict]) -> str:
    return max(carrier_standings.items(), key=lambda kv: (kv[1]["margin"], kv[1]["occupancy"]))[0]


def carrier_outcome_margin(rate: float, refprice_mean: float) -> float:
    return rate - refprice_mean
