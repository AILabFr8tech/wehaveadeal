"""Shared, mirrored value function for shipper and carrier.

Both roles' value functions are the exact structure from Strategy spec §7.1/7.2
(see the paper (docs/PROTOCOL.md) "Value function"): a heuristic blend of "what's
on the table" and "my own aspiration", discounted by sqrt time-pressure over
remaining turns, adjusted by an assignment/occupancy probability. The only
difference between the two roles is (a) which anchor is "worst" vs "best" for
normalization, and (b) the carrier's extra branch for whose turn is next.

There is exactly ONE implementation (`_value_of_position`) reused unchanged
by every value-function-aware engine on both roles (CarrierRV/IS/OPT/OPT2
and ShipperRV/IS/OPT/OPT2) -- per the project brief's explicit requirement
not to duplicate this logic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from wehaveadeal.models import PortfolioTableSnapshot, TableState


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def portfolio_occupancy(
    portfolio: "List[PortfolioTableSnapshot]", effective_threshold: float, higher_is_better: bool = True
) -> float:
    """Occupancy/assignment proxy shared by every value-function-aware
    engine on both roles (CarrierRV/IS/OPT/OPT2 and ShipperRV/IS/OPT/OPT2):
    fraction of my OTHER live tables in this negotiation that look
    "promising" -- already closed, or currently sitting at/beyond my
    tactical threshold. `higher_is_better` picks the direction: True for a
    carrier (a promising sibling table has an offer at/above its tactical
    floor), False for a shipper (a promising sibling table has an offer
    at/below its tactical ceiling). Phase-1 simplification of Strategy
    spec §3.6/§6.4's full cross-round demand model: no history, just my own
    visible sibling tables (which either role legitimately observes -- see
    REVIEW_NOTES.md "Information asymmetry").
    """
    if not portfolio:
        return 0.5  # nothing to compare against: neutral
    promising = 0
    for p in portfolio:
        if p.is_closed and p.closed_rate is not None:
            promising += 1  # already secured elsewhere
        elif not p.is_impasse and p.latest_offer_on_table is not None:
            beats_threshold = (
                p.latest_offer_on_table >= effective_threshold
                if higher_is_better
                else p.latest_offer_on_table <= effective_threshold
            )
            if beats_threshold:
                promising += 1
    return promising / len(portfolio)


def shipper_route_is_feasible(state: "TableState") -> bool:
    """True unless even the best-case estimated carrier floor already
    exceeds this shipper's hard RP -- i.e. no legal deal can exist on this
    route no matter how the negotiation unfolds (Strategy spec §13.0,
    deliberate walk-away). This is a market-feasibility check, not a
    strategy choice, so every shipper engine shares this one implementation
    rather than re-deriving it (first written for ShipperR, reused as-is by
    ShipperRV/IS/OPT/OPT2)."""
    rp = state.sale_price - state.goods_cost
    estimated_carrier_floor = state.refprice_est - 0.5 * state.sigma
    return estimated_carrier_floor <= rp


def value_bar(price: float, worst_anchor: float, best_anchor: float) -> float:
    """Normalize `price` between 0 (worst_anchor) and 1 (best_anchor).

    Shipper usage: worst_anchor=RP (ceiling), best_anchor=target (ambitious low price).
    Carrier usage: worst_anchor=RP_t (floor), best_anchor=target_t (ambitious high price).
    Same function, swapped anchors -- this IS the "mirrored normalization".
    """
    if best_anchor == worst_anchor:
        return 1.0
    return clamp01((price - worst_anchor) / (best_anchor - worst_anchor))


def _value_of_position(
    *,
    worst_anchor: float,
    best_anchor: float,
    terminal: bool,
    close_price: Optional[float],
    offer_on_table: Optional[float],
    my_last_move: Optional[float],
    turn: int,
    turns_max: int,
    weight: float,
    assignment_prob: float,
    conditional_heuristic: bool = False,
    opponent_decides_next: bool = False,
) -> float:
    if terminal:
        assert close_price is not None
        return value_bar(close_price, worst_anchor, best_anchor)

    if offer_on_table is None:
        v_offer = 0.5
    else:
        v_offer = value_bar(offer_on_table, worst_anchor, best_anchor)

    if my_last_move is None:
        v_aspiration = 1.0
    else:
        v_aspiration = value_bar(my_last_move, worst_anchor, best_anchor)

    if conditional_heuristic and opponent_decides_next:
        heuristic = v_aspiration
    else:
        heuristic = (v_offer + v_aspiration) / 2.0

    turns_remaining = max(turns_max - turn, 0)
    if turns_remaining == 0:
        return 0.0
    pressure = (turns_remaining / turns_max) ** 0.5
    adjustment = (1 - weight) + weight * assignment_prob
    return heuristic * pressure * adjustment


def shipper_value(
    *,
    rp: float,
    target: float,
    terminal: bool = False,
    close_price: Optional[float] = None,
    carrier_offer_on_table: Optional[float] = None,
    my_last_offer: Optional[float] = None,
    turn: int,
    turns_max: int,
    weight_assignment: float,
    p_assignment: float,
) -> float:
    """Shipper position value (Strategy spec §7.1). Lower price is better for the
    shipper, so worst=RP (ceiling), best=target (ambitious low price)."""
    return _value_of_position(
        worst_anchor=rp,
        best_anchor=target,
        terminal=terminal,
        close_price=close_price,
        offer_on_table=carrier_offer_on_table,
        my_last_move=my_last_offer,
        turn=turn,
        turns_max=turns_max,
        weight=weight_assignment,
        assignment_prob=p_assignment,
        conditional_heuristic=False,
    )


def carrier_value(
    *,
    rp_t: float,
    target_t: float,
    terminal: bool = False,
    close_price: Optional[float] = None,
    shipper_offer_on_table: Optional[float] = None,
    my_last_counter: Optional[float] = None,
    turn: int,
    turns_max: int,
    weight_occupancy: float,
    p_occupancy: float,
    opponent_decides_next: bool = False,
) -> float:
    """Carrier position value (Strategy spec §7.2). Higher price is better
    for the carrier, so worst=RP_t (floor), best=target_t (ambitious high price)."""
    return _value_of_position(
        worst_anchor=rp_t,
        best_anchor=target_t,
        terminal=terminal,
        close_price=close_price,
        offer_on_table=shipper_offer_on_table,
        my_last_move=my_last_counter,
        turn=turn,
        turns_max=turns_max,
        weight=weight_occupancy,
        assignment_prob=p_occupancy,
        conditional_heuristic=True,
        opponent_decides_next=opponent_decides_next,
    )
