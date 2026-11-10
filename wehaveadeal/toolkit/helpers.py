"""Small helpers for agent authors: what may I legally offer right now?"""
from __future__ import annotations

from wehaveadeal.models import TableState


def legal_offer_range(state: TableState) -> tuple[float, float]:
    """(lowest, highest) amount you may OFFER at this decision point.

    Carrier: inside the hard band [refprice_mean - sigma/2, refprice_mean + sigma/2],
             and never above your own previous offer (concessions only go down).
    Shipper: never above your reservation price (sale_price - goods_cost),
             and never below your own previous offer (concessions only go up).

    If lowest > highest there is no legal offer left: ACCEPT (if there is a
    counter-offer on the table) or REJECT.
    """
    if state.role == "carrier":
        lo = state.refprice_mean - 0.5 * state.sigma
        hi = state.refprice_mean + 0.5 * state.sigma
        if state.my_last_offer is not None:
            hi = min(hi, state.my_last_offer)
        return lo, hi
    hi = state.sale_price - state.goods_cost
    lo = state.my_last_offer if state.my_last_offer is not None else 0.0
    return lo, hi


def clamp_offer(state: TableState, amount: float) -> float:
    """`amount` pulled into the legal range (assumes the range is non-empty)."""
    lo, hi = legal_offer_range(state)
    return min(max(amount, lo), hi)
