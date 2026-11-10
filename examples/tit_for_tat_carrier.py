"""A reactive carrier: mirrors the shipper's last concession instead of following a clock.

Shows how to read the table history (`state.history`) and your own portfolio
(`state.my_portfolio` -- your other concurrent tables in this negotiation).
"""
from wehaveadeal import ACCEPT, REJECT, Action, DecisionEngine, Move, TableState, legal_offer_range, offer


class TitForTatCarrier(DecisionEngine):
    role = "carrier"
    engine_name = "TitForTatCarrier"

    def next_move(self, state: TableState) -> Move:
        lo, hi = legal_offer_range(state)
        band_top = state.refprice_mean + 0.5 * state.sigma

        # Already closed a deal on another table? Then this one is only worth it at a premium.
        closed_elsewhere = any(t.is_closed for t in state.my_portfolio)
        floor = state.refprice_mean if closed_elsewhere else lo

        if state.opp_last_offer is not None:
            asking = state.my_last_offer if state.my_last_offer is not None else band_top
            if state.opp_last_offer >= max(floor, asking) - 1e-9:
                return ACCEPT
            last_chance = state.my_turns_used >= state.turns_per_side - 1
            if last_chance and state.opp_last_offer >= floor:
                return ACCEPT
        if state.my_last_offer is None:
            return offer(hi)
        if lo > hi:
            return REJECT

        # Concede as much as the shipper conceded last time (at least a small step).
        theirs = [t.amount for t in state.history if t.actor == "shipper" and t.action == Action.OFFER]
        their_step = theirs[-1] - theirs[-2] if len(theirs) >= 2 else 0.05 * state.sigma
        target = state.my_last_offer - max(their_step, 0.02 * state.sigma)
        return offer(min(max(target, max(lo, floor)), hi))
