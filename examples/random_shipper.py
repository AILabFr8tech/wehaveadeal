"""The smallest possible legal agent: a shipper that offers a random legal amount
and sometimes accepts any counter-offer it can afford. A floor every real agent should beat.

    wehaveadeal benchmark examples/random_shipper.py --preset quick
"""
import random

from wehaveadeal import ACCEPT, REJECT, DecisionEngine, Move, TableState, legal_offer_range, offer


class RandomShipper(DecisionEngine):
    role = "shipper"
    engine_name = "RandomShipper"

    def next_move(self, state: TableState) -> Move:
        # Seed from the state, not a global RNG, so benchmark runs are reproducible.
        rng = random.Random(f"{state.negotiation_seed}:{state.table_id}:{state.turn}")
        reservation = state.sale_price - state.goods_cost
        if state.opp_last_offer is not None and state.opp_last_offer <= reservation and rng.random() < 0.3:
            return ACCEPT
        lo, hi = legal_offer_range(state)
        if lo > hi:
            return REJECT
        return offer(rng.uniform(lo, hi))
