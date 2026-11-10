"""NoisyMeanCarrier -- the simplest sensible rule: "market mean +/- 10%".

Each turn it quotes a price around the market reference price:

    price = refprice_mean * (1 + u),   u ~ Uniform(-10%, +10%)

It accepts the shipper's offer if that offer is at least the price it would
quote this turn. The random number is seeded from the table and the turn,
so a benchmark re-run gives identical results.

A useful floor to beat: it closes many deals but, accepting anything above a
random draw, often sells below the market mean.
"""
import numpy as np

from wehaveadeal import ACCEPT, REJECT, DecisionEngine, Move, TableState, legal_offer_range, offer


class NoisyMeanCarrier(DecisionEngine):
    role = "carrier"
    engine_name = "NoisyMeanCarrier"
    version = "v1"

    def __init__(self, agent_id: str, frozen: bool = True, spread: float = 0.10) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.spread = spread  # 0.10 = +/- 10% around the market mean

    def next_move(self, state: TableState) -> Move:
        mean = state.refprice_mean
        rng = np.random.default_rng([state.negotiation_seed, state.turn])
        price = mean * (1 + rng.uniform(-self.spread, self.spread))

        # Accept if the shipper already offers at least our price.
        if state.opp_last_offer is not None and state.opp_last_offer >= price:
            return ACCEPT

        # Otherwise quote our price, kept inside what the rules allow right now.
        lo, hi = legal_offer_range(state)
        if lo > hi:
            return REJECT
        return offer(min(max(price, lo), hi))
