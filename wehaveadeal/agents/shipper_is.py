"""ShipperIS: ISMCTS-lite with closed-deck hidden-state inference, mirrored
onto the client role. See `wehaveadeal.agents.carrier_is` for the full rationale
(determinization + information-set aggregation + value-function leaves
instead of a tree) -- everything there applies here unchanged except which
side is hidden and which direction is "better".

Mirroring notes (see also `wehaveadeal.value_functions.value_bar`'s docstring):
worst=RP (ceiling), best=target (ambitious low price); the hidden quantity
is the CARRIER's floor instead of the shipper's RP; and since the shipper
moves first in every negotiation (global turn 1), the "nothing observed
yet" branch that is effectively dead code in `CarrierIS` is the live,
every-negotiation opening move here.
"""
from __future__ import annotations

import random
from typing import List

from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.market import SHIPPER_NOISE_FRACTION
from wehaveadeal.models import Move, TableState
from wehaveadeal.value_functions import portfolio_occupancy, shipper_route_is_feasible, shipper_value


class ShipperIS(DecisionEngine):
    engine_name = "ShipperIS"
    version = "v1"

    def __init__(
        self,
        agent_id: str,
        frozen: bool = True,
        num_particles: int = 24,
        num_candidates: int = 7,
        firmness: float = 0.25,  # tactical-ceiling knob, reused for the assignment proxy
        offer_zone_sigma_frac: float = 0.5,
        target_cushion_k: float = 0.5,
        weight_assignment: float = 0.5,
        particle_band_sigma_frac: float = 1.5,
    ) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.num_particles = num_particles
        self.num_candidates = num_candidates
        self.firmness = firmness
        self.offer_zone_sigma_frac = offer_zone_sigma_frac
        self.target_cushion_k = target_cushion_k
        self.weight_assignment = weight_assignment
        self.particle_band_sigma_frac = particle_band_sigma_frac

    def _band(self, state: TableState) -> tuple[float, float]:
        rp = state.sale_price - state.goods_cost
        sigma_eps = state.sigma * SHIPPER_NOISE_FRACTION
        target = state.refprice_est - 0.5 * state.sigma + self.target_cushion_k * sigma_eps
        return rp, target

    def _sample_particles(self, state: TableState, rp: float) -> List[float]:
        """Determinizations of the carrier's hidden floor. Closed-deck:
        derived only from `state.opp_last_offer` (a real upper bound, since
        a rule-based carrier never counters below its own floor), never
        from the real carrier instance. Seeded from the negotiation seed +
        table + turn so a given decision is always reproducible."""
        rng = random.Random(f"{state.negotiation_seed}:is:{state.table_id}:{state.turn}")
        upper = state.opp_last_offer if state.opp_last_offer is not None else rp
        lower = upper - self.particle_band_sigma_frac * state.sigma
        return [rng.uniform(lower, upper) for _ in range(self.num_particles)]

    def next_move(self, state: TableState) -> Move:
        rp, target = self._band(state)
        tau = state.my_turns_used
        turns_per_side = state.turns_per_side
        turn_index = tau + 1
        carrier_offer = state.opp_last_offer

        if not shipper_route_is_feasible(state):
            return REJECT

        if turn_index > turns_per_side:
            return REJECT

        if carrier_offer is not None and turn_index >= turns_per_side and carrier_offer <= rp:
            return ACCEPT  # hard last-turn floor, unconditional

        target_eff = target + self.firmness * (rp - target)
        p_assignment = portfolio_occupancy(state.my_portfolio, target_eff, higher_is_better=False)

        floor = state.my_last_offer if state.my_last_offer is not None else target - self.offer_zone_sigma_frac * state.sigma

        if floor >= rp:
            candidates = [rp]
        else:
            candidates = [
                floor + (rp - floor) * (i / (self.num_candidates - 1)) for i in range(self.num_candidates)
            ]  # ascending from `floor` to `rp`, inclusive -- every candidate is legal and monotonic-safe by construction

        if carrier_offer is not None:
            particles = self._sample_particles(state, rp)
        else:
            particles = []

        best_candidate, best_eu = candidates[0], float("-inf")
        for cand in candidates:
            if particles:
                accept_frac = sum(1 for p in particles if cand >= p) / len(particles)
            else:
                accept_frac = 0.0  # opening move: nothing observed yet, no basis to expect acceptance
            v_accept = shipper_value(
                rp=rp, target=target, terminal=True, close_price=cand,
                turn=state.turn, turns_max=state.turns_max,
                weight_assignment=self.weight_assignment, p_assignment=p_assignment,
            )
            v_continue = shipper_value(
                rp=rp, target=target, terminal=False,
                carrier_offer_on_table=carrier_offer, my_last_offer=cand,
                turn=state.turn, turns_max=state.turns_max,
                weight_assignment=self.weight_assignment, p_assignment=p_assignment,
            )
            eu = accept_frac * v_accept + (1 - accept_frac) * v_continue
            if eu > best_eu:
                best_eu, best_candidate = eu, cand

        if carrier_offer is not None and carrier_offer <= rp:
            v_accept_now = shipper_value(
                rp=rp, target=target, terminal=True, close_price=carrier_offer,
                turn=state.turn, turns_max=state.turns_max,
                weight_assignment=self.weight_assignment, p_assignment=p_assignment,
            )
            if v_accept_now >= best_eu:
                return ACCEPT

        next_offer = min(best_candidate, rp)
        if state.my_last_offer is not None:
            next_offer = max(next_offer, state.my_last_offer)  # monotonically non-decreasing
        return make_offer(next_offer)
