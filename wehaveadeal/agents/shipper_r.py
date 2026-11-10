"""ShipperR: pure-rules shipper (Strategy spec §15.1, "R" architecture).

Fixed anchor, decreasing concession steps, threshold close. No search, no
learning, no value function -- this is the Phase-1 frozen sparring partner
used to exercise and compare the two carrier engines, not the final shipper
research track (see project brief).

All calibration numbers are constructor parameters, never inline magic
numbers (Strategy spec §11.3: "the closing threshold is a PARAMETER, not a constant").
"""
from __future__ import annotations

from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.market import SHIPPER_NOISE_FRACTION
from wehaveadeal.models import Move, TableState
from wehaveadeal.value_functions import shipper_route_is_feasible


class ShipperR(DecisionEngine):
    engine_name = "ShipperR"
    version = "v1"

    def __init__(
        self,
        agent_id: str,
        frozen: bool = True,
        target_cushion_k: float = 0.5,
        offer_zone_sigma_frac: float = 0.5,
        first_step_sigma_frac: float = 0.5,
        decay_rho: float = 0.5,
        close_threshold_frac: float = 0.10,
    ) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.target_cushion_k = target_cushion_k
        self.offer_zone_sigma_frac = offer_zone_sigma_frac
        self.first_step_sigma_frac = first_step_sigma_frac
        self.decay_rho = decay_rho
        self.close_threshold_frac = close_threshold_frac

    def _band(self, state: TableState) -> tuple[float, float]:
        rp = state.sale_price - state.goods_cost
        sigma = state.sigma
        # target = RefPrice_est - 0.5*sigma + k*sigma_eps (Strategy spec §2.2)
        sigma_eps = sigma * SHIPPER_NOISE_FRACTION
        target = state.refprice_est - 0.5 * sigma + self.target_cushion_k * sigma_eps
        return rp, target

    def next_move(self, state: TableState) -> Move:
        rp, target = self._band(state)
        sigma = state.sigma
        opening = target - self.offer_zone_sigma_frac * sigma

        # Deliberate walk-away (Strategy spec §13.0): if even the best-case
        # estimated carrier floor already exceeds my RP, no legal deal exists.
        if not shipper_route_is_feasible(state):
            return REJECT

        turn_index = state.my_turns_used + 1  # my n-th move, 1-indexed
        turns_per_side = state.turns_per_side

        if state.opp_last_offer is not None:
            counter = state.opp_last_offer
            my_reference = state.my_last_offer if state.my_last_offer is not None else opening
            estimated_carrier_floor = state.refprice_est - 0.5 * sigma
            convergence_gap = self.close_threshold_frac * abs(estimated_carrier_floor - rp)
            converged = abs(my_reference - counter) < convergence_gap
            if counter <= rp and (converged or turn_index >= turns_per_side):
                return ACCEPT

        if turn_index > turns_per_side:
            return REJECT  # out of turns without a deal: impasse

        if state.my_last_offer is None:
            next_offer = opening
        else:
            step = self.first_step_sigma_frac * sigma * (self.decay_rho ** (turn_index - 2))
            next_offer = state.my_last_offer + step

        next_offer = min(next_offer, rp)  # never offer above RP
        if state.my_last_offer is not None:
            next_offer = max(next_offer, state.my_last_offer)  # monotonically non-decreasing

        return make_offer(next_offer)
