"""CarrierRV: CarrierR's rule checklist plus the shared value function
(time pressure + occupancy adjustment) to decide, each turn, whether to
accept, counter, or hold firm (Strategy spec §7.2 / SS10.2).

Reuses CarrierR's band/threshold/anchor logic rather than duplicating it;
the value function itself is imported from wehaveadeal.value_functions, the same
module a future ShipperRV would reuse (mirrored normalization, one
implementation).
"""
from __future__ import annotations

from wehaveadeal.agents.base import ACCEPT, REJECT, make_offer
from wehaveadeal.agents.carrier_r import CarrierR
from wehaveadeal.models import Move, TableState
from wehaveadeal.value_functions import carrier_value, portfolio_occupancy


class CarrierRV(CarrierR):
    engine_name = "CarrierRV"
    version = "v1"

    def __init__(
        self,
        agent_id: str,
        frozen: bool = True,
        weight_occupancy: float = 0.5,
        **carrier_r_kwargs,
    ) -> None:
        super().__init__(agent_id, frozen=frozen, **carrier_r_kwargs)
        self.weight_occupancy = weight_occupancy

    def _p_occupancy(self, state: TableState) -> float:
        rp_t, target_t = self._band(state)
        rp_eff = rp_t + self.firmness * (target_t - rp_t)
        return portfolio_occupancy(state.my_portfolio, rp_eff)

    def next_move(self, state: TableState) -> Move:
        rp_t, target_t = self._band(state)
        tau = state.my_turns_used
        turns_per_side = state.turns_per_side
        turn_index = tau + 1
        shipper_offer = state.opp_last_offer

        if turn_index > turns_per_side:
            return REJECT

        p_occupancy = self._p_occupancy(state)

        if shipper_offer is not None:
            if turn_index >= turns_per_side and shipper_offer >= rp_t:
                return ACCEPT  # hard last-turn floor, unconditional

            threshold = self._threshold(tau, turns_per_side, rp_t, target_t)
            if shipper_offer >= threshold:
                return ACCEPT

            if shipper_offer >= rp_t:
                v_accept_now = carrier_value(
                    rp_t=rp_t,
                    target_t=target_t,
                    terminal=True,
                    close_price=shipper_offer,
                    turn=state.turn,
                    turns_max=state.turns_max,
                    weight_occupancy=self.weight_occupancy,
                    p_occupancy=p_occupancy,
                )
                v_continue = carrier_value(
                    rp_t=rp_t,
                    target_t=target_t,
                    terminal=False,
                    shipper_offer_on_table=shipper_offer,
                    my_last_counter=state.my_last_offer,
                    turn=state.turn,
                    turns_max=state.turns_max,
                    weight_occupancy=self.weight_occupancy,
                    p_occupancy=p_occupancy,
                )
                if v_accept_now >= v_continue:
                    return ACCEPT

        if state.my_last_offer is None:
            next_counter = target_t - self.opening_offset_sigma_frac * state.sigma
        else:
            base_step = self.first_step_sigma_frac * state.sigma * (self.decay_rho ** (turn_index - 2))
            urgency = 1.0 + (1.0 - p_occupancy) * 0.5  # thinner portfolio -> concede up to 50% faster
            next_counter = state.my_last_offer - base_step * urgency

        next_counter = max(next_counter, rp_t)
        if state.my_last_offer is not None:
            next_counter = min(next_counter, state.my_last_offer)
        next_counter = min(next_counter, target_t)

        return make_offer(next_counter)
