"""ShipperRV: ShipperR's rule checklist plus the shared value function
(time pressure + assignment adjustment) to decide, each turn, whether to
accept, counter, or hold firm. Mirrors CarrierRV exactly -- see that module
for the parts that aren't mirrored; here worst=RP (ceiling), best=target
(ambitious low price), and the extra value-function check fires for any
legal carrier offer (<=RP) that ShipperR's own convergence-gap rule wouldn't
already accept.

Reuses ShipperR's `_band` and opening/step logic rather than duplicating it;
the value function itself is `wehaveadeal.value_functions.shipper_value`, the
same shared implementation `carrier_value` uses (mirrored normalization,
one implementation).
"""
from __future__ import annotations

from wehaveadeal.agents.base import ACCEPT, REJECT, make_offer
from wehaveadeal.agents.shipper_r import ShipperR
from wehaveadeal.models import Move, TableState
from wehaveadeal.value_functions import portfolio_occupancy, shipper_route_is_feasible, shipper_value


class ShipperRV(ShipperR):
    engine_name = "ShipperRV"
    version = "v1"

    def __init__(
        self,
        agent_id: str,
        frozen: bool = True,
        weight_assignment: float = 0.5,
        firmness: float = 0.25,  # tactical ceiling for the assignment proxy: target_eff = target + firmness*(RP-target)
        **shipper_r_kwargs,
    ) -> None:
        super().__init__(agent_id, frozen=frozen, **shipper_r_kwargs)
        self.weight_assignment = weight_assignment
        self.firmness = firmness

    def _p_assignment(self, state: TableState) -> float:
        rp, target = self._band(state)
        target_eff = target + self.firmness * (rp - target)
        return portfolio_occupancy(state.my_portfolio, target_eff, higher_is_better=False)

    def next_move(self, state: TableState) -> Move:
        rp, target = self._band(state)
        sigma = state.sigma
        opening = target - self.offer_zone_sigma_frac * sigma

        if not shipper_route_is_feasible(state):
            return REJECT

        turn_index = state.my_turns_used + 1
        turns_per_side = state.turns_per_side
        p_assignment = self._p_assignment(state)

        if state.opp_last_offer is not None:
            counter = state.opp_last_offer
            my_reference = state.my_last_offer if state.my_last_offer is not None else opening
            estimated_carrier_floor = state.refprice_est - 0.5 * sigma
            convergence_gap = self.close_threshold_frac * abs(estimated_carrier_floor - rp)
            converged = abs(my_reference - counter) < convergence_gap

            if counter <= rp:
                if converged or turn_index >= turns_per_side:
                    return ACCEPT  # ShipperR's own baseline rule, unconditional

                v_accept_now = shipper_value(
                    rp=rp,
                    target=target,
                    terminal=True,
                    close_price=counter,
                    turn=state.turn,
                    turns_max=state.turns_max,
                    weight_assignment=self.weight_assignment,
                    p_assignment=p_assignment,
                )
                v_continue = shipper_value(
                    rp=rp,
                    target=target,
                    terminal=False,
                    carrier_offer_on_table=counter,
                    my_last_offer=state.my_last_offer,
                    turn=state.turn,
                    turns_max=state.turns_max,
                    weight_assignment=self.weight_assignment,
                    p_assignment=p_assignment,
                )
                if v_accept_now >= v_continue:
                    return ACCEPT

        if turn_index > turns_per_side:
            return REJECT  # out of turns without a deal: impasse

        if state.my_last_offer is None:
            next_offer = opening
        else:
            base_step = self.first_step_sigma_frac * sigma * (self.decay_rho ** (turn_index - 2))
            urgency = 1.0 + (1.0 - p_assignment) * 0.5  # thinner portfolio -> concede up to 50% faster
            next_offer = state.my_last_offer + base_step * urgency

        next_offer = min(next_offer, rp)  # never offer above RP
        if state.my_last_offer is not None:
            next_offer = max(next_offer, state.my_last_offer)  # monotonically non-decreasing

        return make_offer(next_offer)
