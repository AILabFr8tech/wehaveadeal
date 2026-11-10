"""CarrierR: pure-rules carrier (Strategy spec §15.2 / SS3.4, "R" architecture).

Fixed anchor inside the hard +-0.5*sigma band, decreasing counter-offer
steps, and the normative acceptance-threshold rule from Strategy spec §3.4.
No search, no learning, no value function. Defaults below correspond to the
"medium" carrier style (Strategy spec §11.2) but are all constructor
parameters -- not hardcoded -- since style is an explicit calibration axis
even though Phase 1 ships only one carrier calibration.
"""
from __future__ import annotations

from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.models import Move, TableState


class CarrierR(DecisionEngine):
    engine_name = "CarrierR"
    version = "v1"

    def __init__(
        self,
        agent_id: str,
        frozen: bool = True,
        opening_offset_sigma_frac: float = 0.15,  # first counter = target_t - this*sigma
        firmness: float = 0.25,  # tactical floor: RP_eff = RP_t + firmness*(target_t-RP_t)
        gamma: float = 2.0,  # threshold(tau) curvature, gamma > 1
        first_step_sigma_frac: float = 0.5,
        decay_rho: float = 0.5,
    ) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.opening_offset_sigma_frac = opening_offset_sigma_frac
        self.firmness = firmness
        self.gamma = gamma
        self.first_step_sigma_frac = first_step_sigma_frac
        self.decay_rho = decay_rho

    def _band(self, state: TableState) -> tuple[float, float]:
        rp_t = state.refprice_mean - 0.5 * state.sigma
        target_t = state.refprice_mean + 0.5 * state.sigma
        return rp_t, target_t

    def _threshold(self, tau: int, turns_per_side: int, rp_t: float, target_t: float) -> float:
        rp_eff = rp_t + self.firmness * (target_t - rp_t)
        return rp_eff + (target_t - rp_eff) * (1 - tau / turns_per_side) ** self.gamma

    def next_move(self, state: TableState) -> Move:
        rp_t, target_t = self._band(state)
        opening = target_t - self.opening_offset_sigma_frac * state.sigma

        tau = state.my_turns_used  # turns already consumed by me, before this decision
        turns_per_side = state.turns_per_side
        turn_index = tau + 1

        shipper_offer = state.opp_last_offer
        if shipper_offer is not None:
            threshold = self._threshold(tau, turns_per_side, rp_t, target_t)
            if shipper_offer >= threshold:
                return ACCEPT
            if turn_index >= turns_per_side and shipper_offer >= rp_t:
                return ACCEPT

        if turn_index > turns_per_side:
            return REJECT  # out of turns without a deal: impasse

        if state.my_last_offer is None:
            next_counter = opening
        else:
            step = self.first_step_sigma_frac * state.sigma * (self.decay_rho ** (turn_index - 2))
            next_counter = state.my_last_offer - step

        next_counter = max(next_counter, rp_t)  # never below RP_t
        if state.my_last_offer is not None:
            next_counter = min(next_counter, state.my_last_offer)  # monotonically non-increasing
        next_counter = min(next_counter, target_t)  # never above target_t

        return make_offer(next_counter)
