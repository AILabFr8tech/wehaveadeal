"""ShipperOPT2: ShipperOPT plus (a) a belief fitted by explicit gradient
descent instead of a single-point heuristic, and (b) a performance-neutral
randomization envelope for anti-detection -- mirrored onto the client role.
See `wehaveadeal.agents.carrier_opt2` for the full rationale; everything there
applies here unchanged except the concession direction, which is UPWARD for
a shipper (widen the anti-detection envelope above x*, not below it).
"""
from __future__ import annotations

import random

from wehaveadeal.agents.base import ACCEPT, REJECT, make_offer
from wehaveadeal.agents.shipper_opt import ShipperOPT
from wehaveadeal.models import Action, Move, TableState
from wehaveadeal.value_functions import portfolio_occupancy, shipper_route_is_feasible, value_bar


class ShipperOPT2(ShipperOPT):
    engine_name = "ShipperOPT2"
    version = "v1"

    def __init__(
        self,
        agent_id: str,
        frozen: bool = True,
        belief_fit_steps: int = 60,
        belief_fit_lr: float = 0.15,
        eu_tolerance: float = 0.05,
        envelope_step_sigma_frac: float = 0.01,
        **shipper_opt_kwargs,
    ) -> None:
        super().__init__(agent_id, frozen=frozen, **shipper_opt_kwargs)
        self.belief_fit_steps = belief_fit_steps
        self.belief_fit_lr = belief_fit_lr
        self.eu_tolerance = eu_tolerance
        self.envelope_step_sigma_frac = envelope_step_sigma_frac

    def estimate_carrier_floor(self, state: TableState) -> float:
        carrier_offers = [
            r.amount for r in state.history if r.actor == "carrier" and r.action == Action.OFFER and r.amount is not None
        ]
        if len(carrier_offers) < 2:
            return super().estimate_carrier_floor(state)

        # Fit offer[k] ~ intercept + slope*k via explicit gradient descent
        # (see carrier_opt2.py's module docstring for why not closed-form OLS).
        xs = list(range(len(carrier_offers)))
        ys = carrier_offers
        n = len(ys)
        intercept, slope = ys[0], 0.0
        for _ in range(self.belief_fit_steps):
            preds = [intercept + slope * xi for xi in xs]
            errors = [p - y for p, y in zip(preds, ys)]
            grad_intercept = sum(errors) / n
            grad_slope = sum(e * xi for e, xi in zip(errors, xs)) / n
            intercept -= self.belief_fit_lr * grad_intercept
            slope -= self.belief_fit_lr * grad_slope

        projected_next_offer = intercept + slope * n  # one step beyond the last observed offer
        return projected_next_offer - self.belief_cushion_sigma_frac * state.sigma

    def _anti_detection_envelope(
        self,
        x_star: float,
        rp: float,
        target: float,
        threshold: float,
        state: TableState,
        p_assignment: float,
    ) -> float:
        eu_star = self._expected_utility(x_star, threshold, state, rp, target, p_assignment)
        step = max(state.sigma * self.envelope_step_sigma_frac, 1e-6)
        hi = x_star
        tolerance_abs = self.eu_tolerance * max(abs(eu_star), 1e-6)
        for _ in range(200):
            candidate = hi + step
            if candidate > rp:
                break
            eu_candidate = self._expected_utility(candidate, threshold, state, rp, target, p_assignment)
            if eu_star - eu_candidate > tolerance_abs:
                break
            hi = candidate
        rng = random.Random(f"{state.negotiation_seed}:opt2:{state.table_id}:{state.turn}")
        return rng.uniform(x_star, hi)

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
            return ACCEPT

        target_eff = target + self.firmness * (rp - target)
        p_assignment = portfolio_occupancy(state.my_portfolio, target_eff, higher_is_better=False)
        threshold = self.estimate_carrier_floor(state)

        floor = state.my_last_offer if state.my_last_offer is not None else target - self.offer_zone_sigma_frac * state.sigma

        x_star = self._optimize_offer(state, rp, target, floor, threshold, p_assignment)
        x = self._anti_detection_envelope(x_star, rp, target, threshold, state, p_assignment)

        if carrier_offer is not None and carrier_offer <= rp:
            v_accept_now = value_bar(carrier_offer, rp, target)
            v_continue_at_x = self._expected_utility(x, threshold, state, rp, target, p_assignment)
            if v_accept_now >= v_continue_at_x:
                return ACCEPT

        next_offer = min(x, rp)
        if state.my_last_offer is not None:
            next_offer = max(next_offer, state.my_last_offer)
        return make_offer(next_offer)
