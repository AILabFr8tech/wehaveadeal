"""CarrierOPT2: CarrierOPT plus (a) a belief fitted by explicit gradient
descent instead of a single-point heuristic, and (b) a performance-neutral
randomization envelope for anti-detection.

This is the "publication" variant discussed alongside CarrierOPT: two
genuine methodological deltas from both CarrierOPT and from MIA-RVelous-
style ranking methods, kept in the same "everything is optimized by
gradient" story as CarrierOPT rather than bolting on unrelated machinery.

(a) Belief fitting. CarrierOPT reads a single point (the shipper's last
    offer + a fixed cushion). Here we fit a line
    (offer ~ intercept + slope*turn_number) to the *entire* observed offer
    sequence for this table via a handful of explicit gradient-descent
    steps on squared error, then extrapolate one step ahead. Mathematically
    this converges to the same point ordinary least squares would give --
    the reason to do it via gradient steps rather than the closed-form
    formula is that the belief and the offer are now co-optimized by the
    *same* mechanism, which is the actual point of the exercise.

(b) Anti-detection envelope. Strategy spec §8.3 asks for randomization inside
    "performance-neutral envelopes": ranges where expected payoff does not
    change with statistical significance. Rather than approximate that with
    an ad hoc random jitter, we compute it directly: starting from the
    gradient optimum x*, widen a window downward (concession direction
    only, so legality/monotonicity are never at risk) for as long as EU
    stays within `eu_tolerance` of EU(x*), then sample the emitted offer
    uniformly inside that window with a seed derived from the negotiation
    seed (reproducible, not just "add noise").
"""
from __future__ import annotations

import random

from wehaveadeal.agents.base import ACCEPT, REJECT, make_offer
from wehaveadeal.agents.carrier_opt import CarrierOPT
from wehaveadeal.models import Action, Move, TableState
from wehaveadeal.value_functions import portfolio_occupancy, value_bar


class CarrierOPT2(CarrierOPT):
    engine_name = "CarrierOPT2"
    version = "v1"

    def __init__(
        self,
        agent_id: str,
        frozen: bool = True,
        belief_fit_steps: int = 60,
        belief_fit_lr: float = 0.15,
        eu_tolerance: float = 0.05,
        envelope_step_sigma_frac: float = 0.01,
        **carrier_opt_kwargs,
    ) -> None:
        super().__init__(agent_id, frozen=frozen, **carrier_opt_kwargs)
        self.belief_fit_steps = belief_fit_steps
        self.belief_fit_lr = belief_fit_lr
        self.eu_tolerance = eu_tolerance
        self.envelope_step_sigma_frac = envelope_step_sigma_frac

    def estimate_shipper_threshold(self, state: TableState) -> float:
        shipper_offers = [
            r.amount for r in state.history if r.actor == "shipper" and r.action == Action.OFFER and r.amount is not None
        ]
        if len(shipper_offers) < 2:
            return super().estimate_shipper_threshold(state)

        # Fit offer[k] ~ intercept + slope*k via explicit gradient descent
        # (see module docstring for why not closed-form OLS).
        xs = list(range(len(shipper_offers)))
        ys = shipper_offers
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
        return projected_next_offer + self.belief_cushion_sigma_frac * state.sigma

    def _anti_detection_envelope(
        self,
        x_star: float,
        rp_t: float,
        target_t: float,
        threshold: float,
        state: TableState,
        p_occupancy: float,
    ) -> float:
        eu_star = self._expected_utility(x_star, threshold, state, rp_t, target_t, p_occupancy)
        step = max(state.sigma * self.envelope_step_sigma_frac, 1e-6)
        lo = x_star
        tolerance_abs = self.eu_tolerance * max(abs(eu_star), 1e-6)
        for _ in range(200):
            candidate = lo - step
            if candidate < rp_t:
                break
            eu_candidate = self._expected_utility(candidate, threshold, state, rp_t, target_t, p_occupancy)
            if eu_star - eu_candidate > tolerance_abs:
                break
            lo = candidate
        rng = random.Random(f"{state.negotiation_seed}:opt2:{state.table_id}:{state.turn}")
        return rng.uniform(lo, x_star)

    def next_move(self, state: TableState) -> Move:
        rp_t, target_t = self._band(state)
        tau = state.my_turns_used
        turns_per_side = state.turns_per_side
        turn_index = tau + 1
        shipper_offer = state.opp_last_offer

        if turn_index > turns_per_side:
            return REJECT

        if shipper_offer is not None and turn_index >= turns_per_side and shipper_offer >= rp_t:
            return ACCEPT

        rp_eff = rp_t + self.firmness * (target_t - rp_t)
        p_occupancy = portfolio_occupancy(state.my_portfolio, rp_eff)
        threshold = self.estimate_shipper_threshold(state)

        ceiling = state.my_last_offer if state.my_last_offer is not None else target_t - self.opening_offset_sigma_frac * state.sigma
        ceiling = min(ceiling, target_t)

        x_star = self._optimize_offer(state, rp_t, target_t, ceiling, threshold, p_occupancy)
        x = self._anti_detection_envelope(x_star, rp_t, target_t, threshold, state, p_occupancy)

        if shipper_offer is not None and shipper_offer >= rp_t:
            v_accept_now = value_bar(shipper_offer, rp_t, target_t)
            v_continue_at_x = self._expected_utility(x, threshold, state, rp_t, target_t, p_occupancy)
            if v_accept_now >= v_continue_at_x:
                return ACCEPT

        next_counter = min(max(x, rp_t), target_t)
        if state.my_last_offer is not None:
            next_counter = min(next_counter, state.my_last_offer)
        return make_offer(next_counter)
