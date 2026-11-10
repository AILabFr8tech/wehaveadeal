"""CarrierOPT: continuous, gradient-based optimization instead of a rule
checklist (R/RV) or particle search (IS).

Where MIA-RVelous-style methods (Florijn, Yolum & Baarslag 2024; and our own
CarrierIS above) RANK a small enumerated/sampled set of candidate offers by
expected utility, CarrierOPT treats the offer as a real-valued variable and
climbs the expected-utility surface directly with gradient ascent:

    EU(x) = P_accept(x) * value_bar(x, RP_t, target_t)
          + (1 - P_accept(x)) * carrier_value(continue at x)

`P_accept` is a differentiable sigmoid centered on a belief about the
shipper's current threshold. That belief is *not* stored on the instance
(every other engine in this package is stateless -- a single instance must
safely serve several concurrent tables -- see wehaveadeal/simulate.py's
lockstep loop): it is re-derived fresh each call from `state.history`, the
same turn log every engine already receives.

`_optimize_offer` climbs EU(x) on [rp_t, ceiling] via `wehaveadeal.optimize.
bisect_argmax` -- bisecting on the SIGN of a central-difference gradient at
the midpoint of the current search window, rather than stepping by the
gradient's magnitude (see that module for why). The "aggressive-then-fine"
behavior the brief asked for still shows turn to turn, but it comes from
the EU surface's own optimum actually shifting as belief/pressure evolve,
not from an external step-size decay.

Bisection is deterministic given the state, though, so a pure `CarrierOPT`
(unlike `CarrierOPT2`, whose whole point is a principled anti-detection
envelope) has no camouflage at all: the same market conditions always
produce the exact same counter, which is trivially fingerprintable by a
counterpart or an outside observer watching enough negotiations. `next_move`
therefore concedes a little further than the raw optimum by a small,
reproducibly-seeded jitter (`noise_sigma_frac`, applied only in the
concession direction so legality/monotonicity are never at risk) -- a
naive, fixed-magnitude nudge, deliberately simpler than OPT2's
EU-tolerance-scanned envelope, so the two engines stay distinct: OPT gets
"just enough randomness to not be a fingerprint," OPT2 gets the fully
worked-out performance-neutral version.
"""
from __future__ import annotations

import math
import random

from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.models import Action, Move, TableState
from wehaveadeal.optimize import bisect_argmax
from wehaveadeal.value_functions import carrier_value, portfolio_occupancy, value_bar


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class CarrierOPT(DecisionEngine):
    engine_name = "CarrierOPT"
    version = "v1"

    def __init__(
        self,
        agent_id: str,
        frozen: bool = True,
        firmness: float = 0.25,
        opening_offset_sigma_frac: float = 0.15,
        weight_occupancy: float = 0.5,
        gradient_steps: int = 24,
        sigmoid_steepness: float = 4.0,
        belief_cushion_sigma_frac: float = 0.5,
        noise_sigma_frac: float = 0.1,
    ) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.firmness = firmness
        self.opening_offset_sigma_frac = opening_offset_sigma_frac
        self.weight_occupancy = weight_occupancy
        self.gradient_steps = gradient_steps
        self.sigmoid_steepness = sigmoid_steepness
        self.belief_cushion_sigma_frac = belief_cushion_sigma_frac
        self.noise_sigma_frac = noise_sigma_frac

    def _band(self, state: TableState) -> tuple[float, float]:
        rp_t = state.refprice_mean - 0.5 * state.sigma
        target_t = state.refprice_mean + 0.5 * state.sigma
        return rp_t, target_t

    def estimate_shipper_threshold(self, state: TableState) -> float:
        """Belief about where the shipper is willing to go right now,
        re-derived fresh from `state.history` every call (no persistent
        instance state -- see module docstring). The shipper's most recent
        offer is the tightest available signal; we add a small cushion
        since by the time our counter lands, they'd plausibly have moved up
        another notch."""
        shipper_offers = [
            r.amount for r in state.history if r.actor == "shipper" and r.action == Action.OFFER and r.amount is not None
        ]
        if not shipper_offers:
            rp_t, _ = self._band(state)
            return rp_t  # no signal yet: neutral prior at our own floor
        return shipper_offers[-1] + self.belief_cushion_sigma_frac * state.sigma

    def _p_accept(self, offer: float, threshold: float, sigma: float) -> float:
        if sigma <= 0:
            return 1.0 if offer <= threshold else 0.0
        return _sigmoid(-self.sigmoid_steepness * (offer - threshold) / sigma)

    def _expected_utility(
        self, offer: float, threshold: float, state: TableState, rp_t: float, target_t: float, p_occupancy: float
    ) -> float:
        a = self._p_accept(offer, threshold, state.sigma)
        v_accept = value_bar(offer, rp_t, target_t)
        v_continue = carrier_value(
            rp_t=rp_t, target_t=target_t, terminal=False,
            shipper_offer_on_table=state.opp_last_offer, my_last_counter=offer,
            turn=state.turn, turns_max=state.turns_max,
            weight_occupancy=self.weight_occupancy, p_occupancy=p_occupancy,
        )
        return a * v_accept + (1 - a) * v_continue

    def _optimize_offer(
        self,
        state: TableState,
        rp_t: float,
        target_t: float,
        ceiling: float,
        threshold: float,
        p_occupancy: float,
    ) -> float:
        """Climb EU(x) on x in [rp_t, ceiling] via `bisect_argmax`. Broken
        out from `next_move` so CarrierOPT2 can reuse the exact same
        optimizer and only change the belief model / post-process the
        result.
        """
        return bisect_argmax(
            lambda x: self._expected_utility(x, threshold, state, rp_t, target_t, p_occupancy),
            rp_t,
            ceiling,
            self.gradient_steps,
        )

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
        rng = random.Random(f"{state.negotiation_seed}:opt:{state.table_id}:{state.turn}")
        x = x_star - rng.uniform(0, self.noise_sigma_frac * state.sigma)

        if shipper_offer is not None and shipper_offer >= rp_t:
            # Accepting now is literally "close at shipper_offer": compare
            # against the optimizer's own expected value of holding out for
            # x (a proposal, not a guaranteed close), not against x itself.
            v_accept_now = value_bar(shipper_offer, rp_t, target_t)
            v_continue_at_x = self._expected_utility(x, threshold, state, rp_t, target_t, p_occupancy)
            if v_accept_now >= v_continue_at_x:
                return ACCEPT

        next_counter = min(max(x, rp_t), target_t)
        if state.my_last_offer is not None:
            next_counter = min(next_counter, state.my_last_offer)
        return make_offer(next_counter)
