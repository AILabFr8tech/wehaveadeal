"""ShipperOPT: continuous, gradient-based optimization mirrored onto the
client role. See `wehaveadeal.agents.carrier_opt` for the full rationale
(continuous offer instead of ranked candidates, sign-based bisection instead
of a magnitude-based gradient step) -- everything there applies here
unchanged except which direction is "better" and which side's threshold is
being estimated.

Mirroring notes: worst=RP (ceiling), best=target (ambitious low price); the
belief is about the CARRIER's floor instead of the shipper's threshold, so
`P_accept` must rise (not fall) with the offer -- a carrier is more likely
to accept a HIGHER price, the opposite of a shipper facing a carrier's ask.
The search interval is `[floor, RP]` with `floor` (the shipper's own last
offer, or the opening anchor) rising each turn -- the mirror image of
CarrierOPT's `[RP_t, ceiling]` with a falling ceiling.

Also mirrored: the small, reproducibly-seeded concession-direction jitter
(`noise_sigma_frac`) CarrierOPT applies so a pure gradient optimum -- fully
deterministic given the state -- isn't a trivial fingerprint. Here it nudges
the offer UP (toward RP), never down, so legality/monotonicity are never at
risk. See `carrier_opt.py`'s module docstring for why this stays a naive,
fixed-magnitude nudge rather than OPT2's EU-tolerance-scanned envelope.
"""
from __future__ import annotations

import math
import random

from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.market import SHIPPER_NOISE_FRACTION
from wehaveadeal.models import Action, Move, TableState
from wehaveadeal.optimize import bisect_argmax
from wehaveadeal.value_functions import portfolio_occupancy, shipper_route_is_feasible, shipper_value, value_bar


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class ShipperOPT(DecisionEngine):
    engine_name = "ShipperOPT"
    version = "v1"

    def __init__(
        self,
        agent_id: str,
        frozen: bool = True,
        firmness: float = 0.25,
        offer_zone_sigma_frac: float = 0.5,
        target_cushion_k: float = 0.5,
        weight_assignment: float = 0.5,
        gradient_steps: int = 24,
        sigmoid_steepness: float = 4.0,
        belief_cushion_sigma_frac: float = 0.5,
        noise_sigma_frac: float = 0.1,
    ) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.firmness = firmness
        self.offer_zone_sigma_frac = offer_zone_sigma_frac
        self.target_cushion_k = target_cushion_k
        self.weight_assignment = weight_assignment
        self.gradient_steps = gradient_steps
        self.sigmoid_steepness = sigmoid_steepness
        self.belief_cushion_sigma_frac = belief_cushion_sigma_frac
        self.noise_sigma_frac = noise_sigma_frac

    def _band(self, state: TableState) -> tuple[float, float]:
        rp = state.sale_price - state.goods_cost
        sigma_eps = state.sigma * SHIPPER_NOISE_FRACTION
        target = state.refprice_est - 0.5 * state.sigma + self.target_cushion_k * sigma_eps
        return rp, target

    def estimate_carrier_floor(self, state: TableState) -> float:
        """Belief about the carrier's floor, re-derived fresh from
        `state.history` every call (no persistent instance state -- see
        `carrier_opt.py`'s module docstring). The carrier's most recent
        counter is the tightest available signal; we subtract a small
        cushion since by the time our offer lands, they may have plausibly
        moved down another notch. The shipper moves first in every
        negotiation, so the "no signal yet" fallback below is live on every
        opening move, unlike its carrier-side mirror."""
        carrier_offers = [
            r.amount for r in state.history if r.actor == "carrier" and r.action == Action.OFFER and r.amount is not None
        ]
        if not carrier_offers:
            rp, _ = self._band(state)
            return rp  # no signal yet: neutral prior at our own ceiling
        return carrier_offers[-1] - self.belief_cushion_sigma_frac * state.sigma

    def _p_accept(self, offer: float, threshold: float, sigma: float) -> float:
        if sigma <= 0:
            return 1.0 if offer >= threshold else 0.0
        return _sigmoid(self.sigmoid_steepness * (offer - threshold) / sigma)

    def _expected_utility(
        self, offer: float, threshold: float, state: TableState, rp: float, target: float, p_assignment: float
    ) -> float:
        a = self._p_accept(offer, threshold, state.sigma)
        v_accept = value_bar(offer, rp, target)
        v_continue = shipper_value(
            rp=rp, target=target, terminal=False,
            carrier_offer_on_table=state.opp_last_offer, my_last_offer=offer,
            turn=state.turn, turns_max=state.turns_max,
            weight_assignment=self.weight_assignment, p_assignment=p_assignment,
        )
        return a * v_accept + (1 - a) * v_continue

    def _optimize_offer(
        self,
        state: TableState,
        rp: float,
        target: float,
        floor: float,
        threshold: float,
        p_assignment: float,
    ) -> float:
        """Climb EU(x) on x in [floor, RP] via `bisect_argmax`. Broken out
        from `next_move` so ShipperOPT2 can reuse the exact same optimizer
        and only change the belief model / post-process the result.
        """
        return bisect_argmax(
            lambda x: self._expected_utility(x, threshold, state, rp, target, p_assignment),
            floor,
            rp,
            self.gradient_steps,
        )

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
        rng = random.Random(f"{state.negotiation_seed}:opt:{state.table_id}:{state.turn}")
        x = x_star + rng.uniform(0, self.noise_sigma_frac * state.sigma)

        if carrier_offer is not None and carrier_offer <= rp:
            # Accepting now is literally "close at carrier_offer": compare
            # against the optimizer's own expected value of holding out for
            # x (a proposal, not a guaranteed close), not against x itself.
            v_accept_now = value_bar(carrier_offer, rp, target)
            v_continue_at_x = self._expected_utility(x, threshold, state, rp, target, p_assignment)
            if v_accept_now >= v_continue_at_x:
                return ACCEPT

        next_offer = min(x, rp)
        if state.my_last_offer is not None:
            next_offer = max(next_offer, state.my_last_offer)
        return make_offer(next_offer)
