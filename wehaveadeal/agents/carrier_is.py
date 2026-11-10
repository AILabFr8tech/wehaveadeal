"""CarrierIS: ISMCTS-lite with closed-deck hidden-state inference
(Strategy spec §9.4 / SS10.3, Cowling et al. 2012).

Full ISMCTS (per Lehrach et al. 2025's defaults) builds a tree per
determinization, aggregates statistics at the information-set level, and
budgets ~1000 simulations per decision. A faithful, pure-Python version of
that at full budget is impractical to run over a 100-negotiation round
(12 tables x up to 6 carrier turns each). This engine keeps the three ideas
that actually matter for a single bilateral table and drops the tree:

  1. Determinization: sample `num_particles` plausible values of the
     shipper's hidden RP (closed-deck -- never inspects the real ShipperR
     instance, only the observed offer history: a rule-based shipper never
     offers above its own RP, so the last observed offer is a hard lower
     bound, and we sample a plausible band above it).
  2. A small set of candidate counter-offers is evaluated under EVERY
     particle; averaging accept/reject across particles for a candidate IS
     "aggregating statistics at the information-set level" -- just without
     the tree, since our game is a single bilateral exchange rather than a
     branching multi-agent board game.
  3. Leaves are scored with the shared value function (carrier_value)
     instead of long rollouts -- the modern MCTS-plus-value-network practice
     the source paper itself uses when a strong evaluator is available.

This is a deliberate simplification, not a claim of full ISMCTS fidelity;
`num_particles`/`num_candidates` are constructor parameters so a slower,
higher-fidelity run is just a config change away.
"""
from __future__ import annotations

import random
from typing import List

from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.models import Move, TableState
from wehaveadeal.value_functions import carrier_value, portfolio_occupancy


class CarrierIS(DecisionEngine):
    engine_name = "CarrierIS"
    version = "v1"

    def __init__(
        self,
        agent_id: str,
        frozen: bool = True,
        num_particles: int = 24,
        num_candidates: int = 7,
        firmness: float = 0.25,  # tactical-floor knob, reused for the occupancy proxy
        opening_offset_sigma_frac: float = 0.15,
        weight_occupancy: float = 0.5,
        particle_band_sigma_frac: float = 1.5,
    ) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.num_particles = num_particles
        self.num_candidates = num_candidates
        self.firmness = firmness
        self.opening_offset_sigma_frac = opening_offset_sigma_frac
        self.weight_occupancy = weight_occupancy
        self.particle_band_sigma_frac = particle_band_sigma_frac

    def _band(self, state: TableState) -> tuple[float, float]:
        rp_t = state.refprice_mean - 0.5 * state.sigma
        target_t = state.refprice_mean + 0.5 * state.sigma
        return rp_t, target_t

    def _sample_particles(self, state: TableState, rp_t: float) -> List[float]:
        """Determinizations of the shipper's hidden RP. Closed-deck: derived
        only from `state.opp_last_offer` (a real lower bound, since a
        rule-based shipper never offers above its own RP), never from the
        real ShipperR instance. Seeded from the negotiation seed + table +
        turn so a given decision is always reproducible."""
        rng = random.Random(f"{state.negotiation_seed}:is:{state.table_id}:{state.turn}")
        lower = state.opp_last_offer if state.opp_last_offer is not None else rp_t
        upper = lower + self.particle_band_sigma_frac * state.sigma
        return [rng.uniform(lower, upper) for _ in range(self.num_particles)]

    def next_move(self, state: TableState) -> Move:
        rp_t, target_t = self._band(state)
        tau = state.my_turns_used
        turns_per_side = state.turns_per_side
        turn_index = tau + 1
        shipper_offer = state.opp_last_offer

        if turn_index > turns_per_side:
            return REJECT

        if shipper_offer is not None and turn_index >= turns_per_side and shipper_offer >= rp_t:
            return ACCEPT  # hard last-turn floor, unconditional (WHAD spec §6.2)

        rp_eff = rp_t + self.firmness * (target_t - rp_t)
        p_occupancy = portfolio_occupancy(state.my_portfolio, rp_eff)

        ceiling = state.my_last_offer if state.my_last_offer is not None else target_t - self.opening_offset_sigma_frac * state.sigma
        ceiling = min(ceiling, target_t)

        if ceiling <= rp_t:
            candidates = [rp_t]
        else:
            candidates = [
                rp_t + (ceiling - rp_t) * frac
                for frac in (i / (self.num_candidates - 1) for i in range(self.num_candidates - 1, -1, -1))
            ]  # descending from `ceiling` to `rp_t`, inclusive -- every candidate is legal and monotonic-safe by construction

        if shipper_offer is not None:
            particles = self._sample_particles(state, rp_t)
        else:
            particles = []

        best_candidate, best_eu = candidates[0], float("-inf")
        for cand in candidates:
            if particles:
                accept_frac = sum(1 for p in particles if cand <= p) / len(particles)
            else:
                accept_frac = 0.0  # turn 1: nothing observed yet, no basis to expect acceptance
            v_accept = carrier_value(
                rp_t=rp_t, target_t=target_t, terminal=True, close_price=cand,
                turn=state.turn, turns_max=state.turns_max,
                weight_occupancy=self.weight_occupancy, p_occupancy=p_occupancy,
            )
            v_continue = carrier_value(
                rp_t=rp_t, target_t=target_t, terminal=False,
                shipper_offer_on_table=shipper_offer, my_last_counter=cand,
                turn=state.turn, turns_max=state.turns_max,
                weight_occupancy=self.weight_occupancy, p_occupancy=p_occupancy,
            )
            eu = accept_frac * v_accept + (1 - accept_frac) * v_continue
            if eu > best_eu:
                best_eu, best_candidate = eu, cand

        if shipper_offer is not None and shipper_offer >= rp_t:
            v_accept_now = carrier_value(
                rp_t=rp_t, target_t=target_t, terminal=True, close_price=shipper_offer,
                turn=state.turn, turns_max=state.turns_max,
                weight_occupancy=self.weight_occupancy, p_occupancy=p_occupancy,
            )
            if v_accept_now >= best_eu:
                return ACCEPT

        next_counter = min(max(best_candidate, rp_t), target_t)
        if state.my_last_offer is not None:
            next_counter = min(next_counter, state.my_last_offer)  # monotonically non-increasing
        return make_offer(next_counter)
