"""NegotiationEnv: one bilateral negotiation table as a Gymnasium
environment, from ONE side's perspective, against a frozen hand-designed
opponent sampled fresh each episode from {R, RV, IS, OPT, OPT2} -- the same
"face a genuine variety of counterparts, not just one" idea
`wehaveadeal/ml/collect.py` uses for imitation learning, except here it drives a
real reward loop instead of a fixed dataset.

Deliberately reuses `wehaveadeal.simulate`'s already-tested single-table
mechanics (`_TableRunner` for turn bookkeeping, `_validate_move` for
legality) instead of re-deriving the turn loop here -- that keeps the RL
environment's notion of "legal move" identical to every other engine's,
checked by the exact same code, not a re-implementation that could quietly
drift out of sync.

Observation: the SAME role-agnostic feature vector `wehaveadeal/ml/features.py`
already defines for the imitation-learning engines (band positions, turn
fractions, opponent trend, portfolio promise) -- reused unchanged, so RL
and imitation learning share one definition of "what a negotiator can see".
`my_portfolio` is always empty here (portfolio_promise stays at its neutral
0.5): this environment trains on isolated bilateral tables, not full
12-table rounds, so there are no real sibling tables to report on. That's a
deliberate scope simplification for this first pass, not an oversight --
see the module docstring in `wehaveadeal/rl/train.py`.

Action: a single continuous number in [0, 1] -- the policy's own
"aspiration" position on this role's [worst, best] price band THIS turn.
There is no separate accept/reject action: if the counterpart's current
offer already meets or beats the aspiration, the environment accepts it on
the policy's behalf; otherwise it counters AT the aspiration (clamped for
legality/monotonicity, like every hand-designed engine). This is what lets
the policy actually learn WHEN to close -- the one decision
`CarrierML`/`ShipperML` deliberately left hardcoded (see
`wehaveadeal/agents/carrier_ml.py`'s module docstring).

Reward: 0 on every non-terminal step; at termination, `value_bar` of the
closing price (0 if impasse) -- the same [0, 1] normalization used
everywhere else in this package, not raw dollars, so the reward scale is
comparable across routes of very different size.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover - exercised only when the RL extra isn't installed
    raise ImportError("wehaveadeal.rl needs gymnasium/stable-baselines3 -- run `pip install -r requirements.txt`.") from exc

from wehaveadeal.agents.base import DecisionEngine
from wehaveadeal.market import MarketState
from wehaveadeal.ml.features import FEATURE_NAMES, extract_features, features_to_vector, role_band
from wehaveadeal.models import Action, Move, TableState
from wehaveadeal.simulate import CARRIER_KINDS, SHIPPER_KINDS, TURNS_MAX, _TableRunner, _validate_move, build_carrier, build_shipper
from wehaveadeal.value_functions import shipper_route_is_feasible, value_bar

# "ml"/"ml2"/"rl"/"rl2" excluded from the opponent pool: at training time
# there's no guarantee a trained ml/ml2/rl/rl2 model exists yet for the
# OTHER role, and even once one does, the opponent roster should reflect
# the established hand-designed architectures, not create a circular
# dependency on a model still being trained (same reasoning as
# wehaveadeal/ml/collect.py). `wehaveadeal/rl/league_env.py` reuses this constant.
LEARNED_KINDS = ("ml", "ml2", "rl", "rl2")
OPPONENT_KINDS = {
    "carrier": tuple(k for k in SHIPPER_KINDS if k not in LEARNED_KINDS),
    "shipper": tuple(k for k in CARRIER_KINDS if k not in LEARNED_KINDS),
}


class NegotiationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, role: str, base_seed: int = 0):
        if role not in ("carrier", "shipper"):
            raise ValueError(f"role must be 'carrier' or 'shipper', got {role!r}")
        self.role = role
        self.opp_role = "shipper" if role == "carrier" else "carrier"
        self.observation_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(len(FEATURE_NAMES),), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng(base_seed)
        self._episode_seed = base_seed

        # Filled in on reset(); typed here for clarity.
        self._runner: Optional[_TableRunner] = None
        self._opponent: Optional[DecisionEngine] = None
        self._sigma = 0.0
        self._refprice_mean = 0.0
        self._sale_price = 0.0
        self._goods_cost = 0.0
        self._refprice_est = 0.0
        self._global_turn = 1

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._episode_seed = int(self._rng.integers(0, 2**31))
        market = MarketState(self._episode_seed, 1)
        neg_rng = market.negotiation_rng(0)
        route = market.assign_route(neg_rng)
        self._sigma = route.refprice_sigma
        self._refprice_mean = route.refprice_mean
        # Computed unconditionally, not just "if self.role == shipper": the
        # OPPONENT needs a fully-formed state too, whenever it's the one
        # playing the shipper side (i.e. whenever self.role == "carrier").
        self._sale_price = market.sale_prices[0]
        self._goods_cost = market.goods_costs[0]
        self._refprice_est = route.refprice_mean + market.sample_shipper_epsilon(route.refprice_sigma, neg_rng)

        opp_kind = str(self._rng.choice(OPPONENT_KINDS[self.role]))
        opp_builder = build_shipper if self.role == "carrier" else build_carrier
        self._opponent = opp_builder(opp_kind, f"opp-{opp_kind}")

        shipper_id = "agent" if self.role == "shipper" else "opp"
        carrier_id = "opp" if self.role == "shipper" else "agent"
        self._runner = _TableRunner(shipper_id=shipper_id, carrier_id=carrier_id, sigma=self._sigma)
        self._global_turn = 1

        # Shipper always moves first (odd global turns); if the agent is
        # the carrier, let the opponent take its opening move before
        # handing control back.
        if self.role == "carrier":
            self._advance_opponent()

        return self._build_observation(), {}

    def _build_state(self, for_role: str) -> TableState:
        r = self._runner
        if for_role == "shipper":
            return TableState(
                role="shipper", table_id=r.table_id, negotiation_seed=self._episode_seed,
                turn=self._global_turn, turns_max=TURNS_MAX,
                my_turns_used=r.shipper_turns_used, opp_turns_used=r.carrier_turns_used,
                my_last_offer=r.last_shipper_offer, opp_last_offer=r.last_carrier_offer,
                history=list(r.history), sigma=self._sigma,
                sale_price=self._sale_price, goods_cost=self._goods_cost, refprice_est=self._refprice_est,
                my_portfolio=[],
            )
        return TableState(
            role="carrier", table_id=r.table_id, negotiation_seed=self._episode_seed,
            turn=self._global_turn, turns_max=TURNS_MAX,
            my_turns_used=r.carrier_turns_used, opp_turns_used=r.shipper_turns_used,
            my_last_offer=r.last_carrier_offer, opp_last_offer=r.last_shipper_offer,
            history=list(r.history), sigma=self._sigma, refprice_mean=self._refprice_mean,
            my_portfolio=[],
        )

    def _advance_opponent(self) -> Move:
        state = self._build_state(self.opp_role)
        move = self._opponent.next_move(state)
        _validate_move(move, state)
        self._runner.apply(self._global_turn, self.opp_role, move)
        self._global_turn += 1
        return move

    def _build_observation(self) -> np.ndarray:
        state = self._build_state(self.role)
        return np.array(features_to_vector(extract_features(state)), dtype=np.float32)

    def step(self, action):
        p = float(np.clip(action[0], 0.0, 1.0))
        state = self._build_state(self.role)
        worst, best = role_band(state)
        higher_is_better = self.role == "carrier"  # carrier: worst=RP_t (floor), best=target_t (ceiling)
        aspiration = worst + p * (best - worst)

        opp_offer = state.opp_last_offer
        legal_for_me = opp_offer is not None and ((opp_offer >= worst) if higher_is_better else (opp_offer <= worst))
        meets_aspiration = opp_offer is not None and ((opp_offer >= aspiration) if higher_is_better else (opp_offer <= aspiration))
        can_accept = legal_for_me and meets_aspiration

        turns_per_side = state.turns_per_side
        turn_index = state.my_turns_used + 1
        terminated = False
        reward = 0.0

        if self.role == "shipper" and not shipper_route_is_feasible(state):
            # Mirrors every hand-designed shipper engine's deliberate
            # walk-away (Strategy spec §13.0): even in the best case, no
            # legal deal can exist on this route. Also the one guard that
            # keeps `target` (the ambition anchor `worst`/`best` are built
            # from) from ever landing on the wrong side of RP -- see the
            # `higher_is_better` branch below for why that distinction is
            # load-bearing, not cosmetic.
            move = Move(action=Action.REJECT)
            self._runner.apply(self._global_turn, self.role, move)
            terminated = True
        elif can_accept:
            move = Move(action=Action.ACCEPT)
            _validate_move(move, state)
            self._runner.apply(self._global_turn, self.role, move)
            reward = value_bar(opp_offer, worst, best)
            terminated = True
        elif turn_index > turns_per_side:
            move = Move(action=Action.REJECT)
            self._runner.apply(self._global_turn, self.role, move)
            terminated = True
        else:
            if higher_is_better:  # carrier: both worst=RP_t and best=target_t are true hard bounds
                next_offer = min(max(aspiration, worst), best)
            else:  # shipper: the ONLY hard bound is worst=RP (a ceiling) -- no hard floor at all,
                # and `best`=target is just a soft tactical anchor that can (rarely) sit above RP
                # on edge-case routes, so it must never be used as a clamp bound here.
                next_offer = min(aspiration, worst)
            if state.my_last_offer is not None:
                next_offer = min(next_offer, state.my_last_offer) if higher_is_better else max(next_offer, state.my_last_offer)
            move = Move(action=Action.OFFER, amount=next_offer)
            _validate_move(move, state)
            self._runner.apply(self._global_turn, self.role, move)
            self._global_turn += 1

            opp_move = self._advance_opponent()
            if opp_move.action == Action.ACCEPT:
                reward = value_bar(next_offer, worst, best)
                terminated = True
            elif opp_move.action == Action.REJECT:
                terminated = True
            # else: opponent countered -- negotiation continues

        obs = np.zeros(len(FEATURE_NAMES), dtype=np.float32) if terminated else self._build_observation()
        return obs, reward, terminated, False, {}
