"""LeagueNegotiationEnv: the learning agent's own bilateral table, embedded
in a small competitive round -- fixing `wehaveadeal.rl.env.NegotiationEnv`'s
documented root cause (see that module's docstring and `wehaveadeal/rl/train.py`'s
"KNOWN CONSEQUENCE" section): a policy trained against exactly one frozen
opponent, with no rival on its own side, has no way to learn that a merely
ACCEPTABLE price isn't enough -- it has to be competitive, because
`wehaveadeal.matching.gale_shapley` only keeps the cheapest (for a shipper) or
priciest (for a carrier) CLOSED offer and discards the rest.

The fix is not to show the agent any rival's offer -- a real broker never
sees a competitor's live negotiation either. It's to make the agent's OWN
reward actually depend on winning the match, by really running the
competition: `n_rivals` additional hand-designed agents on the SAME side as
the learner, all courting the exact same single counterpart, with
`gale_shapley` deciding at the end who the counterpart actually goes with.
The counterpart itself gets real `my_portfolio` visibility across all of
its tables (mine + every rival's) via `wehaveadeal.simulate._portfolio_snapshot`
-- the one piece of "sensing competition" that's completely legitimate,
because it's the counterpart's OWN negotiation history, not a rival's.

Reward, at the learning agent's OWN table's outcome:
  - never reached any deal (impasse, incl. a shipper's deliberate
    walk-away): `reward_impasse` (default -0.5) -- the worst outcome, a
    real broker's idle truck / unfulfilled shipment.
  - reached a legal deal, but the counterpart's Gale-Shapley preference
    went to a rival instead: `reward_closed_lost` (default -0.15) -- a
    real but smaller cost (turns spent, no margin), and deliberately
    *less* bad than impasse, so the policy isn't pushed toward "never
    close" as a way to dodge the harsher penalty.
  - reached a legal deal AND kept the match: `value_bar(rate, worst,
    best)` in [0, 1], same scale as `NegotiationEnv` and every
    imitation-learning engine.

Mechanically: reuses `wehaveadeal.simulate`'s already-tested single-table
machinery (`_TableRunner`, `_validate_move`, `_portfolio_snapshot`) and
`wehaveadeal.matching.gale_shapley` unchanged, exactly like `NegotiationEnv`
already reuses `_TableRunner`/`_validate_move` -- this is that same
"don't re-derive the turn loop" discipline, extended to more than one
table. The turn loop is written as a generator that `yield`s the
observation only at the learning agent's own decision points (everyone
else's moves -- the shared counterpart's and every rival's -- are decided
internally by their own `DecisionEngine.next_move`, exactly like a normal
round); `step()` drives it with `generator.send(action)` and catches the
`StopIteration` the generator raises (via a bare `return`) once every
table has run its course and `gale_shapley` has been scored.

Scope, stated up front like `NegotiationEnv`'s docstring does: rivals and
the counterpart are all hand-designed/frozen (drawn fresh each episode from
R/RV/IS/OPT/OPT2), not other copies of this same policy mid-training -- a
true self-play "league" (mixing in frozen past checkpoints of this same
policy) is a natural follow-up, not attempted in this first pass.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover - exercised only when the RL extra isn't installed
    raise ImportError("wehaveadeal.rl needs gymnasium/stable-baselines3 -- run `pip install -r requirements.txt`.") from exc

from wehaveadeal.agents.base import DecisionEngine
from wehaveadeal.market import MarketState
from wehaveadeal.matching import gale_shapley
from wehaveadeal.ml.features import FEATURE_NAMES, extract_features, features_to_vector, role_band
from wehaveadeal.models import Action, Move, TableState
from wehaveadeal.rl.env import LEARNED_KINDS, OPPONENT_KINDS
from wehaveadeal.simulate import CARRIER_KINDS, SHIPPER_KINDS, TURNS_MAX, _TableRunner, _portfolio_snapshot, _validate_move, build_carrier, build_shipper
from wehaveadeal.value_functions import shipper_route_is_feasible, value_bar

# Rivals sit on the SAME side as the learner (other carriers, if training a
# carrier; other shippers, if training a shipper) -- unlike OPPONENT_KINDS
# (imported above), which is the OTHER role's hand-designed roster, used
# here for the single shared counterpart, same as in NegotiationEnv.
RIVAL_KINDS = {
    "carrier": tuple(k for k in CARRIER_KINDS if k not in LEARNED_KINDS),
    "shipper": tuple(k for k in SHIPPER_KINDS if k not in LEARNED_KINDS),
}


class LeagueNegotiationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        role: str,
        base_seed: int = 0,
        min_rivals: int = 1,
        max_rivals: int = 2,
        reward_impasse: float = -0.5,
        reward_closed_lost: float = -0.15,
    ):
        if role not in ("carrier", "shipper"):
            raise ValueError(f"role must be 'carrier' or 'shipper', got {role!r}")
        if min_rivals < 0 or max_rivals < min_rivals:
            raise ValueError(f"need 0 <= min_rivals <= max_rivals, got min_rivals={min_rivals} max_rivals={max_rivals}")
        self.role = role
        self.opp_role = "shipper" if role == "carrier" else "carrier"
        self.min_rivals = min_rivals
        self.max_rivals = max_rivals
        self.reward_impasse = reward_impasse
        self.reward_closed_lost = reward_closed_lost
        self.observation_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(len(FEATURE_NAMES),), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng(base_seed)

        # Filled in on reset(); typed here for clarity.
        self._episode_seed = base_seed
        self._my_id = "agent"
        self._counterpart_id = "counterpart"
        self._counterpart_engine: Optional[DecisionEngine] = None
        self._rival_engines: Dict[str, DecisionEngine] = {}
        self._my_runner: Optional[_TableRunner] = None
        self._runners: list = []
        self._sale_price = 0.0
        self._goods_cost = 0.0
        # Per-entity route data. When role == "carrier" there's one shared
        # route (the single counterpart shipper's), used for every table.
        # When role == "shipper", each shipper (me + each rival) has its
        # OWN route -- `_rival_route[...]` holds the rivals'; mine lives in
        # the plain `_sigma`/`_refprice_mean`/`_refprice_est` attributes.
        self._sigma = 0.0
        self._refprice_mean = 0.0
        self._refprice_est = 0.0
        self._rival_route: Dict[str, tuple] = {}
        self._my_worst = 0.0
        self._my_best = 0.0
        self._gen = None

    def _route_for(self, r: _TableRunner) -> tuple:
        """(sigma, refprice_mean, refprice_est, sale_price, goods_cost) for
        table `r` -- shared for every table when training a carrier (they
        all court the same one shipper/route); per-shipper when training a
        shipper (see the module docstring's mirroring note)."""
        if self.role == "carrier" or r.shipper_id == self._my_id:
            return self._sigma, self._refprice_mean, self._refprice_est, self._sale_price, self._goods_cost
        return self._rival_route[r.shipper_id] + (self._sale_price, self._goods_cost)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        for _attempt in range(50):
            obs, info = self._reset_once()
            if obs is not None:
                return obs, info
        raise RuntimeError(
            "LeagueNegotiationEnv: 50 consecutive episodes ended before the learning agent ever got a single "
            "turn -- check whether the opponent/rival roster has become near-universally infeasible."
        )

    def _reset_once(self):
        """One episode-sampling attempt. Returns (None, {}) if the episode
        already ended (my table closed/rejected) before the learning agent
        ever reached its own decision point -- e.g. the shared counterpart
        rejects on turn 1 for every one of its tables at once (role ==
        "carrier"), or my own walk-away check fires immediately (role ==
        "shipper"). That's not a real training example (no action of mine
        caused it, so there's nothing to learn from its reward), and -- more
        importantly -- letting `reset()` return a normal-looking observation
        for an already-exhausted generator would crash the NEXT `step()`
        call (`generator.send()` on an exhausted generator raises
        `StopIteration` with no value). Rerolling a fresh episode is the
        correct fix, not a workaround: `NegotiationEnv` never has an
        equivalent case, because there termination is only ever produced
        inside `step()`, never inside `reset()`."""
        self._episode_seed = int(self._rng.integers(0, 2**31))
        market = MarketState(self._episode_seed, 1)
        neg_rng = market.negotiation_rng(0)
        self._sale_price = market.sale_prices[0]
        self._goods_cost = market.goods_costs[0]

        n_rivals = int(self._rng.integers(self.min_rivals, self.max_rivals + 1))
        rival_kinds = [str(k) for k in self._rng.choice(RIVAL_KINDS[self.role], size=n_rivals)] if n_rivals > 0 else []
        counterpart_kind = str(self._rng.choice(OPPONENT_KINDS[self.role]))

        rival_runners: Dict[str, _TableRunner] = {}
        self._rival_engines = {}
        self._rival_route = {}

        if self.role == "carrier":
            route = market.assign_route(neg_rng)
            self._sigma = route.refprice_sigma
            self._refprice_mean = route.refprice_mean
            self._refprice_est = route.refprice_mean + market.sample_shipper_epsilon(route.refprice_sigma, neg_rng)

            self._counterpart_engine = build_shipper(counterpart_kind, self._counterpart_id)
            for i, kind in enumerate(rival_kinds):
                rid = f"rival-{i}-{kind}"
                self._rival_engines[rid] = build_carrier(kind, rid)
                rival_runners[rid] = _TableRunner(shipper_id=self._counterpart_id, carrier_id=rid, sigma=self._sigma)
            self._my_runner = _TableRunner(shipper_id=self._counterpart_id, carrier_id=self._my_id, sigma=self._sigma)
        else:
            route = market.assign_route(neg_rng)
            self._sigma = route.refprice_sigma
            self._refprice_mean = route.refprice_mean
            self._refprice_est = route.refprice_mean + market.sample_shipper_epsilon(route.refprice_sigma, neg_rng)

            self._counterpart_engine = build_carrier(counterpart_kind, self._counterpart_id)
            for i, kind in enumerate(rival_kinds):
                rid = f"rival-{i}-{kind}"
                r_route = market.assign_route(neg_rng)
                r_refprice_est = r_route.refprice_mean + market.sample_shipper_epsilon(r_route.refprice_sigma, neg_rng)
                self._rival_route[rid] = (r_route.refprice_sigma, r_route.refprice_mean, r_refprice_est)
                self._rival_engines[rid] = build_shipper(kind, rid)
                rival_runners[rid] = _TableRunner(shipper_id=rid, carrier_id=self._counterpart_id, sigma=r_route.refprice_sigma)
            self._my_runner = _TableRunner(shipper_id=self._my_id, carrier_id=self._counterpart_id, sigma=self._sigma)

        self._runners = [self._my_runner] + list(rival_runners.values())
        self._my_worst, self._my_best = role_band(self._build_state(self._my_runner, self.role, 1))

        self._gen = self._play()
        try:
            first_state = next(self._gen)
        except StopIteration:
            return None, {}
        return self._obs(first_state), {}

    def _build_state(self, r: _TableRunner, acting_role: str, global_turn: int) -> TableState:
        sigma, refprice_mean, refprice_est, sale_price, goods_cost = self._route_for(r)
        # The counterpart always gets real portfolio visibility across all of its
        # tables (see module docstring). The learner's OWN table also gets real
        # visibility across its siblings (itself excluded) when it is the one
        # deciding -- the exact signal `wehaveadeal.simulate._portfolio_snapshot` gives
        # it at real deployment, where it is genuinely negotiating several tables
        # at once. Hand-designed rivals keep seeing `[]` here, same as before:
        # fixing their training-time portfolio view is out of scope for this fix,
        # which targets only the learner's own train/deploy feature mismatch
        # (see `wehaveadeal/agents/carrier_rl2.py`'s module docstring).
        if acting_role == self.opp_role or (acting_role == self.role and r is self._my_runner):
            portfolio = _portfolio_snapshot(self._runners, acting_role, r)
        else:
            portfolio = []
        if acting_role == "shipper":
            return TableState(
                role="shipper", table_id=r.table_id, negotiation_seed=self._episode_seed,
                turn=global_turn, turns_max=TURNS_MAX,
                my_turns_used=r.shipper_turns_used, opp_turns_used=r.carrier_turns_used,
                my_last_offer=r.last_shipper_offer, opp_last_offer=r.last_carrier_offer,
                history=list(r.history), sigma=sigma,
                sale_price=sale_price, goods_cost=goods_cost, refprice_est=refprice_est,
                my_portfolio=portfolio,
            )
        return TableState(
            role="carrier", table_id=r.table_id, negotiation_seed=self._episode_seed,
            turn=global_turn, turns_max=TURNS_MAX,
            my_turns_used=r.carrier_turns_used, opp_turns_used=r.shipper_turns_used,
            my_last_offer=r.last_carrier_offer, opp_last_offer=r.last_shipper_offer,
            history=list(r.history), sigma=sigma, refprice_mean=refprice_mean,
            my_portfolio=portfolio,
        )

    def _obs(self, state: TableState) -> np.ndarray:
        return np.array(features_to_vector(extract_features(state)), dtype=np.float32)

    def _decide_my_move(self, state: TableState):
        """Generator: mirrors `NegotiationEnv.step()`'s per-turn decision
        logic exactly (hard guards first, then the aspiration-vs-offer
        accept rule), just restructured so its `return`s become the
        guarded moves and its one live decision point `yield`s the state
        instead of returning immediately -- `yield from` in `_play` passes
        an externally-sent action straight through to the `yield` below."""
        if self.role == "shipper" and not shipper_route_is_feasible(state):
            return Move(action=Action.REJECT)
        turns_per_side = state.turns_per_side
        turn_index = state.my_turns_used + 1
        if turn_index > turns_per_side:
            return Move(action=Action.REJECT)

        action = yield state
        p = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        worst, best = role_band(state)
        higher_is_better = self.role == "carrier"
        aspiration = worst + p * (best - worst)
        opp_offer = state.opp_last_offer
        legal_for_me = opp_offer is not None and ((opp_offer >= worst) if higher_is_better else (opp_offer <= worst))
        meets_aspiration = opp_offer is not None and ((opp_offer >= aspiration) if higher_is_better else (opp_offer <= aspiration))
        if legal_for_me and meets_aspiration:
            return Move(action=Action.ACCEPT)

        if higher_is_better:
            next_offer = min(max(aspiration, worst), best)
        else:
            next_offer = min(aspiration, worst)
        if state.my_last_offer is not None:
            next_offer = min(next_offer, state.my_last_offer) if higher_is_better else max(next_offer, state.my_last_offer)
        return Move(action=Action.OFFER, amount=next_offer)

    def _play(self):
        for global_turn in range(1, TURNS_MAX + 1):
            acting_role = "shipper" if global_turn % 2 == 1 else "carrier"
            for r in self._runners:
                if r.is_done():
                    continue
                state = self._build_state(r, acting_role, global_turn)
                if acting_role == self.role and r is self._my_runner:
                    move = yield from self._decide_my_move(state)
                elif acting_role == self.role:
                    rival_id = r.carrier_id if self.role == "carrier" else r.shipper_id
                    move = self._rival_engines[rival_id].next_move(state)
                else:
                    move = self._counterpart_engine.next_move(state)
                _validate_move(move, state)
                r.apply(global_turn, acting_role, move)
        return self._finalize()

    def _finalize(self):
        tables = [r.to_result() for r in self._runners]
        if self.role == "carrier":
            shipper_ids = [self._counterpart_id]
        else:
            shipper_ids = [self._my_id] + list(self._rival_engines.keys())
        matching = gale_shapley(tables, shipper_ids, seed=self._episode_seed)

        my_result = self._my_runner.to_result()
        if not my_result.closed:
            return self.reward_impasse, {"outcome": "impasse"}

        if self.role == "carrier":
            winner = next((m.carrier_id for m in matching.matches if m.shipper_id == self._counterpart_id), None)
        else:
            winner = next((m.shipper_id for m in matching.matches if m.carrier_id == self._counterpart_id), None)
        if winner != self._my_id:
            return self.reward_closed_lost, {"outcome": "closed_lost"}

        return value_bar(my_result.rate, self._my_worst, self._my_best), {"outcome": "won", "rate": my_result.rate}

    def step(self, action):
        try:
            state = self._gen.send(np.asarray(action, dtype=np.float32))
        except StopIteration as stop:
            reward, info = stop.value
            return np.zeros(len(FEATURE_NAMES), dtype=np.float32), reward, True, False, info
        return self._obs(state), 0.0, False, False, {}
