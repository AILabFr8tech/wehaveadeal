"""CarrierRL2: reinforcement-learning carrier, trained for competitive
awareness -- see `wehaveadeal/rl/league_env.py`'s module docstring for what
changed relative to `CarrierRL` and why. The ACTION mechanism at inference
time is identical to `CarrierRL`'s (this class is a near-verbatim copy):
the policy still outputs a single "aspiration" position in [0, 1] on the
hard band, and accepts on the spot whenever the shipper's current offer
already meets or beats it. What's different is entirely on the training
side -- the reward the policy was optimized against actually depended on
winning `wehaveadeal.matching.gale_shapley` against rival carriers, not just on
closing a legal deal in isolation (`CarrierRL`'s documented failure mode).

REGRESSION-TESTED BUG, real and confirmed by tracing a live round: this
class forces `my_portfolio=[]` before extracting features, discarding
whatever real sibling-table data `state` carries. `LeagueNegotiationEnv`
(faithfully mirroring `NegotiationEnv`'s existing scope) always trained
this policy with `portfolio_promise` pinned at its neutral 0.5 default --
the learning agent's OWN table never had siblings there, only the shared
counterpart did (see that module's docstring). Deployed into a real round
where this carrier genuinely negotiates several tables at once,
`portfolio_promise` stops being 0.5 (e.g. 0/4 siblings closed yet -> 0.0)
-- a value the network never saw in training, which is exactly what
produced the observed failure: a wildly miscalibrated aspiration,
confirmed by tracing `ShipperRL2` (the same bug, mirrored) opening at ~94%
of its own worst-case ceiling on turn 1 and getting accepted instantly by
every carrier it faced. Neutralizing the feature at inference restores the
exact distribution this policy was actually trained on. Teaching it to use
real portfolio signal is a legitimate future upgrade -- it would just need
`LeagueNegotiationEnv` itself to give the learner real siblings during
training too, not a fix applied only here at inference time.
"""
from __future__ import annotations

from pathlib import Path

from wehaveadeal._paths import baseline_artifact
from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.ml.features import extract_features, features_to_vector
from wehaveadeal.models import Move, TableState

DEFAULT_MODEL_PATH = baseline_artifact("rl/carrier_rl2_model.zip")


class CarrierRL2(DecisionEngine):
    engine_name = "CarrierRL2"
    version = "v1"

    def __init__(self, agent_id: str, frozen: bool = True, model_path: str = DEFAULT_MODEL_PATH) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.model_path = model_path
        self._model = None  # lazily loaded on first decision, see _predict

    def _load_model(self):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError(
                "CarrierRL2 needs torch/stable-baselines3/gymnasium -- run `pip install -r requirements.txt`."
            ) from exc
        path = Path(self.model_path)
        if not path.exists():
            raise RuntimeError(
                f"No trained model at {path}. Run `python -m wehaveadeal.rl.train_league --role carrier` first, "
                "or pass a different model_path."
            )
        return PPO.load(str(path))

    def _predict(self, state: TableState) -> float:
        if self._model is None:
            self._model = self._load_model()
        import numpy as np

        # See the module docstring: this policy was trained with
        # portfolio_promise pinned at its neutral 0.5 default and must see
        # exactly that at inference too, or its output is garbage.
        training_state = state.model_copy(update={"my_portfolio": []})
        obs = np.array(features_to_vector(extract_features(training_state)), dtype=np.float32)
        action, _ = self._model.predict(obs, deterministic=True)
        return float(min(max(action[0], 0.0), 1.0))

    def next_move(self, state: TableState) -> Move:
        rp_t = state.refprice_mean - 0.5 * state.sigma
        target_t = state.refprice_mean + 0.5 * state.sigma
        turn_index = state.my_turns_used + 1
        turns_per_side = state.turns_per_side
        shipper_offer = state.opp_last_offer

        if turn_index > turns_per_side:
            return REJECT

        if shipper_offer is not None and turn_index >= turns_per_side and shipper_offer >= rp_t:
            return ACCEPT

        p = self._predict(state)
        aspiration = rp_t + p * (target_t - rp_t)

        if shipper_offer is not None and shipper_offer >= rp_t and shipper_offer >= aspiration:
            return ACCEPT

        next_counter = min(max(aspiration, rp_t), target_t)
        if state.my_last_offer is not None:
            next_counter = min(next_counter, state.my_last_offer)
        return make_offer(next_counter)
