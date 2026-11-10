"""CarrierRL: reinforcement-learning carrier. Unlike `CarrierML` (imitation
learning, which deliberately keeps the accept/reject decision hardcoded --
see that module's docstring), `CarrierRL`'s trained policy controls the
WHOLE decision: what to offer AND when to close. It was trained with PPO
(stable-baselines3) against `wehaveadeal.rl.env.NegotiationEnv`, and this class
replays exactly the same accept-or-counter mechanism the environment used
during training, so what the policy learned is exactly what runs here, not
an approximation of it.

The policy outputs a single number in [0, 1]: its own "aspiration" position
on the hard band `[RP_t, target_t]` for this turn (same units `value_bar`
uses everywhere in this package -- see `wehaveadeal/ml/features.py`). If the
shipper's current offer already meets or beats that aspiration (and is
legal, >= RP_t), this accepts on the spot -- the policy's own learned
judgment about when a deal is good enough decides that, not a hand-derived
value-function comparison. Otherwise it counters AT the aspiration, clamped
into the hard band and never allowed to exceed its own last offer -- the
exact same legality/monotonicity guarantee every other engine in this
package has, applied regardless of what the model outputs. The hard
last-turn floor (accept unconditionally if legal, out of turns) is checked
before consulting the model at all, same as every other engine.
"""
from __future__ import annotations

from pathlib import Path

from wehaveadeal._paths import baseline_artifact
from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.ml.features import extract_features, features_to_vector
from wehaveadeal.models import Move, TableState

DEFAULT_MODEL_PATH = baseline_artifact("rl/carrier_rl_model.zip")


class CarrierRL(DecisionEngine):
    engine_name = "CarrierRL"
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
                "CarrierRL needs torch/stable-baselines3/gymnasium -- run `pip install -r requirements.txt`."
            ) from exc
        path = Path(self.model_path)
        if not path.exists():
            raise RuntimeError(
                f"No trained model at {path}. Run `python -m wehaveadeal.rl.train --role carrier` first, "
                "or pass a different model_path."
            )
        return PPO.load(str(path))

    def _predict(self, state: TableState) -> float:
        if self._model is None:
            self._model = self._load_model()
        import numpy as np

        obs = np.array(features_to_vector(extract_features(state)), dtype=np.float32)
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
