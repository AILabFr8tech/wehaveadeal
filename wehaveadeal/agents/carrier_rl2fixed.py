"""CarrierRL2Fixed: ablation variant of `CarrierRL2`, retrained against the
corrected `wehaveadeal.rl.league_env.LeagueNegotiationEnv` (see that module's
`_build_state`), which now gives the learner real portfolio visibility
across its own sibling tables during training instead of pinning
`my_portfolio` at its neutral default. This class is `CarrierRL2` with the
inference-time neutralization removed: it feeds the model the real,
unmodified `state`, since the model it loads was actually trained on that
real distribution. Not registered in `wehaveadeal.simulate`'s `CARRIER_KINDS` --
this is a paper-ablation-only variant, loaded directly by the ablation
script that compares it against the original (buggy) `CarrierRL2`, not a
tenth architecture in the main nine-architecture tournament.
"""
from __future__ import annotations

from pathlib import Path

from wehaveadeal._paths import baseline_artifact
from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.ml.features import extract_features, features_to_vector
from wehaveadeal.models import Move, TableState

DEFAULT_MODEL_PATH = baseline_artifact("rl/carrier_rl2fixed_model.zip")


class CarrierRL2Fixed(DecisionEngine):
    engine_name = "CarrierRL2Fixed"
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
                "CarrierRL2Fixed needs torch/stable-baselines3/gymnasium -- run `pip install -r requirements.txt`."
            ) from exc
        path = Path(self.model_path)
        if not path.exists():
            raise RuntimeError(
                f"No trained model at {path}. Run `python -m wehaveadeal.rl.train_league --role carrier "
                f"--out {path}` against the fixed LeagueNegotiationEnv first, or pass a different model_path."
            )
        return PPO.load(str(path))

    def _predict(self, state: TableState) -> float:
        if self._model is None:
            self._model = self._load_model()
        import numpy as np

        # Unlike CarrierRL2, no neutralization: this model was trained on the
        # real portfolio distribution, so it must see the real state here too.
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
