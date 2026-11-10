"""ShipperRL: reinforcement-learning shipper, mirroring `CarrierRL` exactly
-- see that module for the full rationale. Mirrored specifics: worst=RP
(ceiling), best=target (ambitious low price); the walk-away feasibility
check (`shipper_route_is_feasible`) runs before consulting the model at
all, same as every hand-designed shipper engine and same as
`wehaveadeal.rl.env.NegotiationEnv` used during training; the aspiration maps to
a price DECREASING from RP toward target as the policy's output rises
toward 1 (the mirror image of `CarrierRL`'s floor-to-ceiling mapping); and
there is no hard floor to clamp against, only the hard ceiling RP -- see
the paper Section 6.0 for why that asymmetry is
real, not an oversight.
"""
from __future__ import annotations

from pathlib import Path

from wehaveadeal._paths import baseline_artifact
from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.market import SHIPPER_NOISE_FRACTION
from wehaveadeal.ml.features import extract_features, features_to_vector
from wehaveadeal.models import Move, TableState
from wehaveadeal.value_functions import shipper_route_is_feasible

DEFAULT_MODEL_PATH = baseline_artifact("rl/shipper_rl_model.zip")


class ShipperRL(DecisionEngine):
    engine_name = "ShipperRL"
    version = "v1"

    def __init__(
        self, agent_id: str, frozen: bool = True, model_path: str = DEFAULT_MODEL_PATH, target_cushion_k: float = 0.5
    ) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.model_path = model_path
        self.target_cushion_k = target_cushion_k
        self._model = None  # lazily loaded on first decision, see _predict

    def _band(self, state: TableState) -> tuple[float, float]:
        rp = state.sale_price - state.goods_cost
        sigma_eps = state.sigma * SHIPPER_NOISE_FRACTION
        target = state.refprice_est - 0.5 * state.sigma + self.target_cushion_k * sigma_eps
        return rp, target

    def _load_model(self):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError(
                "ShipperRL needs torch/stable-baselines3/gymnasium -- run `pip install -r requirements.txt`."
            ) from exc
        path = Path(self.model_path)
        if not path.exists():
            raise RuntimeError(
                f"No trained model at {path}. Run `python -m wehaveadeal.rl.train --role shipper` first, "
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
        rp, target = self._band(state)
        carrier_offer = state.opp_last_offer

        if not shipper_route_is_feasible(state):
            return REJECT

        turn_index = state.my_turns_used + 1
        turns_per_side = state.turns_per_side
        if turn_index > turns_per_side:
            return REJECT

        if carrier_offer is not None and turn_index >= turns_per_side and carrier_offer <= rp:
            return ACCEPT

        p = self._predict(state)
        aspiration = rp + p * (target - rp)

        if carrier_offer is not None and carrier_offer <= rp and carrier_offer <= aspiration:
            return ACCEPT

        next_offer = min(aspiration, rp)
        if state.my_last_offer is not None:
            next_offer = max(next_offer, state.my_last_offer)
        return make_offer(next_offer)
