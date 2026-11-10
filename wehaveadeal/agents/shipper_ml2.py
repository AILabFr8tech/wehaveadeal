"""ShipperML2: imitation-learning shipper, mirroring `CarrierML2` exactly --
see that module for the full rationale (two independent learned models, an
offer regressor and an accept classifier, neither feeding into the other).
Mirrored specifics: worst=RP (ceiling), best=target (ambitious low price);
walk-away feasibility check via `shipper_route_is_feasible` (no carrier-side
equivalent needed, see the paper Section 6.0); the
predicted offer position maps to a price DECREASING from RP toward target,
the mirror image of CarrierML2's floor-to-ceiling mapping.
"""
from __future__ import annotations

from pathlib import Path

from wehaveadeal._paths import baseline_artifact, load_sklearn_model
from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.market import SHIPPER_NOISE_FRACTION
from wehaveadeal.ml.features import extract_features, features_to_vector
from wehaveadeal.models import Move, TableState
from wehaveadeal.value_functions import shipper_route_is_feasible

DEFAULT_OFFER_MODEL_PATH = baseline_artifact("ml/shipper_ml2_offer_model.joblib")
DEFAULT_ACCEPT_MODEL_PATH = baseline_artifact("ml/shipper_ml2_accept_model.joblib")


class ShipperML2(DecisionEngine):
    engine_name = "ShipperML2"
    version = "v1"

    def __init__(
        self,
        agent_id: str,
        frozen: bool = True,
        offer_model_path: str = DEFAULT_OFFER_MODEL_PATH,
        accept_model_path: str = DEFAULT_ACCEPT_MODEL_PATH,
        accept_threshold: float = 0.5,
        target_cushion_k: float = 0.5,
    ) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.offer_model_path = offer_model_path
        self.accept_model_path = accept_model_path
        self.accept_threshold = accept_threshold
        self.target_cushion_k = target_cushion_k
        self._offer_model = None  # lazily loaded, see _predict_offer_position
        self._accept_model = None  # lazily loaded, see _predict_accept_probability

    def _band(self, state: TableState) -> tuple[float, float]:
        rp = state.sale_price - state.goods_cost
        sigma_eps = state.sigma * SHIPPER_NOISE_FRACTION
        target = state.refprice_est - 0.5 * state.sigma + self.target_cushion_k * sigma_eps
        return rp, target

    def _load_joblib(self, path_str: str, purpose: str, train_hint: str):
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError(
                "ShipperML2 needs scikit-learn/joblib -- run `pip install -r requirements.txt`."
            ) from exc
        path = Path(path_str)
        if not path.exists():
            raise RuntimeError(
                f"No trained {purpose} model at {path}. Run wehaveadeal/ml/collect.py then {train_hint} first, "
                "or pass a different model path."
            )
        return load_sklearn_model(path)

    def _predict_offer_position(self, state: TableState) -> float:
        if self._offer_model is None:
            self._offer_model = self._load_joblib(self.offer_model_path, "offer", "wehaveadeal/ml/train.py --role shipper")
        vector = features_to_vector(extract_features(state))
        p = float(self._offer_model.predict([vector])[0])
        return min(max(p, 0.0), 1.0)

    def _predict_accept_probability(self, state: TableState) -> float:
        if self._accept_model is None:
            self._accept_model = self._load_joblib(
                self.accept_model_path, "accept", "wehaveadeal/ml/train_accept.py --role shipper"
            )
        vector = features_to_vector(extract_features(state))
        return float(self._accept_model.predict_proba([vector])[0][1])

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

        if carrier_offer is not None and carrier_offer <= rp:
            if self._predict_accept_probability(state) >= self.accept_threshold:
                return ACCEPT

        p = self._predict_offer_position(state)
        x = rp + p * (target - rp)

        next_offer = min(x, rp)
        if state.my_last_offer is not None:
            next_offer = max(next_offer, state.my_last_offer)
        return make_offer(next_offer)
