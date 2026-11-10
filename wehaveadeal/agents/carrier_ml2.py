"""CarrierML2: imitation-learning carrier, extending `CarrierML` with a
second learned model that also decides WHEN to accept -- the one decision
`CarrierML` deliberately kept hardcoded (see that module's docstring).

Two INDEPENDENT models, not one stacked pipeline. An offer regressor
(same kind as CarrierML's, trained on its own self-play data) predicts what
to counter with; a separate classifier predicts P(accept) for the
opponent's current offer, using the exact same feature vector -- neither
model's output feeds into the other as an input. That's a deliberate
choice, not a missed optimization: feeding the regressor's freshly
predicted (and not yet offered) counter into the classifier as a feature
would risk recreating, conceptually, the exact bug `CarrierML` shipped with
once already -- anchoring an accept decision on a hypothetical value
instead of the negotiation's actual, already-confirmed state (see
`carrier_ml.py`'s module docstring). See `wehaveadeal/ml/train_accept.py` for
how the classifier is trained.

Same defense-in-depth guarantees as every other engine in this package: the
hard last-turn floor (accept unconditionally if legal, out of turns) is
checked before consulting either model, and whatever the offer regressor
predicts still gets clamped into the hard band and the monotonic-concession
constraint -- nothing here can produce an illegal or non-monotonic move, no
matter what either model outputs.
"""
from __future__ import annotations

from pathlib import Path

from wehaveadeal._paths import baseline_artifact, load_sklearn_model
from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.ml.features import extract_features, features_to_vector
from wehaveadeal.models import Move, TableState

DEFAULT_OFFER_MODEL_PATH = baseline_artifact("ml/carrier_ml2_offer_model.joblib")
DEFAULT_ACCEPT_MODEL_PATH = baseline_artifact("ml/carrier_ml2_accept_model.joblib")


class CarrierML2(DecisionEngine):
    engine_name = "CarrierML2"
    version = "v1"

    def __init__(
        self,
        agent_id: str,
        frozen: bool = True,
        offer_model_path: str = DEFAULT_OFFER_MODEL_PATH,
        accept_model_path: str = DEFAULT_ACCEPT_MODEL_PATH,
        accept_threshold: float = 0.5,
    ) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.offer_model_path = offer_model_path
        self.accept_model_path = accept_model_path
        self.accept_threshold = accept_threshold
        self._offer_model = None  # lazily loaded, see _predict_offer_position
        self._accept_model = None  # lazily loaded, see _predict_accept_probability

    def _load_joblib(self, path_str: str, purpose: str, train_hint: str):
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError(
                "CarrierML2 needs scikit-learn/joblib -- run `pip install -r requirements.txt`."
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
            self._offer_model = self._load_joblib(self.offer_model_path, "offer", "wehaveadeal/ml/train.py --role carrier")
        vector = features_to_vector(extract_features(state))
        p = float(self._offer_model.predict([vector])[0])
        return min(max(p, 0.0), 1.0)

    def _predict_accept_probability(self, state: TableState) -> float:
        if self._accept_model is None:
            self._accept_model = self._load_joblib(
                self.accept_model_path, "accept", "wehaveadeal/ml/train_accept.py --role carrier"
            )
        vector = features_to_vector(extract_features(state))
        return float(self._accept_model.predict_proba([vector])[0][1])

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

        if shipper_offer is not None and shipper_offer >= rp_t:
            if self._predict_accept_probability(state) >= self.accept_threshold:
                return ACCEPT

        p = self._predict_offer_position(state)
        x = rp_t + p * (target_t - rp_t)

        next_counter = min(max(x, rp_t), target_t)
        if state.my_last_offer is not None:
            next_counter = min(next_counter, state.my_last_offer)
        return make_offer(next_counter)
