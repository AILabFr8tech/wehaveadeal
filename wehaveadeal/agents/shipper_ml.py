"""ShipperML: imitation-learning shipper, mirroring `CarrierML` exactly --
see that module for the full rationale (learn "what would the teacher
offer" from self-play data instead of a hand-written rule; the model only
ever predicts a [0,1] band position, clamped, then mapped to a price; the
same hard clamp + monotonic-concession safety every other shipper engine
applies still runs afterward regardless of what the model outputs; the
accept-vs-continue decision stays grounded in the shared value function
rather than the model -- and, per `carrier_ml.py`'s module docstring, that
comparison has to anchor on `my_last_offer`, not the model's freshly
predicted `x`, or it triggers accepting far too early).

Mirrored specifics: worst=RP (ceiling), best=target (ambitious low price);
walk-away feasibility check via `shipper_route_is_feasible` (no carrier-side
equivalent needed, see the paper Section 6.5); the
predicted position is mapped to a price INCREASING from `target` to `RP`
(the mirror image of CarrierML's ceiling-to-floor mapping).
"""
from __future__ import annotations

from pathlib import Path

from wehaveadeal._paths import baseline_artifact, load_sklearn_model
from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.ml.features import extract_features, features_to_vector
from wehaveadeal.models import Move, TableState
from wehaveadeal.value_functions import portfolio_occupancy, shipper_route_is_feasible, shipper_value

DEFAULT_MODEL_PATH = baseline_artifact("ml/shipper_ml_model.joblib")


class ShipperML(DecisionEngine):
    engine_name = "ShipperML"
    version = "v1"

    def __init__(
        self,
        agent_id: str,
        frozen: bool = True,
        model_path: str = DEFAULT_MODEL_PATH,
        target_cushion_k: float = 0.5,
        firmness: float = 0.25,
        weight_assignment: float = 0.5,
    ) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.model_path = model_path
        self.target_cushion_k = target_cushion_k
        self.firmness = firmness
        self.weight_assignment = weight_assignment
        self._model = None  # lazily loaded on first decision, see _predict

    def _band(self, state: TableState) -> tuple[float, float]:
        from wehaveadeal.market import SHIPPER_NOISE_FRACTION

        rp = state.sale_price - state.goods_cost
        sigma_eps = state.sigma * SHIPPER_NOISE_FRACTION
        target = state.refprice_est - 0.5 * state.sigma + self.target_cushion_k * sigma_eps
        return rp, target

    def _load_model(self):
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError(
                "ShipperML needs scikit-learn/joblib -- run `pip install -r requirements.txt`."
            ) from exc
        path = Path(self.model_path)
        if not path.exists():
            raise RuntimeError(
                f"No trained model at {path}. Run wehaveadeal/ml/collect.py then wehaveadeal/ml/train.py "
                "(--role shipper) first, or pass a different model_path."
            )
        return load_sklearn_model(path)

    def _predict(self, state: TableState) -> float:
        if self._model is None:
            self._model = self._load_model()
        vector = features_to_vector(extract_features(state))
        p = float(self._model.predict([vector])[0])
        return min(max(p, 0.0), 1.0)

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

        target_eff = target + self.firmness * (rp - target)
        p_assignment = portfolio_occupancy(state.my_portfolio, target_eff, higher_is_better=False)

        p = self._predict(state)
        x = rp + p * (target - rp)

        if carrier_offer is not None and carrier_offer <= rp:
            v_accept_now = shipper_value(
                rp=rp, target=target, terminal=True, close_price=carrier_offer,
                turn=state.turn, turns_max=state.turns_max,
                weight_assignment=self.weight_assignment, p_assignment=p_assignment,
            )
            v_continue = shipper_value(
                rp=rp, target=target, terminal=False,
                carrier_offer_on_table=carrier_offer, my_last_offer=state.my_last_offer,
                turn=state.turn, turns_max=state.turns_max,
                weight_assignment=self.weight_assignment, p_assignment=p_assignment,
            )
            if v_accept_now >= v_continue:
                return ACCEPT

        next_offer = min(x, rp)
        if state.my_last_offer is not None:
            next_offer = max(next_offer, state.my_last_offer)
        return make_offer(next_offer)
