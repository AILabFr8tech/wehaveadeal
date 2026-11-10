"""CarrierML: imitation-learning carrier. Learns "what would the teacher
offer here" from self-play data (see `wehaveadeal/ml/collect.py` /
`wehaveadeal/ml/train.py`) instead of computing the offer from a hand-written
rule/value-function/optimizer.

The trained model only ever predicts a POSITION on the hard band
`[RP_t, target_t]` (a 0-1 fraction, the same units `value_bar` already
uses everywhere in this package -- see `wehaveadeal/ml/features.py`), never a
raw price directly, and that prediction is clamped into `[0, 1]` before
being mapped back to a price. Whatever the model outputs, this class still
applies the exact same hard-band + monotonic-concession clamp every other
carrier engine applies, so a bad prediction can degrade the offer, but can
never produce an illegal or non-monotonic move.

The accept-vs-continue decision is deliberately NOT left to the model: it
reuses the same value-function comparison `CarrierRV` uses (accept now if
that's worth at least as much as the expected value of holding my CURRENT
position). Control-flow decisions (accept/reject) are more consequential
than a numeric offer that gets clamped anyway, so they stay grounded in the
well-tested shared value function rather than an untrusted prediction.

This has to compare against `my_last_offer` (the position I've already
stated), not the model's freshly-predicted `x` for this turn -- `CarrierOPT`
gets away with comparing against its own new `x` only because its
"continue" value is a P_accept-weighted BLEND that also credits x with a
chance of being accepted outright; a plain `carrier_value` on an
already-conceded `x`, with no such blend, systematically undervalues
continuing and triggers accepting far too early. (Found by literally
comparing CarrierML against CarrierOPT2 turn-by-turn on the same
negotiations: CarrierOPT2 negotiated for 12 turns to close at a strong
price; the first version of this file, using `x`, accepted the shipper's
very first counter-offer instead.)
"""
from __future__ import annotations

from pathlib import Path

from wehaveadeal._paths import baseline_artifact, load_sklearn_model
from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.ml.features import extract_features, features_to_vector
from wehaveadeal.models import Move, TableState
from wehaveadeal.value_functions import carrier_value, portfolio_occupancy

DEFAULT_MODEL_PATH = baseline_artifact("ml/carrier_ml_model.joblib")


class CarrierML(DecisionEngine):
    engine_name = "CarrierML"
    version = "v1"

    def __init__(
        self,
        agent_id: str,
        frozen: bool = True,
        model_path: str = DEFAULT_MODEL_PATH,
        firmness: float = 0.25,
        weight_occupancy: float = 0.5,
    ) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.model_path = model_path
        self.firmness = firmness
        self.weight_occupancy = weight_occupancy
        self._model = None  # lazily loaded on first decision, see _predict

    def _load_model(self):
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError(
                "CarrierML needs scikit-learn/joblib -- run `pip install -r requirements.txt`."
            ) from exc
        path = Path(self.model_path)
        if not path.exists():
            raise RuntimeError(
                f"No trained model at {path}. Run wehaveadeal/ml/collect.py then wehaveadeal/ml/train.py "
                "(--role carrier) first, or pass a different model_path."
            )
        return load_sklearn_model(path)

    def _predict(self, state: TableState) -> float:
        if self._model is None:
            self._model = self._load_model()
        vector = features_to_vector(extract_features(state))
        p = float(self._model.predict([vector])[0])
        return min(max(p, 0.0), 1.0)

    def next_move(self, state: TableState) -> Move:
        rp_t = state.refprice_mean - 0.5 * state.sigma
        target_t = state.refprice_mean + 0.5 * state.sigma
        tau = state.my_turns_used
        turns_per_side = state.turns_per_side
        turn_index = tau + 1
        shipper_offer = state.opp_last_offer

        if turn_index > turns_per_side:
            return REJECT

        if shipper_offer is not None and turn_index >= turns_per_side and shipper_offer >= rp_t:
            return ACCEPT

        rp_eff = rp_t + self.firmness * (target_t - rp_t)
        p_occupancy = portfolio_occupancy(state.my_portfolio, rp_eff)

        p = self._predict(state)
        x = rp_t + p * (target_t - rp_t)

        if shipper_offer is not None and shipper_offer >= rp_t:
            v_accept_now = carrier_value(
                rp_t=rp_t, target_t=target_t, terminal=True, close_price=shipper_offer,
                turn=state.turn, turns_max=state.turns_max,
                weight_occupancy=self.weight_occupancy, p_occupancy=p_occupancy,
            )
            v_continue = carrier_value(
                rp_t=rp_t, target_t=target_t, terminal=False,
                shipper_offer_on_table=shipper_offer, my_last_counter=state.my_last_offer,
                turn=state.turn, turns_max=state.turns_max,
                weight_occupancy=self.weight_occupancy, p_occupancy=p_occupancy,
            )
            if v_accept_now >= v_continue:
                return ACCEPT

        next_counter = min(max(x, rp_t), target_t)
        if state.my_last_offer is not None:
            next_counter = min(next_counter, state.my_last_offer)
        return make_offer(next_counter)
