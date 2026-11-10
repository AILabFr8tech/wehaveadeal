"""LinRegCarrier -- a carrier driven by a trained linear regression.

The model (trained by linreg_carrier_model/train.py) answers one question:
"given what I can see right now, at what rate will this table close?"
The agent asks a little above that prediction early on, concedes towards it
over time, and accepts as soon as the shipper offers at least the prediction.

This file is also the reference pattern for any agent that ships a model
file (.joblib/.pkl, .h5/.keras, .pt, .onnx, ...):

  1. Keep the weights in a folder next to the agent file.
  2. Locate them relative to __file__, never with an absolute path.
  3. Load them in __init__, not in next_move: the referee times next_move
     only, so a slow load on the first move could forfeit tables.
  4. Cache the loaded model at module level: the dashboard creates a new
     agent for every negotiation it shows, so it then loads only once.

Train your own copy with:  python examples/linreg_carrier_model/train.py
"""
from pathlib import Path

import joblib

from wehaveadeal import ACCEPT, REJECT, DecisionEngine, Move, TableState, legal_offer_range, offer

MODEL_PATH = Path(__file__).parent / "linreg_carrier_model" / "model.joblib"
FEATURE_NAMES = ("turn_frac", "opp_offer_rel", "has_my_offer", "my_offer_rel")
_MODEL_CACHE = {}


def features(state: TableState) -> list:
    """What a carrier can see, with prices relative to the market mean so one
    model works on every route. Shared by the agent and by train.py.

    (The shipper always opens, so a carrier normally has an offer in front of
    it; the market mean stands in if not. Route sigma is left out: it is a
    fixed fraction of the mean, so relative to the mean it never varies.)"""
    mean = state.refprice_mean
    has_my = state.my_last_offer is not None
    return [
        state.turn / state.turns_max,
        state.opp_last_offer / mean if state.opp_last_offer is not None else 1.0,
        1.0 if has_my else 0.0,
        state.my_last_offer / mean if has_my else 0.0,
    ]


def load_model(path: Path = MODEL_PATH):
    if path not in _MODEL_CACHE:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `python examples/linreg_carrier_model/train.py` first")
        _MODEL_CACHE[path] = joblib.load(path)
    return _MODEL_CACHE[path]


class LinRegCarrier(DecisionEngine):
    role = "carrier"
    engine_name = "LinRegCarrier"
    version = "v1"

    def __init__(self, agent_id: str, frozen: bool = True, markup: float = 0.05) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.model = load_model()
        self.markup = markup  # how far above the predicted closing rate to open

    def next_move(self, state: TableState) -> Move:
        mean = state.refprice_mean
        predicted = mean * float(self.model.predict([features(state)])[0])
        floor = mean - 0.5 * state.sigma  # never aim below the carrier's hard floor
        target = max(predicted, floor)

        if state.opp_last_offer is not None and state.opp_last_offer >= target:
            return ACCEPT

        progress = min(1.0, state.my_turns_used / max(1, state.turns_per_side - 1))
        ask = target * (1 + self.markup * (1 - progress))
        lo, hi = legal_offer_range(state)
        if lo > hi:
            return REJECT
        return offer(min(max(ask, lo), hi))
