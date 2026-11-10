import numpy as np
import pytest

from wehaveadeal.agents import CarrierML2, ShipperML2
from wehaveadeal.market import MarketState
from wehaveadeal.models import Action, TableState


class _ConstantRegressor:
    """Stand-in for the offer regressor: always predicts a fixed band
    position, so tests don't depend on a real trained artifact."""

    def __init__(self, p: float):
        self.p = p

    def predict(self, X):
        return np.full(len(X), self.p)


class _ConstantClassifier:
    """Stand-in for the accept classifier: always predicts a fixed
    P(accept), via predict_proba like a real sklearn classifier."""

    def __init__(self, p_accept: float):
        self.p_accept = p_accept

    def predict_proba(self, X):
        return np.tile([1 - self.p_accept, self.p_accept], (len(X), 1))


def _dump(tmp_path, name, obj):
    import joblib

    path = tmp_path / name
    joblib.dump(obj, path)
    return str(path)


def _sample_carrier_state(seed=7, turn=4, my_last_offer=None, opp_last_offer=None):
    market = MarketState(seed=seed, n_negotiations=1)
    route = market.assign_route(market.negotiation_rng(0))
    return TableState(
        role="carrier",
        table_id="t",
        negotiation_seed=seed,
        turn=turn,
        turns_max=12,
        my_turns_used=turn // 2 - 1 if turn > 1 else 0,
        opp_turns_used=turn // 2,
        my_last_offer=my_last_offer,
        opp_last_offer=opp_last_offer,
        sigma=route.refprice_sigma,
        refprice_mean=route.refprice_mean,
    )


def _sample_shipper_state(seed=11, turn=1, my_last_offer=None, opp_last_offer=None):
    market = MarketState(seed=seed, n_negotiations=1)
    route = market.assign_route(market.negotiation_rng(0))
    return TableState(
        role="shipper",
        table_id="t",
        negotiation_seed=seed,
        turn=turn,
        turns_max=12,
        my_turns_used=turn // 2 if turn > 1 else 0,
        opp_turns_used=turn // 2,
        my_last_offer=my_last_offer,
        opp_last_offer=opp_last_offer,
        sigma=route.refprice_sigma,
        sale_price=route.refprice_mean + 0.6 * route.refprice_sigma,
        goods_cost=0.0,
        refprice_est=route.refprice_mean,
    )


def test_carrier_ml2_stays_legal_and_monotonic_with_any_prediction(tmp_path):
    accept_path = _dump(tmp_path, "carrier_accept_never.joblib", _ConstantClassifier(0.0))
    for p in (0.0, 0.3, 0.7, 1.0):
        offer_path = _dump(tmp_path, f"carrier_offer_{p}.joblib", _ConstantRegressor(p))
        agent = CarrierML2(agent_id="carrier-ml2-1", offer_model_path=offer_path, accept_model_path=accept_path)

        state = _sample_carrier_state(turn=4, my_last_offer=None, opp_last_offer=None)
        move1 = agent.next_move(state)
        assert move1.action == Action.OFFER
        rp_t = state.refprice_mean - 0.5 * state.sigma
        target_t = state.refprice_mean + 0.5 * state.sigma
        assert rp_t - 1e-6 <= move1.amount <= target_t + 1e-6

        state2 = _sample_carrier_state(turn=6, my_last_offer=move1.amount, opp_last_offer=rp_t + 1.0)
        move2 = agent.next_move(state2)
        if move2.action == Action.OFFER:
            assert move2.amount <= move1.amount + 1e-6


def test_shipper_ml2_stays_legal_and_monotonic_with_any_prediction(tmp_path):
    accept_path = _dump(tmp_path, "shipper_accept_never.joblib", _ConstantClassifier(0.0))
    for p in (0.0, 0.3, 0.7, 1.0):
        offer_path = _dump(tmp_path, f"shipper_offer_{p}.joblib", _ConstantRegressor(p))
        agent = ShipperML2(agent_id="shipper-ml2-1", offer_model_path=offer_path, accept_model_path=accept_path)

        state = _sample_shipper_state(turn=1, my_last_offer=None, opp_last_offer=None)
        move1 = agent.next_move(state)
        assert move1.action == Action.OFFER
        rp = state.sale_price - state.goods_cost
        assert move1.amount <= rp + 1e-6

        state2 = TableState(**{**state.model_dump(), "turn": 3, "my_last_offer": move1.amount, "opp_last_offer": rp - 1.0})
        move2 = agent.next_move(state2)
        if move2.action == Action.OFFER:
            assert move2.amount >= move1.amount - 1e-6


def test_carrier_ml2_accept_classifier_gates_the_decision(tmp_path):
    """When the offer regressor would predict the worst possible counter
    (p=0.0, i.e. it would offer right at RP_t) but the classifier says
    P(accept) is high for a legal shipper offer, ML2 should accept -- and
    conversely, a classifier that never wants to accept should keep
    countering even on a strong shipper offer, deferring entirely to the
    classifier rather than any hardcoded comparison (the thing ML1 could
    not do, see carrier_ml.py's module docstring)."""
    offer_path = _dump(tmp_path, "carrier_offer_worst.joblib", _ConstantRegressor(0.0))

    market = MarketState(seed=7, n_negotiations=1)
    route = market.assign_route(market.negotiation_rng(0))
    rp_t = route.refprice_mean - 0.5 * route.refprice_sigma
    state = TableState(
        role="carrier",
        table_id="t",
        negotiation_seed=7,
        turn=4,
        turns_max=12,
        my_turns_used=1,
        opp_turns_used=2,
        my_last_offer=route.refprice_mean,
        opp_last_offer=rp_t + 0.05 * route.refprice_sigma,  # legal, mediocre shipper offer
        sigma=route.refprice_sigma,
        refprice_mean=route.refprice_mean,
    )

    accept_always = _dump(tmp_path, "carrier_accept_always.joblib", _ConstantClassifier(1.0))
    agent_accepts = CarrierML2(agent_id="carrier-ml2-1", offer_model_path=offer_path, accept_model_path=accept_always)
    assert agent_accepts.next_move(state).action == Action.ACCEPT

    accept_never = _dump(tmp_path, "carrier_accept_never.joblib", _ConstantClassifier(0.0))
    agent_continues = CarrierML2(agent_id="carrier-ml2-2", offer_model_path=offer_path, accept_model_path=accept_never)
    assert agent_continues.next_move(state).action == Action.OFFER


def test_carrier_ml2_raises_a_clear_error_without_trained_models(tmp_path):
    accept_path = _dump(tmp_path, "carrier_accept.joblib", _ConstantClassifier(0.0))
    agent = CarrierML2(
        agent_id="carrier-ml2-1",
        offer_model_path=str(tmp_path / "missing_offer.joblib"),
        accept_model_path=accept_path,
    )
    state = _sample_carrier_state()
    with pytest.raises(RuntimeError, match="No trained offer model"):
        agent.next_move(state)


def test_shipper_ml2_raises_a_clear_error_without_trained_models(tmp_path):
    offer_path = _dump(tmp_path, "shipper_offer.joblib", _ConstantRegressor(0.5))
    agent = ShipperML2(
        agent_id="shipper-ml2-1",
        offer_model_path=offer_path,
        accept_model_path=str(tmp_path / "missing_accept.joblib"),
    )
    state = _sample_shipper_state(opp_last_offer=1.0)
    state = TableState(**{**state.model_dump(), "opp_last_offer": state.sale_price - state.goods_cost - 1.0})
    with pytest.raises(RuntimeError, match="No trained accept model"):
        agent.next_move(state)


def test_collect_rows_records_accepted_and_accept_eligible_columns():
    from wehaveadeal.ml.collect import collect_rows
    from wehaveadeal.ml.features import FEATURE_NAMES

    rows = collect_rows(
        teacher_kind="opt2",
        role="carrier",
        seeds=[201, 202],
        opponent_kinds=["r", "rv", "is", "opt", "opt2"],
        n_negotiations=5,
        n_teacher_agents=2,
    )
    assert len(rows) > 0
    saw_accept = False
    for row in rows:
        assert row["accepted"] in (0, 1)
        assert row["accept_eligible"] in (0, 1)
        for name in FEATURE_NAMES:
            assert name in row
        if row["accepted"] == 1:
            saw_accept = True
            assert row["accept_eligible"] == 1  # an ACCEPT row is definitionally eligible
    assert saw_accept, "opt2 vs a full opponent roster over 2 seeds should close at least once"
