import numpy as np
import pytest

from wehaveadeal.agents import CarrierML, ShipperML
from wehaveadeal.ml.collect import collect_rows
from wehaveadeal.ml.features import FEATURE_NAMES, extract_features, features_to_vector, role_band
from wehaveadeal.market import MarketState
from wehaveadeal.models import Action, TableState


def _sample_carrier_state(seed=7, turn=4, my_last_offer=None, opp_last_offer=None, history=None):
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
        history=history or [],
        sigma=route.refprice_sigma,
        refprice_mean=route.refprice_mean,
    )


def test_extract_features_is_deterministic_and_complete():
    state = _sample_carrier_state(opp_last_offer=1500.0, my_last_offer=1800.0)
    f1 = extract_features(state)
    f2 = extract_features(state)
    assert f1 == f2
    assert set(f1.keys()) == set(FEATURE_NAMES)
    vec = features_to_vector(f1)
    assert len(vec) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in vec)


def test_extract_features_handles_no_history_yet():
    state = _sample_carrier_state(turn=2, my_last_offer=None, opp_last_offer=1500.0)
    feats = extract_features(state)
    assert feats["my_last_seen"] == 0.0
    assert feats["opp_last_seen"] == 1.0
    assert feats["opp_trend_seen"] == 0.0  # no OFFER history recorded yet, just opp_last_offer on the table


def test_role_band_matches_carrier_hard_band():
    state = _sample_carrier_state()
    worst, best = role_band(state)
    assert worst == state.refprice_mean - 0.5 * state.sigma
    assert best == state.refprice_mean + 0.5 * state.sigma


class _ConstantModel:
    """Stand-in for a trained sklearn regressor: always predicts a fixed
    band position, so tests don't depend on a real trained artifact."""

    def __init__(self, p: float):
        self.p = p

    def predict(self, X):
        return np.full(len(X), self.p)


def test_carrier_ml_accept_decision_anchors_on_my_last_offer_not_prediction(tmp_path):
    """Regression test for a real bug: the accept-vs-continue check must
    compare against `state.my_last_offer` (the position already on the
    table), not the model's freshly predicted `x` for this turn. Using `x`
    made CarrierML accept far too early (confirmed by tracing it against
    CarrierOPT2 turn-by-turn on identical negotiations -- CarrierOPT2 held
    out for 12 turns to close well; CarrierML using `x` accepted the
    shipper's very first counter instead).

    Here the model always predicts p=0.0 (x = RP_t, the worst possible
    counter), while `my_last_offer` is already sitting near `target_t` (a
    strong position). A mediocre-but-legal shipper offer near RP_t should
    NOT be accepted while a strong offer is already on the table -- that
    would only happen if the code wrongly anchored on the model's `x`
    instead of the real `my_last_offer`.
    """
    import joblib

    model_path = tmp_path / "carrier_worst_case.joblib"
    joblib.dump(_ConstantModel(0.0), model_path)
    agent = CarrierML(agent_id="carrier-ml-1", model_path=str(model_path))

    market = MarketState(seed=7, n_negotiations=1)
    route = market.assign_route(market.negotiation_rng(0))
    rp_t = route.refprice_mean - 0.5 * route.refprice_sigma
    target_t = route.refprice_mean + 0.5 * route.refprice_sigma

    state = TableState(
        role="carrier",
        table_id="t",
        negotiation_seed=7,
        turn=4,
        turns_max=12,
        my_turns_used=1,
        opp_turns_used=2,
        my_last_offer=target_t - 0.01 * route.refprice_sigma,  # strong position already on the table
        opp_last_offer=rp_t + 0.05 * route.refprice_sigma,  # legal, but a mediocre offer
        sigma=route.refprice_sigma,
        refprice_mean=route.refprice_mean,
    )
    move = agent.next_move(state)
    assert move.action == Action.OFFER, "should hold out for a better price, not accept a mediocre offer"


def test_carrier_ml_stays_legal_and_monotonic_with_any_prediction(tmp_path, monkeypatch):
    import joblib

    for p in (0.0, 0.3, 0.7, 1.0):
        model_path = tmp_path / f"carrier_{p}.joblib"
        joblib.dump(_ConstantModel(p), model_path)
        agent = CarrierML(agent_id="carrier-ml-1", model_path=str(model_path))

        state = _sample_carrier_state(turn=4, my_last_offer=None, opp_last_offer=None)
        move1 = agent.next_move(state)
        assert move1.action == Action.OFFER
        rp_t = state.refprice_mean - 0.5 * state.sigma
        target_t = state.refprice_mean + 0.5 * state.sigma
        assert rp_t - 1e-6 <= move1.amount <= target_t + 1e-6

        # A second, later turn must not increase the counter beyond the first.
        state2 = _sample_carrier_state(turn=6, my_last_offer=move1.amount, opp_last_offer=rp_t + 1.0)
        move2 = agent.next_move(state2)
        if move2.action == Action.OFFER:
            assert move2.amount <= move1.amount + 1e-6


def test_shipper_ml_stays_legal_and_monotonic_with_any_prediction(tmp_path):
    import joblib

    market = MarketState(seed=11, n_negotiations=1)
    route = market.assign_route(market.negotiation_rng(0))

    for p in (0.0, 0.3, 0.7, 1.0):
        model_path = tmp_path / f"shipper_{p}.joblib"
        joblib.dump(_ConstantModel(p), model_path)
        agent = ShipperML(agent_id="shipper-ml-1", model_path=str(model_path))

        state = TableState(
            role="shipper",
            table_id="t",
            negotiation_seed=11,
            turn=1,
            turns_max=12,
            my_turns_used=0,
            opp_turns_used=0,
            my_last_offer=None,
            opp_last_offer=None,
            sigma=route.refprice_sigma,
            sale_price=route.refprice_mean + 0.6 * route.refprice_sigma,
            goods_cost=0.0,
            refprice_est=route.refprice_mean,
        )
        move1 = agent.next_move(state)
        assert move1.action == Action.OFFER
        rp = state.sale_price - state.goods_cost
        assert move1.amount <= rp + 1e-6

        state2 = TableState(**{**state.model_dump(), "turn": 3, "my_last_offer": move1.amount, "opp_last_offer": rp - 1.0})
        move2 = agent.next_move(state2)
        if move2.action == Action.OFFER:
            assert move2.amount >= move1.amount - 1e-6


def test_carrier_ml_raises_a_clear_error_without_a_trained_model(tmp_path):
    agent = CarrierML(agent_id="carrier-ml-1", model_path=str(tmp_path / "missing.joblib"))
    state = _sample_carrier_state()
    with pytest.raises(RuntimeError, match="No trained model"):
        agent.next_move(state)


def test_collect_rows_smoke():
    """collect_rows against a small self-play sample should produce
    well-formed rows with labels in [0, 1] and the teacher/role tags set."""
    rows = collect_rows(
        teacher_kind="opt2",
        role="carrier",
        seeds=[101, 102],
        opponent_kinds=["r", "rv", "is", "opt", "opt2"],
        n_negotiations=5,
        n_teacher_agents=2,
    )
    assert len(rows) > 0
    for row in rows[:20]:
        assert row["role"] == "carrier"
        assert row["teacher_kind"] == "opt2"
        assert -1e-6 <= row["label"] <= 1 + 1e-6
        for name in FEATURE_NAMES:
            assert name in row
