import numpy as np
import pytest

from wehaveadeal.agents import CarrierRL2, ShipperRL2
from wehaveadeal.market import MarketState
from wehaveadeal.models import Action, PortfolioTableSnapshot, TableState


class _ConstantPolicy:
    """Stand-in for a trained PPO model: always predicts a fixed action,
    matching stable_baselines3's `model.predict(obs, deterministic=True) ->
    (action, state)` interface, so engine tests don't depend on a real
    trained artifact (slow to construct, and not what these tests check)."""

    def __init__(self, p: float):
        self.p = p

    def predict(self, obs, deterministic=True):
        return np.array([self.p], dtype=np.float32), None


def _sample_carrier_state(seed=7, turn=4, my_last_offer=None, opp_last_offer=None):
    market = MarketState(seed=seed, n_negotiations=1)
    route = market.assign_route(market.negotiation_rng(0))
    return TableState(
        role="carrier", table_id="t", negotiation_seed=seed, turn=turn, turns_max=12,
        my_turns_used=turn // 2 - 1 if turn > 1 else 0, opp_turns_used=turn // 2,
        my_last_offer=my_last_offer, opp_last_offer=opp_last_offer,
        sigma=route.refprice_sigma, refprice_mean=route.refprice_mean,
    )


def _sample_shipper_state(seed=11, turn=1, my_last_offer=None, opp_last_offer=None):
    market = MarketState(seed=seed, n_negotiations=1)
    route = market.assign_route(market.negotiation_rng(0))
    return TableState(
        role="shipper", table_id="t", negotiation_seed=seed, turn=turn, turns_max=12,
        my_turns_used=turn // 2, opp_turns_used=turn // 2,
        my_last_offer=my_last_offer, opp_last_offer=opp_last_offer,
        sigma=route.refprice_sigma,
        sale_price=route.refprice_mean + 0.6 * route.refprice_sigma, goods_cost=0.0,
        refprice_est=route.refprice_mean,
    )


@pytest.mark.parametrize("p", [0.0, 0.3, 0.7, 1.0])
def test_carrier_rl2_stays_legal_and_monotonic(p):
    agent = CarrierRL2(agent_id="carrier-rl2-1", model_path="unused")
    agent._model = _ConstantPolicy(p)

    state = _sample_carrier_state(turn=4, my_last_offer=None, opp_last_offer=None)
    move1 = agent.next_move(state)
    assert move1.action == Action.OFFER
    rp_t = state.refprice_mean - 0.5 * state.sigma
    target_t = state.refprice_mean + 0.5 * state.sigma
    assert rp_t - 1e-6 <= move1.amount <= target_t + 1e-6

    state2 = _sample_carrier_state(turn=6, my_last_offer=move1.amount, opp_last_offer=rp_t + 1.0)
    agent2 = CarrierRL2(agent_id="carrier-rl2-1", model_path="unused")
    agent2._model = _ConstantPolicy(p)
    move2 = agent2.next_move(state2)
    if move2.action == Action.OFFER:
        assert move2.amount <= move1.amount + 1e-6


@pytest.mark.parametrize("p", [0.0, 0.3, 0.7, 1.0])
def test_shipper_rl2_stays_legal_and_monotonic(p):
    agent = ShipperRL2(agent_id="shipper-rl2-1", model_path="unused")
    agent._model = _ConstantPolicy(p)

    state = _sample_shipper_state(turn=1, my_last_offer=None, opp_last_offer=None)
    move1 = agent.next_move(state)
    assert move1.action == Action.OFFER
    rp = state.sale_price - state.goods_cost
    assert move1.amount <= rp + 1e-6

    state2 = _sample_shipper_state(turn=3, my_last_offer=move1.amount, opp_last_offer=rp - 1.0)
    agent2 = ShipperRL2(agent_id="shipper-rl2-1", model_path="unused")
    agent2._model = _ConstantPolicy(p)
    move2 = agent2.next_move(state2)
    if move2.action == Action.OFFER:
        assert move2.amount >= move1.amount - 1e-6


def test_carrier_rl2_accepts_when_offer_meets_its_own_aspiration():
    agent = CarrierRL2(agent_id="carrier-rl2-1", model_path="unused")
    agent._model = _ConstantPolicy(0.2)  # modest aspiration, close to the floor

    state = _sample_carrier_state(turn=4, my_last_offer=2500.0, opp_last_offer=None)
    rp_t = state.refprice_mean - 0.5 * state.sigma
    target_t = state.refprice_mean + 0.5 * state.sigma
    generous_offer = rp_t + 0.9 * (target_t - rp_t)  # well above a p=0.2 aspiration
    state = _sample_carrier_state(turn=4, my_last_offer=2500.0, opp_last_offer=generous_offer)
    move = agent.next_move(state)
    assert move.action == Action.ACCEPT


def test_carrier_rl2_raises_a_clear_error_without_a_trained_model(tmp_path):
    agent = CarrierRL2(agent_id="carrier-rl2-1", model_path=str(tmp_path / "missing.zip"))
    state = _sample_carrier_state()
    with pytest.raises(RuntimeError, match="No trained model"):
        agent.next_move(state)


def test_shipper_rl2_raises_a_clear_error_without_a_trained_model(tmp_path):
    agent = ShipperRL2(agent_id="shipper-rl2-1", model_path=str(tmp_path / "missing.zip"))
    state = _sample_shipper_state()
    with pytest.raises(RuntimeError, match="No trained model"):
        agent.next_move(state)


def _fake_portfolio(n: int):
    return [
        PortfolioTableSnapshot(counterpart_id=f"c{i}", is_closed=False, is_impasse=False, closed_rate=None, latest_offer_on_table=None)
        for i in range(n)
    ]


class _RecordingPolicy:
    """Records the observation vector it was called with, so a test can
    check what my_portfolio-derived feature actually reached the model."""

    def __init__(self, p: float):
        self.p = p
        self.last_obs = None

    def predict(self, obs, deterministic=True):
        self.last_obs = obs
        return np.array([self.p], dtype=np.float32), None


def test_carrier_rl2_neutralizes_real_portfolio_before_predicting():
    """Regression test for a real bug, confirmed by tracing a live round:
    CarrierRL2/ShipperRL2 were trained with portfolio_promise pinned at its
    neutral 0.5 default (LeagueNegotiationEnv never gives the LEARNING
    agent real siblings, see that module's docstring), so feeding the
    model a real, non-neutral portfolio at inference produced a wildly
    miscalibrated aspiration (confirmed: ~94% of the shipper's own
    worst-case ceiling on turn 1, accepted instantly by every carrier).
    The fix must discard `state.my_portfolio` before predicting -- this
    checks the observation actually sent to the model has the neutral
    portfolio_promise feature (index of "portfolio_promise" in
    FEATURE_NAMES) regardless of what state.my_portfolio contained."""
    from wehaveadeal.ml.features import FEATURE_NAMES

    idx = FEATURE_NAMES.index("portfolio_promise")

    agent = CarrierRL2(agent_id="carrier-rl2-1", model_path="unused")
    policy = _RecordingPolicy(0.5)
    agent._model = policy

    state = _sample_carrier_state(turn=4, my_last_offer=None, opp_last_offer=None)
    state = TableState(**{**state.model_dump(), "my_portfolio": _fake_portfolio(4)})
    agent.next_move(state)

    assert policy.last_obs[idx] == pytest.approx(0.5), "a non-empty my_portfolio leaked into the model's input"


def test_shipper_rl2_neutralizes_real_portfolio_before_predicting():
    from wehaveadeal.ml.features import FEATURE_NAMES

    idx = FEATURE_NAMES.index("portfolio_promise")

    agent = ShipperRL2(agent_id="shipper-rl2-1", model_path="unused")
    policy = _RecordingPolicy(0.5)
    agent._model = policy

    state = _sample_shipper_state(turn=1, my_last_offer=None, opp_last_offer=None)
    state = TableState(**{**state.model_dump(), "my_portfolio": _fake_portfolio(3)})
    agent.next_move(state)

    assert policy.last_obs[idx] == pytest.approx(0.5), "a non-empty my_portfolio leaked into the model's input"
