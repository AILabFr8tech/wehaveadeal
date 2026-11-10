import numpy as np
import pytest

from wehaveadeal.rl.env import NegotiationEnv


@pytest.mark.parametrize("role", ["carrier", "shipper"])
def test_env_reset_and_step_shapes(role):
    env = NegotiationEnv(role=role, base_seed=1)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    assert obs2.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)


def test_opponent_state_is_valid_when_agent_is_carrier():
    """Regression test for a real bug: sale_price/goods_cost/refprice_est
    were only being set when self.role == 'shipper', so when the RL agent
    played the CARRIER, the shipper OPPONENT it faced got refprice_est=0.0
    -- producing nonsense negative offers instead of real prices. Every
    opponent shipper offer must land within a sane, strictly positive
    range relative to the route's own scale."""
    env = NegotiationEnv(role="carrier", base_seed=3)
    for _ in range(50):
        env.reset()
        # The opponent's opening move already happened inside reset()
        # (shipper always moves first) -- check it landed in a sane range.
        first_offer = env._runner.last_shipper_offer
        assert first_offer is not None
        assert first_offer > 0, f"shipper opponent opened with a nonsense offer: {first_offer}"
        assert first_offer < 5 * env._refprice_mean


@pytest.mark.parametrize("role", ["carrier", "shipper"])
def test_random_policy_closes_most_episodes(role):
    """A purely random policy should still close the large majority of
    episodes (the opponent roster is designed to close deals reasonably,
    and the legal band is wide), and never produce a runaway episode."""
    env = NegotiationEnv(role=role, base_seed=7)
    closes = 0
    n = 300
    for _ in range(n):
        env.reset()
        terminated = truncated = False
        steps = 0
        reward = 0.0
        while not (terminated or truncated) and steps < 20:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        assert steps < 20, "episode did not terminate within a sane number of steps"
        if reward > 0:
            closes += 1
    assert closes / n > 0.8


@pytest.mark.parametrize("role", ["carrier", "shipper"])
def test_reward_is_bounded_in_unit_interval(role):
    env = NegotiationEnv(role=role, base_seed=11)
    for _ in range(100):
        env.reset()
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 20:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert -1e-6 <= reward <= 1 + 1e-6
            steps += 1
