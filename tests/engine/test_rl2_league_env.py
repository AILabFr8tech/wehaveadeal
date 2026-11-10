import pytest

from wehaveadeal.rl.league_env import LeagueNegotiationEnv


@pytest.mark.parametrize("role", ["carrier", "shipper"])
def test_env_reset_and_step_shapes(role):
    env = LeagueNegotiationEnv(role=role, base_seed=1)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    assert obs2.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)


@pytest.mark.parametrize("role", ["carrier", "shipper"])
def test_random_policy_episodes_terminate_with_a_valid_outcome(role):
    env = LeagueNegotiationEnv(role=role, base_seed=7)
    outcomes = {"won": 0, "closed_lost": 0, "impasse": 0}
    n = 200
    for _ in range(n):
        env.reset()
        terminated = truncated = False
        steps = 0
        info = {}
        while not (terminated or truncated) and steps < 20:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        assert steps < 20, "episode did not terminate within a sane number of steps"
        outcomes[info["outcome"]] += 1
    assert sum(outcomes.values()) == n


@pytest.mark.parametrize("role", ["carrier", "shipper"])
def test_reward_matches_outcome_severity_ordering(role):
    """The whole point of this environment: closing but losing to a rival
    must be a real, distinguishable penalty (strictly worse than the worst
    possible win, i.e. < 0), and impasse must be strictly worse than
    closed-but-lost -- otherwise a policy has no gradient pushing it away
    from "never concede" (the exact failure `NegotiationEnv` documented)."""
    env = LeagueNegotiationEnv(role=role, base_seed=11)
    assert env.reward_impasse < env.reward_closed_lost < 0.0

    for _ in range(150):
        env.reset()
        terminated = truncated = False
        steps = 0
        reward = 0.0
        info = {}
        while not (terminated or truncated) and steps < 20:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        if info["outcome"] == "impasse":
            assert reward == env.reward_impasse
        elif info["outcome"] == "closed_lost":
            assert reward == env.reward_closed_lost
        else:
            assert -1e-6 <= reward <= 1 + 1e-6


@pytest.mark.parametrize("role", ["carrier", "shipper"])
def test_random_policy_wins_less_often_than_it_closes(role):
    """A random policy should still close most of its tables (the legal
    band is wide) but should NOT win most of them -- some rival almost
    always undercuts a random price. If "won" ever matched "closed" (i.e.
    closed_lost never happens), the competitive pressure this environment
    exists to add would be silently missing."""
    env = LeagueNegotiationEnv(role=role, base_seed=23, min_rivals=2, max_rivals=2)
    outcomes = {"won": 0, "closed_lost": 0, "impasse": 0}
    n = 200
    for _ in range(n):
        env.reset()
        terminated = truncated = False
        steps = 0
        info = {}
        while not (terminated or truncated) and steps < 20:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        outcomes[info["outcome"]] += 1
    closed = outcomes["won"] + outcomes["closed_lost"]
    assert closed > 0.5 * n
    assert outcomes["closed_lost"] > 0, "a random policy never lost to a rival -- competitive pressure isn't reaching the reward"


def test_counterpart_and_learner_both_see_real_portfolio_pressure_across_rival_tables():
    """Both legitimate 'sensing competition' signals in this design: the
    shared counterpart must see its OTHER tables (rivals' tables, in
    addition to mine) via my_portfolio, not an empty/neutral default --
    otherwise it behaves as if it only ever talks to one carrier, defeating
    the point of this environment -- and, since the fix traced in
    `wehaveadeal/agents/carrier_rl2.py`'s module docstring (a policy trained with
    my_portfolio pinned at [] is then deployed where it genuinely negotiates
    several tables at once, a real train/deploy feature mismatch), the
    learning agent's OWN table gets that same real visibility too, so the
    policy trains on exactly the distribution it will see deployed."""
    env = LeagueNegotiationEnv(role="carrier", base_seed=5, min_rivals=2, max_rivals=2)
    env.reset()

    state = env._build_state(env._my_runner, env.opp_role, global_turn=1)
    assert len(state.my_portfolio) == 2  # the two rival tables, excluding "mine"

    # The learning agent itself now gets the same real visibility across its
    # own sibling (rival) tables, excluding itself -- matching what it will
    # actually receive at deployment via wehaveadeal.simulate._portfolio_snapshot.
    my_turn_state = env._build_state(env._my_runner, env.role, global_turn=2)
    assert len(my_turn_state.my_portfolio) == 2


def test_rival_kinds_never_include_a_learned_kind():
    from wehaveadeal.rl.league_env import RIVAL_KINDS

    for role in ("carrier", "shipper"):
        for kind in RIVAL_KINDS[role]:
            assert kind not in ("ml", "ml2", "rl", "rl2")
