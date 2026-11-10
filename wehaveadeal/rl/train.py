"""Train a negotiation policy with PPO (stable-baselines3) against
`wehaveadeal.rl.env.NegotiationEnv`, and save it for
`wehaveadeal.agents.carrier_rl.CarrierRL` / `wehaveadeal.agents.shipper_rl.ShipperRL`
to load at inference time.

Scope, stated up front so results are read correctly: this trains on
ISOLATED bilateral tables, one at a time, against a frozen opponent sampled
fresh each episode from the five hand-designed architectures (R/RV/IS/OPT/
OPT2) -- not a full 12-table round, and not self-play against a copy of
itself. `portfolio_promise` in the observation is therefore always its
neutral default (see `wehaveadeal/ml/features.py`); the policy has to learn
without ever seeing a real portfolio signal. That's a deliberate v1
simplification (a full-round multi-agent PPO setup is a substantially
bigger undertaking), not an oversight.

KNOWN CONSEQUENCE, confirmed by actually running the trained policy in a
real round (see the README's "Motor de refuerzo (RL)" section for the
numbers): evaluated in isolation the trained carrier policy is excellent
(>2x a random policy's mean reward, ~91% close rate) -- but it converges to
"anchor near target_t and never concede," which is a rational exploit of
having exactly one opponent and no rival carriers in training. Deployed
into a round alongside R/RV/OPT2/ML, it stops getting picked by
Gale-Shapley almost entirely, because it never offers the cheapest price to
a shipper who has other options. The training environment has no signal
for "a rival carrier undercut you" at all, so nothing here can currently
teach the policy that a competitive price -- not just an eventually-
acceptable one -- is what actually earns a match. Fixing this needs a
competition-aware environment (multiple learning carriers, or the full
round with matching in the loop), not a longer run of this same setup.

Usage:
    python -m wehaveadeal.rl.train --role carrier --timesteps 300000
    python -m wehaveadeal.rl.train --role shipper --timesteps 300000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = {
    "carrier": "rl/carrier_rl_model.zip",
    "shipper": "rl/shipper_rl_model.zip",
}


def _make_env(role: str, base_seed: int):
    from wehaveadeal.rl.env import NegotiationEnv

    def _init():
        return NegotiationEnv(role=role, base_seed=base_seed)

    return _init


def _opponent_kinds(role: str):
    from wehaveadeal.rl.env import OPPONENT_KINDS

    return OPPONENT_KINDS[role]


def evaluate(model, role: str, n_episodes: int, base_seed: int) -> dict:
    """Deterministic-policy rollout against the same opponent roster used
    in training, reporting mean reward and close rate -- the same two
    numbers `env.py`'s random-policy smoke test reports, so training
    progress is directly comparable to that baseline."""
    from wehaveadeal.rl.env import NegotiationEnv

    env = NegotiationEnv(role=role, base_seed=base_seed)
    rewards = []
    closes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        reward = 0.0
        steps = 0
        while not (terminated or truncated) and steps < 20:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1
        rewards.append(reward)
        if reward > 0:
            closes += 1
    rewards = np.array(rewards)
    return {
        "n_episodes": n_episodes,
        "mean_reward": float(rewards.mean()),
        "close_rate": closes / n_episodes,
    }


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    parser = argparse.ArgumentParser(description="Train a negotiation policy with PPO")
    parser.add_argument("--role", choices=["carrier", "shipper"], required=True)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=8, help="parallel environments for rollout collection")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256, help="rollout length per env before each PPO update")
    args = parser.parse_args()

    vec_env = make_vec_env(_make_env(args.role, args.base_seed), n_envs=args.n_envs, seed=args.base_seed)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        gamma=0.99,
        verbose=1,
        seed=args.base_seed,
        policy_kwargs={"net_arch": [64, 64]},
    )

    start = time.time()
    model.learn(total_timesteps=args.timesteps)
    elapsed = time.time() - start
    print(f"Trained {args.timesteps} timesteps in {elapsed:.1f}s")

    print("Evaluating trained policy (deterministic) ...")
    trained_stats = evaluate(model, args.role, args.eval_episodes, base_seed=args.base_seed + 999_000)

    print("Evaluating a random policy on the same opponent roster, for comparison ...")
    from wehaveadeal.rl.env import NegotiationEnv

    random_env = NegotiationEnv(role=args.role, base_seed=args.base_seed + 999_000)
    random_rewards = []
    for _ in range(args.eval_episodes):
        random_env.reset()
        terminated = truncated = False
        reward = 0.0
        steps = 0
        while not (terminated or truncated) and steps < 20:
            action = random_env.action_space.sample()
            obs, reward, terminated, truncated, _ = random_env.step(action)
            steps += 1
        random_rewards.append(reward)
    random_rewards = np.array(random_rewards)
    random_stats = {
        "n_episodes": args.eval_episodes,
        "mean_reward": float(random_rewards.mean()),
        "close_rate": float((random_rewards > 0).mean()),
    }

    print(f"trained : mean_reward={trained_stats['mean_reward']:.4f}  close_rate={trained_stats['close_rate']:.2%}")
    print(f"random  : mean_reward={random_stats['mean_reward']:.4f}  close_rate={random_stats['close_rate']:.2%}")

    out = Path(args.out) if args.out else Path(DEFAULT_MODEL_PATH[args.role])
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out.with_suffix("")))  # SB3 appends .zip itself

    meta = {
        "role": args.role,
        "algorithm": "PPO",
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "training_seconds": elapsed,
        "trained_eval": trained_stats,
        "random_baseline_eval": random_stats,
        "opponent_kinds": list(_opponent_kinds(args.role)),
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote model to {out}")
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
