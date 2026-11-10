"""Train a negotiation policy with PPO (stable-baselines3) against
`wehaveadeal.rl.league_env.LeagueNegotiationEnv`, and save it for
`wehaveadeal.agents.carrier_rl2.CarrierRL2` / `wehaveadeal.agents.shipper_rl2.ShipperRL2`
to load at inference time.

This is the direct fix for the "KNOWN CONSEQUENCE" `wehaveadeal/rl/train.py`
documents: RL1 (`NegotiationEnv`) trains against one frozen opponent with no
rival on its own side, so "never concede" is a rational strategy in
training and a catastrophic one in a real competitive round. Here, every
episode embeds the learner's own table in a small competitive round
(`n_rivals` hand-designed agents on the SAME side, all courting the exact
same single counterpart, with `wehaveadeal.matching.gale_shapley` actually
deciding who the counterpart goes with) -- see `wehaveadeal/rl/league_env.py`'s
module docstring for the full mechanics and the reward's severity ordering
(win > closed-but-lost-to-a-rival > impasse, strictly, so the policy has an
actual gradient away from "never concede").

Usage:
    python -m wehaveadeal.rl.train_league --role carrier --timesteps 300000
    python -m wehaveadeal.rl.train_league --role shipper --timesteps 300000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = {
    "carrier": "rl/carrier_rl2_model.zip",
    "shipper": "rl/shipper_rl2_model.zip",
}


def _make_env(role: str, base_seed: int, min_rivals: int, max_rivals: int, reward_impasse: float, reward_closed_lost: float):
    from wehaveadeal.rl.league_env import LeagueNegotiationEnv

    def _init():
        return LeagueNegotiationEnv(
            role=role, base_seed=base_seed, min_rivals=min_rivals, max_rivals=max_rivals,
            reward_impasse=reward_impasse, reward_closed_lost=reward_closed_lost,
        )

    return _init


def evaluate(model, role: str, n_episodes: int, base_seed: int, min_rivals: int, max_rivals: int) -> dict:
    """Deterministic-policy rollout, reporting mean reward AND the three
    outcome rates separately (won / closed_lost / impasse) -- mean reward
    and a bare "close rate" (as `wehaveadeal/rl/train.py`'s evaluate() reports)
    would hide exactly the failure mode this environment exists to fix: a
    policy that closes constantly but never actually WINS the match."""
    from wehaveadeal.rl.league_env import LeagueNegotiationEnv

    env = LeagueNegotiationEnv(role=role, base_seed=base_seed, min_rivals=min_rivals, max_rivals=max_rivals)
    rewards = []
    outcomes = {"won": 0, "closed_lost": 0, "impasse": 0}
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        reward = 0.0
        info = {}
        steps = 0
        while not (terminated or truncated) and steps < 20:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        rewards.append(reward)
        outcomes[info["outcome"]] += 1
    rewards = np.array(rewards)
    return {
        "n_episodes": n_episodes,
        "mean_reward": float(rewards.mean()),
        "won_rate": outcomes["won"] / n_episodes,
        "closed_lost_rate": outcomes["closed_lost"] / n_episodes,
        "impasse_rate": outcomes["impasse"] / n_episodes,
    }


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    parser = argparse.ArgumentParser(description="Train a competition-aware negotiation policy with PPO")
    parser.add_argument("--role", choices=["carrier", "shipper"], required=True)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=8, help="parallel environments for rollout collection")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--min-rivals", type=int, default=1)
    parser.add_argument("--max-rivals", type=int, default=2)
    parser.add_argument("--reward-impasse", type=float, default=-0.5)
    parser.add_argument("--reward-closed-lost", type=float, default=-0.15)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256, help="rollout length per env before each PPO update")
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="load an existing model (e.g. a prior chunk's --out) and continue training it for --timesteps "
             "MORE steps, instead of initializing a fresh policy. Lets an unattended multi-chunk run survive "
             "a crash by resuming from the last successfully-saved chunk rather than restarting from scratch.",
    )
    args = parser.parse_args()

    vec_env = make_vec_env(
        _make_env(args.role, args.base_seed, args.min_rivals, args.max_rivals, args.reward_impasse, args.reward_closed_lost),
        n_envs=args.n_envs, seed=args.base_seed,
    )

    if args.resume_from:
        print(f"Resuming from {args.resume_from}")
        model = PPO.load(args.resume_from, env=vec_env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            gamma=0.99,
            verbose=1,
            seed=args.base_seed,
            policy_kwargs={"net_arch": [64, 64]},
            # A 64x64 MLP on this tiny observation/action space gains nothing from
            # GPU (kernel-launch overhead would dominate), and forcing CPU sidesteps
            # this machine's CUDA toolkit/driver instability entirely.
            device="cpu",
        )

    start = time.time()
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not args.resume_from)
    elapsed = time.time() - start
    print(f"Trained {args.timesteps} timesteps in {elapsed:.1f}s")

    print("Evaluating trained policy (deterministic) ...")
    trained_stats = evaluate(model, args.role, args.eval_episodes, args.base_seed + 999_000, args.min_rivals, args.max_rivals)

    print("Evaluating a random policy on the same league, for comparison ...")
    from wehaveadeal.rl.league_env import LeagueNegotiationEnv

    random_env = LeagueNegotiationEnv(
        role=args.role, base_seed=args.base_seed + 999_000, min_rivals=args.min_rivals, max_rivals=args.max_rivals,
        reward_impasse=args.reward_impasse, reward_closed_lost=args.reward_closed_lost,
    )
    random_rewards = []
    random_outcomes = {"won": 0, "closed_lost": 0, "impasse": 0}
    for _ in range(args.eval_episodes):
        random_env.reset()
        terminated = truncated = False
        reward = 0.0
        info = {}
        steps = 0
        while not (terminated or truncated) and steps < 20:
            action = random_env.action_space.sample()
            obs, reward, terminated, truncated, info = random_env.step(action)
            steps += 1
        random_rewards.append(reward)
        random_outcomes[info["outcome"]] += 1
    random_rewards = np.array(random_rewards)
    random_stats = {
        "n_episodes": args.eval_episodes,
        "mean_reward": float(random_rewards.mean()),
        "won_rate": random_outcomes["won"] / args.eval_episodes,
        "closed_lost_rate": random_outcomes["closed_lost"] / args.eval_episodes,
        "impasse_rate": random_outcomes["impasse"] / args.eval_episodes,
    }

    print(
        f"trained : mean_reward={trained_stats['mean_reward']:.4f}  "
        f"won={trained_stats['won_rate']:.2%}  closed_lost={trained_stats['closed_lost_rate']:.2%}  "
        f"impasse={trained_stats['impasse_rate']:.2%}"
    )
    print(
        f"random  : mean_reward={random_stats['mean_reward']:.4f}  "
        f"won={random_stats['won_rate']:.2%}  closed_lost={random_stats['closed_lost_rate']:.2%}  "
        f"impasse={random_stats['impasse_rate']:.2%}"
    )

    out = Path(args.out) if args.out else Path(DEFAULT_MODEL_PATH[args.role])
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out.with_suffix("")))  # SB3 appends .zip itself

    meta = {
        "role": args.role,
        "algorithm": "PPO",
        "env": "LeagueNegotiationEnv",
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "training_seconds": elapsed,
        "min_rivals": args.min_rivals,
        "max_rivals": args.max_rivals,
        "reward_impasse": args.reward_impasse,
        "reward_closed_lost": args.reward_closed_lost,
        "trained_eval": trained_stats,
        "random_baseline_eval": random_stats,
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote model to {out}")
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
