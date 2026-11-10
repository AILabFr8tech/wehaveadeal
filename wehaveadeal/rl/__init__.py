"""Reinforcement-learning pipeline for We Have a Deal: `env.py` (one
bilateral table as a Gymnasium environment) and `train.py` (PPO training
CLI via stable-baselines3). Produces the same kind of artifact
`wehaveadeal/ml/train.py` does for imitation learning -- a model file loaded by
`wehaveadeal.agents.carrier_rl.CarrierRL` / `wehaveadeal.agents.shipper_rl.ShipperRL`.
"""
