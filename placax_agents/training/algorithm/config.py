"""PPO's tunable hyperparameters, threaded together through the training
loops. A plain frozen dataclass: ordinary floats that don't affect array
shapes, passed as a static jit argument."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99  # GAE discount
    lam: float = 0.95  # GAE smoothing
    clip_eps: float = 0.2  # PPO ratio clip
    value_coef: float = 0.5  # value-loss weight
    entropy_coef: float = 0.01  # entropy-bonus weight
