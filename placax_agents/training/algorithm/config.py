"""Bundles PPO's tunable hyperparameters - gamma/lam belong to GAE,
clip_eps/value_coef/entropy_coef belong to the loss, but both are
genuinely "the PPO algorithm's knobs" and get threaded through train()
together. A plain dataclass, not a flax struct: these are ordinary
Python floats that don't affect array shapes, so they're passed as a
static jit argument (like optimizer/policy_apply_fn already are), not
traced.

Found by auditing whether these were actually reachable from train():
compute_gae() and ppo_loss() were already correctly parameterized with
real defaults, but nothing calling them ever passed overrides through -
gamma/lam/clip_eps/value_coef/entropy_coef were silently stuck at
whatever the inner defaults happened to be, with no way to tune them
from the public train() API at all."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
