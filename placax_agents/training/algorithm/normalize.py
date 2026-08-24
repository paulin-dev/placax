"""Advantage normalization - near-universal in real PPO implementations.
Doesn't fix reward *scale* (that's running_stats.py's job), just keeps
the policy loss's gradient magnitude sane regardless of scale."""
import jax
import jax.numpy as jnp


def normalize_advantages(advantages: jax.Array, eps: float = 1e-8) -> jax.Array:
    return (advantages - advantages.mean()) / (advantages.std() + eps)
