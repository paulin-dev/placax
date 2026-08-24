"""Advantage normalization - standard in PPO; keeps gradient magnitude
sane regardless of reward scale (scale itself is running_stats' job)."""
import jax


def normalize_advantages(advantages: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Zero-mean, unit-std advantages."""
    return (advantages - advantages.mean()) / (advantages.std() + eps)
