"""Advantage normalization, standard in PPO."""
import jax


def normalize_advantages(advantages: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Zero-mean, unit-std advantages."""
    return (advantages - advantages.mean()) / (advantages.std() + eps)
