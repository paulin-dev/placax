"""Advantage normalization, standard in PPO."""
import jax
import jax.numpy as jnp


def normalize_advantages(advantages: jax.Array, eps: float = 1e-8, min_std: float = 1e-3) -> jax.Array:
    """Zero-mean advantages, additionally scaled to unit std when std is large enough to trust safely."""
    centered = advantages - advantages.mean()
    std = advantages.std()
    return jnp.where(std < min_std, centered, centered / (std + eps))
