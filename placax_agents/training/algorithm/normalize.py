"""Advantage normalization, standard in PPO."""
import jax
import jax.numpy as jnp


def normalize_advantages(advantages: jax.Array, eps: float = 1e-8, min_std: float = 1e-3) -> jax.Array:
    """Zero-mean advantages, additionally scaled to unit std when std is large enough to trust.

    Normalizing by std is what keeps a handful of heavy-tailed transitions (e.g. placing an
    unusually large or heavily-connected macro produces a much bigger HPWL delta than a typical
    step - confirmed empirically on this environment's own reward: the largest advantage in a
    buffer commonly runs 30x the median) from dominating a minibatch's mean gradient the way an
    un-normalized batch would. But dividing by std is only safe when std reflects real spread: if
    a batch's episodes happen to be nearly identical (e.g. early in training, or a saturated,
    near-deterministic policy), std collapses toward zero and `/ (std + eps)` amplifies ordinary
    floating-point-scale noise into huge spurious values instead of failing safely - the actual
    mechanism that saturated an earlier run's policy by iteration ~18. Below min_std, skip the
    scale step (keep just the mean-centering) rather than risk that amplification."""
    centered = advantages - advantages.mean()
    std = advantages.std()
    return jnp.where(std < min_std, centered, centered / (std + eps))
