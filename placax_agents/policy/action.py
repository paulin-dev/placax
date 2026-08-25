"""Turns a policy's raw logits into a legal action."""
from placax.extras.masks import boundary_mask, occupancy_mask  # noqa: F401  must precede jax imports
from placax.types import EnvParams  # noqa: F401

import jax
import jax.numpy as jnp


def legal_action_logits(
    logits: jax.Array, occupied: jax.Array, params: EnvParams, macro_size: tuple[int, int]
) -> jax.Array:
    """Masks illegal cells (overlap or out-of-bounds) to -inf. Skips
    masking if every cell is illegal (all -inf would NaN log_softmax)."""
    illegal = occupancy_mask(occupied, macro_size) | boundary_mask(params, macro_size)
    illegal = jnp.where(illegal.all(), False, illegal)
    return jnp.where(illegal, -jnp.inf, logits)


def sample_action(key: jax.Array, logits: jax.Array) -> jax.Array:
    """Samples one (x, y) from a (grid_x, grid_y) logits map."""
    grid = logits.shape[1]
    flat_idx = jax.random.categorical(key, logits.ravel())
    return jnp.array([flat_idx // grid, flat_idx % grid])


def action_log_prob(logits: jax.Array, action: jax.Array) -> jax.Array:
    """Log probability of `action` under logits - used at rollout time
    and recomputed under updated params for PPO's ratio."""
    grid = logits.shape[1]
    flat_idx = action[0] * grid + action[1]
    return jax.nn.log_softmax(logits.ravel())[flat_idx]
