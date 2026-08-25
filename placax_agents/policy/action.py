"""Turns a policy's raw logits into a legal action."""
from placax.extras.masks import boundary_mask, occupancy_mask  # must precede jax imports
from placax.types import EnvParams

import jax
import jax.numpy as jnp


def legal_action_logits(
    logits: jax.Array,
    occupied: jax.Array,
    params: EnvParams,
    macro_size: tuple[int, int],
    extra_illegal: jax.Array | None = None,
) -> jax.Array:
    """Masks illegal cells (overlap or out-of-bounds) to -inf. extra_illegal,
    if given, is OR'd in too - any other (grid_x, grid_y) bool cutoff, e.g.
    extras.masks.quality_mask() over a wiremask/congestion score, for
    algorithms that restrict actions beyond bare legality. Skips masking
    if every cell is illegal (all -inf would NaN log_softmax)."""
    illegal = occupancy_mask(occupied, macro_size) | boundary_mask(params, macro_size)
    if extra_illegal is not None:
        illegal = illegal | extra_illegal
    illegal = jnp.where(illegal.all(), False, illegal)  # bail out of masking rather than go all -inf
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
