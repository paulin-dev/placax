"""Turns any policy's raw logits into a legal action - independent of
which architecture produced them (CNN, GNN, or anything else)."""
from placax.extras.masks import boundary_mask, occupancy_mask  # noqa: F401  must precede jax imports
from placax.types import EnvParams  # noqa: F401

import jax
import jax.numpy as jnp


def legal_action_logits(
    logits: jax.Array, occupied: jax.Array, params: EnvParams, macro_size: tuple[int, int]
) -> jax.Array:
    """Masks illegal cells to -inf before sampling - overlap or
    out-of-bounds placements become impossible to select, not just
    penalized after the fact.

    Takes occupied directly (a (grid_x, grid_y) boolean array of real
    footprints, e.g. from render()), not EnvState: this is what a saved
    rollout trajectory already has as `canvas`, so the exact same mask
    used at rollout time can be reconstructed at training time - needed
    for PPO's ratio, which compares probabilities under the *same*
    distribution, not a live state that no longer exists once training
    starts.

    If literally every cell is illegal (a real scenario: 543 real
    macros in a 64x64 grid with an untrained, near-random policy can
    genuinely run out of room - confirmed directly, not hypothetical),
    masking is skipped entirely rather than producing all -inf logits,
    which makes log_softmax return NaN (-inf - logsumexp(all -inf) is
    -inf - -inf). Allowing one illegal placement is far better than
    crashing the whole training run."""
    illegal = occupancy_mask(occupied, macro_size) | boundary_mask(params, macro_size)
    illegal = jnp.where(illegal.all(), False, illegal)
    return jnp.where(illegal, -jnp.inf, logits)


def sample_action(key: jax.Array, logits: jax.Array) -> jax.Array:
    """Samples one (x, y) from a (grid_x, grid_y) logits map."""
    grid = logits.shape[1]
    flat_idx = jax.random.categorical(key, logits.ravel())
    return jnp.array([flat_idx // grid, flat_idx % grid])


def action_log_prob(logits: jax.Array, action: jax.Array) -> jax.Array:
    """Log probability of `action` under the categorical distribution
    logits defines - used both to record the log prob at rollout time
    and, later, to recompute it under updated params for PPO's
    importance-sampling ratio (same action, different policy weights)."""
    grid = logits.shape[1]
    flat_idx = action[0] * grid + action[1]
    return jax.nn.log_softmax(logits.ravel())[flat_idx]
