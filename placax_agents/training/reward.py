"""Wraps make_hpwl_reward with the grid-to-real-unit conversion."""
from placax.extras.masks import regularity_cost, regularity_max  # must precede jax imports
from placax.extras.rewards import make_hpwl_reward
from placax.types import EnvParams, RewardFn
from placax_agents.policy.scale import to_grid_units, to_real_centers

import jax
import jax.numpy as jnp


def make_scaled_hpwl_reward(
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    sizes_array: jax.Array,
    cell_size: float,
    dense: bool = False,
    reward_scale: float = 1.0,
) -> RewardFn:
    """Wraps make_hpwl_reward so it scores real-unit macro centers instead of grid positions."""
    # Build the underlying -HPWL reward once; only its inputs get rescaled below. cell_size is
    # passed through so hpwl() quantizes pins to the nearest grid cell, matching the reference's
    # own per-step reward (place_env.py rounds every pin position it stores).
    base_reward_fn = make_hpwl_reward(
        padded_pin_idx, padded_pin_offset, valid_mask, dense=dense, reward_scale=reward_scale, cell_size=cell_size
    )

    def reward_fn(
        old_positions: jax.Array, new_positions: jax.Array, old_placed: jax.Array, new_placed: jax.Array
    ) -> jax.Array:
        """Converts grid positions to real-unit centers, then delegates to the -HPWL reward."""
        return base_reward_fn(
            to_real_centers(old_positions, sizes_array, cell_size),
            to_real_centers(new_positions, sizes_array, cell_size),
            old_placed,
            new_placed,
        )

    return reward_fn


def make_expert_reward(
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    sizes_array: jax.Array,
    cell_size: float,
    params: EnvParams,
    dense: bool = True,
    reward_scale: float = 1.0,
    regularity_weight: float = 0.0,
    regularity_mode: str = "corner",
) -> RewardFn:
    """Adds EXPlace's regularity (periphery) term to the -HPWL reward: `hpwl_reward - w * normalized_cost`.

    The regularity cost is normalized to [0, 1] by regularity_max() for the macro just placed, so
    `regularity_weight` is directly comparable across macros of different sizes. Note this deviates
    from EXPlace, which instead rescales by running per-episode min/max - a stateful scheme placax's
    pure RewardFn has nowhere to keep, and one that makes the reward non-stationary early in an
    episode. regularity_weight=0.0 reproduces make_scaled_hpwl_reward exactly.

    The two terms are NOT auto-balanced: the HPWL term keeps whatever units `reward_scale` leaves
    it in, while the regularity term is always [0, 1] * regularity_weight. Pick regularity_weight
    against the HPWL term's actual per-step magnitude for your benchmark, not against EXPlace's
    published coefficients - those are ratios between six terms that it has already normalized
    to a common scale, so they do not transfer directly.
    """
    hpwl_reward_fn = make_scaled_hpwl_reward(
        padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size,
        dense=dense, reward_scale=reward_scale,
    )

    def reward_fn(
        old_positions: jax.Array, new_positions: jax.Array, old_placed: jax.Array, new_placed: jax.Array
    ) -> jax.Array:
        reward = hpwl_reward_fn(old_positions, new_positions, old_placed, new_placed)
        if regularity_weight == 0.0:
            return reward
        # Recover which macro this step placed, and where, straight from the RewardFn signature -
        # exactly one macro flips from unplaced to placed per step(), so no wider signature is needed.
        newly_placed = new_placed & ~old_placed
        idx = jnp.argmax(newly_placed)
        macro_size = to_grid_units(sizes_array[idx], cell_size)
        cost = regularity_cost(params, macro_size, new_positions[idx], cell_size, regularity_mode)
        # A macro too big to leave any interior has an all-zero cost map; guard that 0/0.
        scale = regularity_max(params, macro_size, cell_size, regularity_mode)
        normalized = jnp.where(scale > 0, cost / jnp.where(scale > 0, scale, 1.0), 0.0)
        # Nothing placed (never happens via step(), but keeps the term well-defined) costs nothing.
        return reward - regularity_weight * jnp.where(newly_placed.any(), normalized, 0.0)

    return reward_fn
