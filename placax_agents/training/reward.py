"""Wraps make_hpwl_reward with the grid-to-real-unit conversion."""
from placax.extras.rewards import make_hpwl_reward  # must precede jax imports
from placax.types import RewardFn
from placax_agents.policy.scale import to_real_centers

import jax


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
