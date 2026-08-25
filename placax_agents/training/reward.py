"""Wraps make_hpwl_reward with the grid-to-real-unit conversion."""
from placax.extras.rewards import make_hpwl_reward  # noqa: F401  must precede jax imports
from placax.types import RewardFn  # noqa: F401
from placax_agents.policy.scale import to_real_centers  # noqa: F401

import jax


def make_scaled_hpwl_reward(
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    sizes_array: jax.Array,
    cell_size: float,
) -> RewardFn:
    """reward(positions) = -HPWL of real-unit macro centers (grid
    positions converted via to_real_centers)."""
    base_reward_fn = make_hpwl_reward(padded_pin_idx, padded_pin_offset, valid_mask)

    def reward_fn(positions: jax.Array) -> jax.Array:
        return base_reward_fn(to_real_centers(positions, sizes_array, cell_size))

    return reward_fn
