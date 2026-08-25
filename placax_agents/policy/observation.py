"""Turns EnvState into the dict a policy consumes."""
from placax.extras.render import render  # must precede jax imports
from placax.extras.rewards import lookahead_wiremasks
from placax.types import EnvParams, EnvState

import jax
import jax.numpy as jnp


def lookahead_sizes(state: EnvState, params: EnvParams, sizes_array: jax.Array, horizon: int) -> jax.Array:
    """Returns the REAL-unit sizes of the next `horizon` macros, starting with the one about to be placed."""
    # Indices of the next `horizon` macros (horizon is a static int, so this shape is jit-stable).
    offsets = jnp.arange(horizon)
    in_range = (state.step + offsets) < params.n_macros
    # Clip so we never index past the end of the array, even for out-of-range slots.
    idx = jnp.clip(state.step + offsets, 0, params.n_macros - 1)
    # Zero out any slot beyond the last macro so the caller can tell it's padding, not a real size.
    return sizes_array[idx] * in_range[:, None]


def observation(state: EnvState, params: EnvParams, sizes_array: jax.Array, lookahead: int = 1) -> dict:
    """Builds the base observation dict (canvas, current/lookahead macro sizes, positions, step) any policy can read."""
    # Render already-placed macros onto the grid; this is what the policy "sees" as the board state.
    canvas = render(state.positions, sizes_array, params.grid_x, params.effective_grid_y)
    return {
        "canvas": canvas,
        "current_macro_size": sizes_array[state.step],
        "lookahead_sizes": lookahead_sizes(state, params, sizes_array, lookahead),
        "positions": state.positions,
        "sizes_array": sizes_array,
        "placed_mask": state.positions[:, 0] >= 0,
        "step": state.step,
    }


def make_wiremask_observation(
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    macro_net_idx: jax.Array,
    macro_net_offset: jax.Array,
    macro_net_valid: jax.Array,
    base_state_fn=observation,
    lookahead: int = 1,
):
    """Wraps a StateFn to add "wiremask"/"lookahead_wiremasks" (per-cell HPWL-increase previews); build once per benchmark."""

    def state_fn(state: EnvState, params: EnvParams, sizes_array: jax.Array) -> dict:
        # Start from the base observation (canvas, sizes, etc.), then layer wiremask info on top.
        obs = base_state_fn(state, params, sizes_array)
        # For each of the next `lookahead` macros, compute the per-cell HPWL cost of placing it there.
        maps = lookahead_wiremasks(
            state, params, padded_pin_idx, padded_pin_offset, valid_mask,
            macro_net_idx, macro_net_offset, macro_net_valid, lookahead,
        )
        # "wiremask" is just the first (current-macro) slice, kept for policies that only look one step ahead.
        obs["wiremask"] = maps[0]
        obs["lookahead_wiremasks"] = maps
        return obs

    return state_fn
