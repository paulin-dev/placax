"""Turns EnvState into the dict a policy consumes."""
from placax.extras.render import render  # must precede jax imports
from placax.types import EnvParams, EnvState

import jax


def observation(state: EnvState, params: EnvParams, sizes_array: jax.Array) -> dict:
    """Keys: canvas (grid_x, grid_y) bool, already-placed footprints;
    current_macro_size (2,) float, REAL units; positions (n_macros, 2)
    int, grid units (state.positions passed through); sizes_array
    (n_macros, 2) float, REAL units, passed through unchanged;
    placed_mask (n_macros,) bool; step () int, index being placed next."""
    canvas = render(state.positions, sizes_array, params.grid_x, params.effective_grid_y)
    return {
        "canvas": canvas,
        "current_macro_size": sizes_array[state.step],
        "positions": state.positions,
        "sizes_array": sizes_array,
        "placed_mask": state.positions[:, 0] >= 0,
        "step": state.step,
    }
