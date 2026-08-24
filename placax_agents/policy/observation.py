"""Turns EnvState into the dict a policy consumes - canvas for image
policies, plus raw positions/sizes/placed_mask any other architecture
(e.g. a GNN) could build features from."""
from placax.extras.render import render  # noqa: F401  must precede jax imports
from placax.types import EnvParams, EnvState  # noqa: F401

import jax


def observation(state: EnvState, params: EnvParams, sizes_array: jax.Array) -> dict:
    """Keys: canvas (grid_x, grid_y) of already-placed footprints,
    current_macro_size (2,), positions (n_macros, 2), sizes_array passed
    through, placed_mask (n_macros,), step (index being placed)."""
    canvas = render(state.positions, sizes_array, params.grid_x, params.effective_grid_y)
    return {
        "canvas": canvas,
        "current_macro_size": sizes_array[state.step],
        "positions": state.positions,
        "sizes_array": sizes_array,
        "placed_mask": state.positions[:, 0] >= 0,
        "step": state.step,
    }
