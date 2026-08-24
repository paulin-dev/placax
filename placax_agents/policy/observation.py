"""Turns EnvState into what a policy consumes. Deliberately returns more
than one architecture needs: canvas is image-shaped (what a CNN wants),
but positions/sizes_array/placed_mask are the raw underlying facts any
other representation - a GNN's node/edge features, for instance - could
be built from, without this module needing to know anything about how
they'll be used. A policy that only wants canvas just ignores the rest;
nothing here favors one architecture over another beyond that."""
from placax.extras.render import render  # noqa: F401  must precede jax imports
from placax.types import EnvParams, EnvState  # noqa: F401

import jax
import jax.numpy as jnp


def observation(state: EnvState, params: EnvParams, sizes_array: jax.Array) -> dict:
    """canvas: (grid_x, grid_y) - already-placed macros only, via render().
    current_macro_size: (2,) - the macro about to be placed this step.
    positions: (n_macros, 2) - raw positions, (-1,-1) sentinel for unplaced.
    sizes_array: (n_macros, 2) - passed through unchanged, for convenience.
    placed_mask: (n_macros,) - which macros are already placed.
    step: scalar - index of the macro about to be placed."""
    canvas = render(state.positions, sizes_array, params.grid_x, params.effective_grid_y)
    return {
        "canvas": canvas,
        "current_macro_size": sizes_array[state.step],
        "positions": state.positions,
        "sizes_array": sizes_array,
        "placed_mask": state.positions[:, 0] >= 0,
        "step": state.step,
    }
