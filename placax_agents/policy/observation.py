"""Turns EnvState into the dict a policy consumes."""
from placax.extras.orientation import effective_sizes  # must precede jax imports
from placax.extras.render import render
from placax.extras.rewards import lookahead_wiremasks
from placax.types import EnvParams, EnvState
from placax_agents.policy.scale import to_grid_units, to_real_centers

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


def observation(
    state: EnvState, params: EnvParams, sizes_array: jax.Array, cell_size: float = 1.0, lookahead: int = 1
) -> dict:
    """Builds the base observation dict (canvas, current/lookahead macro sizes, positions, step) any policy can read; excludes sizes_array since it's constant across an episode."""
    # Render already-placed macros onto the grid; this is what the policy "sees" as the board state.
    # Footprints are taken AS PLACED: a macro turned a quarter turn occupies its height by its
    # width, and a canvas drawn from unrotated sizes would show a different design from the one
    # being scored. `effective_sizes` is the identity when nothing is oriented, so this costs the
    # un-oriented path nothing. See placax/extras/orientation.py.
    placed_sizes = effective_sizes(sizes_array, state.orientations)
    grid_sizes = to_grid_units(placed_sizes, cell_size)
    canvas = render(state.positions, grid_sizes, params.grid_x, params.effective_grid_y)
    return {
        "canvas": canvas,
        # The macro about to be placed, in its UNROTATED size: its orientation is part of the
        # action still to be chosen, not of the state being observed.
        "current_macro_size": sizes_array[state.step],
        "lookahead_sizes": lookahead_sizes(state, params, sizes_array, lookahead),
        "positions": state.positions,
        "placed_mask": state.positions[:, 0] >= 0,
        "step": state.step,
        "orientations": state.orientations,
    }


# `current_macro_size`, `lookahead_sizes` and `step` are all about the macro being placed NEXT, so
# this observation only means anything under a space that names one - see ActionSpace.target() and
# build()'s checks. The canvas, positions and placed mask are true whatever the space is, which is
# why local_search can read those under a perturbation space.
observation.needs_current_macro = True


def make_wiremask_observation(
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    macro_net_idx: jax.Array,
    macro_net_offset: jax.Array,
    macro_net_valid: jax.Array,
    cell_size: float = 1.0,
    base_state_fn=observation,
    lookahead: int = 1,
):
    """Wraps a StateFn to add "lookahead_wiremasks" (per-cell HPWL-increase previews); build once per benchmark."""

    def state_fn(state: EnvState, params: EnvParams, sizes_array: jax.Array) -> dict:
        # Start from the base observation (canvas, sizes, etc.), then layer wiremask info on top.
        obs = base_state_fn(state, params, sizes_array, cell_size=cell_size)
        # For each of the next `lookahead` macros, compute the per-cell HPWL cost of placing it there.
        real_state = EnvState(positions=to_real_centers(state.positions, sizes_array, cell_size), step=state.step)
        obs["lookahead_wiremasks"] = lookahead_wiremasks(
            real_state, params, padded_pin_idx, padded_pin_offset, valid_mask,
            macro_net_idx, macro_net_offset, macro_net_valid, sizes_array, lookahead,
            cell_size=cell_size,
        )
        return obs

    # The wiremask baseline is "the first state.step macros are placed" - constructive-only, for
    # the same reason as the base observation above.
    state_fn.needs_current_macro = True
    return state_fn
