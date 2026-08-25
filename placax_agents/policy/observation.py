"""Turns EnvState into the dict a policy consumes."""
from placax.extras.render import render  # must precede jax imports
from placax.extras.rewards import wiremask
from placax.types import EnvParams, EnvState

import jax
import jax.numpy as jnp


def lookahead_sizes(state: EnvState, params: EnvParams, sizes_array: jax.Array, horizon: int) -> jax.Array:
    """(horizon, 2) REAL-unit sizes of the next `horizon` macros still to
    place, starting with the one about to be placed now. horizon is a
    static Python int, so the shape stays fixed under jit regardless of
    how many macros remain; slots past the last macro are zeroed out."""
    offsets = jnp.arange(horizon)
    in_range = (state.step + offsets) < params.n_macros
    idx = jnp.clip(state.step + offsets, 0, params.n_macros - 1)
    return sizes_array[idx] * in_range[:, None]


def observation(state: EnvState, params: EnvParams, sizes_array: jax.Array, lookahead: int = 1) -> dict:
    """Keys: canvas (grid_x, grid_y) bool, already-placed footprints;
    current_macro_size (2,) float, REAL units, same as lookahead_sizes[0];
    lookahead_sizes (lookahead, 2) float, REAL units, see lookahead_sizes()
    above - pass a larger `lookahead` for policies that plan ahead;
    positions (n_macros, 2) int, grid units (state.positions passed
    through); sizes_array (n_macros, 2) float, REAL units, passed through
    unchanged; placed_mask (n_macros,) bool; step () int, index being
    placed next."""
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
):
    """Wraps any StateFn, adding a "wiremask" key: (grid_x, grid_y) float,
    the per-cell HPWL increase from placing the current macro there (see
    extras.rewards.wiremask()). Closes over one benchmark's netlist
    arrays - build once per benchmark, pass the result as state_fn to
    collect_rollout/train_*/evaluate. Generic: pairs with any base
    StateFn and any policy that reads the extra "wiremask" key, e.g.
    policy.architectures.wiremask_cnn.WiremaskCNNActorCritic."""

    def state_fn(state: EnvState, params: EnvParams, sizes_array: jax.Array) -> dict:
        obs = base_state_fn(state, params, sizes_array)
        obs["wiremask"] = wiremask(
            state, params, padded_pin_idx, padded_pin_offset, valid_mask,
            macro_net_idx, macro_net_offset, macro_net_valid,
        )
        return obs

    return state_fn
