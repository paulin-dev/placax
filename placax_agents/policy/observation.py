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
    """Builds the base observation dict (canvas, current/lookahead macro sizes, positions, step) any policy can read.

    Deliberately excludes a "sizes_array" key: unlike positions/placed_mask (genuinely different
    every step), sizes_array is the exact same (n_macros, 2) array on every single step of an
    episode - buffering it per step (see training.rollout.collect_rollout) would replicate it
    n_macros times over for zero new information. A policy that needs it has it available the
    same way this function does: as its own sizes_array argument, not through obs."""
    # Render already-placed macros onto the grid; this is what the policy "sees" as the board state.
    canvas = render(state.positions, sizes_array, params.grid_x, params.effective_grid_y)
    return {
        "canvas": canvas,
        "current_macro_size": sizes_array[state.step],
        "lookahead_sizes": lookahead_sizes(state, params, sizes_array, lookahead),
        "positions": state.positions,
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
    """Wraps a StateFn to add "lookahead_wiremasks" (per-cell HPWL-increase previews); build once per benchmark.

    The current-macro slice is lookahead_wiremasks[0] - deliberately not also duplicated under a
    separate "wiremask" key, since every full grid-resolution channel here gets buffered per step
    of every episode in the PPO trajectory (see training.rollout.collect_rollout); a policy or
    ExtraIllegalFn that only cares about one step ahead should read lookahead_wiremasks[0] directly
    (placax_agents.policy.action.make_wiremask_quality_illegal already falls back to exactly that
    when its "wiremask" key isn't present in obs)."""

    def state_fn(state: EnvState, params: EnvParams, sizes_array: jax.Array) -> dict:
        # Start from the base observation (canvas, sizes, etc.), then layer wiremask info on top.
        obs = base_state_fn(state, params, sizes_array)
        # For each of the next `lookahead` macros, compute the per-cell HPWL cost of placing it there.
        obs["lookahead_wiremasks"] = lookahead_wiremasks(
            state, params, padded_pin_idx, padded_pin_offset, valid_mask,
            macro_net_idx, macro_net_offset, macro_net_valid, lookahead,
        )
        return obs

    return state_fn
