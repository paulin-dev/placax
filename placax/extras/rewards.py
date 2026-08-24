"""HPWL (half-perimeter wirelength) - the standard cheap wirelength proxy
- plus wiremask(), the per-candidate HPWL-increase map used for guidance."""
from placax import _device  # noqa: F401  must run before any `import jax` below

import jax
import jax.numpy as jnp

from placax.types import EnvParams, EnvState, RewardFn

_BIG = jnp.float32(1e9)


def hpwl(
    positions: jax.Array,
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
) -> jax.Array:
    """Sum over nets of bounding-box width + height, padding masked out.
    A net with zero valid pins contributes 0 (guarded explicitly - the
    masking sentinels alone would give a large negative value)."""
    pin_xy = positions[padded_pin_idx].astype(jnp.float32) + padded_pin_offset
    lo = jnp.where(valid_mask[..., None], pin_xy, _BIG).min(axis=1)
    hi = jnp.where(valid_mask[..., None], pin_xy, -_BIG).max(axis=1)
    net_has_pins = valid_mask.any(axis=1)
    return jnp.where(net_has_pins[:, None], hi - lo, 0.0).sum()


def _wiremask_baseline(
    state: EnvState, params: EnvParams, padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array, valid_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-net (lo, hi) bounds and total HPWL from already-placed macros
    only - computed once, reused by wiremask's per-candidate combine."""
    idx = state.step
    already_placed = jnp.arange(params.n_macros) < idx
    baseline_mask = valid_mask & already_placed[padded_pin_idx]

    pin_xy = state.positions[padded_pin_idx].astype(jnp.float32) + padded_pin_offset
    lo = jnp.where(baseline_mask[..., None], pin_xy, _BIG).min(axis=1)
    hi = jnp.where(baseline_mask[..., None], pin_xy, -_BIG).max(axis=1)
    has_pins = baseline_mask.any(axis=1)
    total = jnp.where(has_pins, (hi - lo).sum(axis=1), 0.0).sum()
    return lo, hi, total


def wiremask(
    state: EnvState,
    params: EnvParams,
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    macro_net_idx: jax.Array,
    macro_net_offset: jax.Array,
    macro_net_valid: jax.Array,
) -> jax.Array:
    """(grid_x, grid_y) array of the HPWL increase from placing the macro
    at state.step at each candidate cell:

        wiremask(x, y) = HPWL(hypothetical at x, y) - HPWL(baseline)

    Uses build_macro_net_index's reverse index so each candidate touches
    only this macro's own small participation list - what makes vmap
    over every cell affordable at real scale."""
    idx = state.step
    baseline_lo, baseline_hi, baseline_total = _wiremask_baseline(
        state, params, padded_pin_idx, padded_pin_offset, valid_mask
    )
    my_nets = macro_net_idx[idx]
    my_offsets = macro_net_offset[idx]
    my_valid = macro_net_valid[idx]

    def cost_at(xy: jax.Array) -> jax.Array:
        my_positions = xy + my_offsets
        full_lo = baseline_lo.at[my_nets].min(jnp.where(my_valid[:, None], my_positions, _BIG))
        full_hi = baseline_hi.at[my_nets].max(jnp.where(my_valid[:, None], my_positions, -_BIG))
        full_has_pins = full_lo[:, 0] < _BIG
        return jnp.where(full_has_pins, (full_hi - full_lo).sum(axis=1), 0.0).sum()

    xs, ys = jnp.meshgrid(jnp.arange(params.grid_x), jnp.arange(params.effective_grid_y), indexing="ij")
    coords = jnp.stack([xs.ravel(), ys.ravel()], axis=1)
    return jax.vmap(cost_at)(coords).reshape(params.grid_x, params.effective_grid_y) - baseline_total


def make_hpwl_reward(
    padded_pin_idx: jax.Array, padded_pin_offset: jax.Array, valid_mask: jax.Array
) -> RewardFn:
    """Builds a reward_fn closing over one netlist's fixed pin structure:
    reward(positions) = -HPWL(positions) (sign flipped to match RL
    convention: smaller wirelength = larger reward)."""

    def reward_fn(positions: jax.Array) -> jax.Array:
        return -hpwl(positions, padded_pin_idx, padded_pin_offset, valid_mask)

    return reward_fn
