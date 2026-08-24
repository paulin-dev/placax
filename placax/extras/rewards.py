"""HPWL: half-perimeter wirelength, the standard cheap wirelength proxy."""
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
    """Sum of per-net bounding-box (width + height), padding masked out.

    HPWL(net) = (max_x - min_x) + (max_y - min_y), summed over nets.
    A net with zero valid pins contributes 0 (guarded explicitly - the
    masking sentinels alone would otherwise produce a large negative
    value instead)."""
    pin_xy = positions[padded_pin_idx].astype(jnp.float32) + padded_pin_offset
    lo = jnp.where(valid_mask[..., None], pin_xy, _BIG).min(axis=1)
    hi = jnp.where(valid_mask[..., None], pin_xy, -_BIG).max(axis=1)
    net_has_pins = valid_mask.any(axis=1)
    per_net_span = jnp.where(net_has_pins[:, None], hi - lo, 0.0)
    return per_net_span.sum()


def _wiremask_baseline(
    state: EnvState, params: EnvParams, padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array, valid_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-net (lo, hi) and total HPWL from already-placed macros only -
    computed once, reused by wiremask's per-candidate combine step."""
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
    """For the macro about to be placed (state.step), returns a
    (grid_x, grid_y) array: the HPWL increase from placing it at each
    candidate cell, counting only already-placed macros plus this one.

        wiremask(x, y) = HPWL(hypothetical at x, y) - HPWL(baseline)

    Uses build_macro_net_index's reverse index rather than gathering the
    whole netlist per candidate cell - keeps the per-candidate cost to
    just this macro's own (small) participation list, which is what
    makes vmap over every candidate cell affordable at real scale."""
    idx = state.step
    baseline_lo, baseline_hi, baseline_total = _wiremask_baseline(
        state, params, padded_pin_idx, padded_pin_offset, valid_mask
    )
    my_nets = macro_net_idx[idx]
    my_offsets = macro_net_offset[idx]
    my_valid = macro_net_valid[idx]

    def cost_at(xy: jax.Array) -> jax.Array:
        my_positions = xy + my_offsets
        for_min = jnp.where(my_valid[:, None], my_positions, _BIG)
        for_max = jnp.where(my_valid[:, None], my_positions, -_BIG)
        full_lo = baseline_lo.at[my_nets].min(for_min)
        full_hi = baseline_hi.at[my_nets].max(for_max)
        full_has_pins = full_lo[:, 0] < _BIG
        return jnp.where(full_has_pins, (full_hi - full_lo).sum(axis=1), 0.0).sum()

    xs, ys = jnp.meshgrid(jnp.arange(params.grid_x), jnp.arange(params.effective_grid_y), indexing="ij")
    coords = jnp.stack([xs.ravel(), ys.ravel()], axis=1)
    total_flat = jax.vmap(cost_at)(coords)
    return total_flat.reshape(params.grid_x, params.effective_grid_y) - baseline_total


def make_hpwl_reward(
    padded_pin_idx: jax.Array, padded_pin_offset: jax.Array, valid_mask: jax.Array
) -> RewardFn:
    """Build a reward_fn closing over one netlist's fixed pin structure.

        reward(positions) = -HPWL(positions)

    Sign flipped so smaller wirelength means larger (better) reward,
    matching standard RL convention."""

    def reward_fn(positions: jax.Array) -> jax.Array:
        return -hpwl(positions, padded_pin_idx, padded_pin_offset, valid_mask)

    return reward_fn
