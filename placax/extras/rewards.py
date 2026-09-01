"""HPWL (half-perimeter wirelength) and wiremask(), its per-candidate increase map used for guidance."""
from placax import _device  # noqa: F401  must run before any `import jax` below

import jax
import jax.numpy as jnp

from placax.types import EnvParams, EnvState, RewardFn

_BIG = jnp.float32(1e9)


def _quantize(pin_xy: jax.Array, cell_size: float | None) -> jax.Array:
    """Snaps a REAL-unit pin position to the nearest grid-cell-multiple lattice point, matching MaskPlace; cell_size=None skips this."""
    if cell_size is None:
        return pin_xy
    return jnp.round(pin_xy / cell_size) * cell_size


def hpwl(
    positions: jax.Array,
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    placed_mask: jax.Array | None = None,
    cell_size: float | None = None,
) -> jax.Array:
    """Sum over nets of (bbox width + height); positions must be REAL units, not grid cells; cell_size optionally quantizes pins first."""
    # 1. Default to "everything is placed" when the caller doesn't pass a partial-placement mask.
    if placed_mask is None:
        placed_mask = jnp.ones(positions.shape[0], dtype=bool)
    # 2. Compute every pin's absolute (x, y) center from its macro's position plus its offset.
    pin_xy = _quantize(positions[padded_pin_idx].astype(jnp.float32) + padded_pin_offset, cell_size)
    counted = valid_mask & placed_mask[padded_pin_idx]
    # 3. Push padding/unplaced pins to +-_BIG so they can never win the per-net min/max.
    lo = jnp.where(counted[..., None], pin_xy, _BIG).min(axis=1)
    hi = jnp.where(counted[..., None], pin_xy, -_BIG).max(axis=1)
    # 4. A net with no counted pins contributes 0 instead of a bogus (+_BIG) - (-_BIG) span.
    net_has_pins = counted.any(axis=1)
    return jnp.where(net_has_pins[:, None], hi - lo, 0.0).sum()


def _wiremask_baseline(
    state: EnvState, params: EnvParams, padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array, valid_mask: jax.Array,
    cell_size: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-net (lo, hi, has_pins) bounds from already-placed macros only, computed once and reused across every previewed macro."""
    # 1. Only macros placed before the current step count toward the baseline.
    idx = state.step
    already_placed = jnp.arange(params.n_macros) < idx
    baseline_mask = valid_mask & already_placed[padded_pin_idx]

    # 2. Same min/max-bounds trick as hpwl(), quantized the same way as candidates so bounds stay on the same lattice.
    pin_xy = _quantize(state.positions[padded_pin_idx].astype(jnp.float32) + padded_pin_offset, cell_size)
    lo = jnp.where(baseline_mask[..., None], pin_xy, _BIG).min(axis=1)
    hi = jnp.where(baseline_mask[..., None], pin_xy, -_BIG).max(axis=1)
    has_pins = baseline_mask.any(axis=1)
    return lo, hi, has_pins


def _wiremask_from_baseline(
    baseline_lo: jax.Array,
    baseline_hi: jax.Array,
    baseline_has_pins: jax.Array,
    params: EnvParams,
    macro_net_idx: jax.Array,
    macro_net_offset: jax.Array,
    macro_net_valid: jax.Array,
    macro_idx: jax.Array | int,
    cell_size: float,
    macro_size: jax.Array,
) -> jax.Array:
    """(grid_x, grid_y) HPWL-increase map for macro_idx against an already-computed baseline, touching only its own net list."""
    my_nets = macro_net_idx[macro_idx]
    my_offsets = macro_net_offset[macro_idx]
    my_valid = macro_net_valid[macro_idx]
    n_slots = my_nets.shape[0]

    # 1. Collapse this macro's padded net list to unique local buckets once per macro; padding slots are marked -1 to share one excluded bucket.
    local_net_ids, bucket_of = jnp.unique(
        jnp.where(my_valid, my_nets, -1), size=n_slots, fill_value=-1, return_inverse=True
    )
    bucket_is_real = local_net_ids >= 0
    safe_ids = jnp.where(bucket_is_real, local_net_ids, 0)
    bucket_baseline_lo = baseline_lo[safe_ids]
    bucket_baseline_hi = baseline_hi[safe_ids]
    bucket_baseline_has_pins = baseline_has_pins[safe_ids] & bucket_is_real

    def cost_at(xy: jax.Array) -> jax.Array:
        # This macro's own pins at candidate xy, folded into their local net buckets, quantized the same way as the baseline's pins.
        center = xy.astype(jnp.float32) * cell_size + macro_size / 2.0
        my_positions = _quantize(center + my_offsets, cell_size)
        row_lo = jnp.where(my_valid[:, None], my_positions, _BIG)
        row_hi = jnp.where(my_valid[:, None], my_positions, -_BIG)
        bucket_lo = jax.ops.segment_min(row_lo, bucket_of, num_segments=n_slots)
        bucket_hi = jax.ops.segment_max(row_hi, bucket_of, num_segments=n_slots)
        new_lo = jnp.minimum(bucket_baseline_lo, bucket_lo)
        new_hi = jnp.maximum(bucket_baseline_hi, bucket_hi)
        new_span = jnp.where(bucket_is_real, (new_hi - new_lo).sum(axis=-1), 0.0)
        baseline_span = jnp.where(
            bucket_baseline_has_pins, (bucket_baseline_hi - bucket_baseline_lo).sum(axis=-1), 0.0
        )
        return (new_span - baseline_span).sum()

    # 2. Evaluate cost_at for every grid cell at once, each call touching only this macro's small bucket set.
    xs, ys = jnp.meshgrid(jnp.arange(params.grid_x), jnp.arange(params.effective_grid_y), indexing="ij")
    coords = jnp.stack([xs.ravel(), ys.ravel()], axis=1)
    return jax.vmap(cost_at)(coords).reshape(params.grid_x, params.effective_grid_y)


def wiremask(
    state: EnvState,
    params: EnvParams,
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    macro_net_idx: jax.Array,
    macro_net_offset: jax.Array,
    macro_net_valid: jax.Array,
    sizes_array: jax.Array,
    cell_size: float = 1.0,
    macro_idx: jax.Array | int | None = None,
) -> jax.Array:
    """(grid_x, grid_y) array of HPWL(hypothetical placement at x, y) - HPWL(baseline) for macro_idx; state.positions must be REAL-unit centers."""
    idx = state.step if macro_idx is None else macro_idx
    baseline_lo, baseline_hi, baseline_has_pins = _wiremask_baseline(
        state, params, padded_pin_idx, padded_pin_offset, valid_mask, cell_size
    )
    return _wiremask_from_baseline(
        baseline_lo, baseline_hi, baseline_has_pins, params,
        macro_net_idx, macro_net_offset, macro_net_valid, idx,
        cell_size, sizes_array[idx],
    )


def lookahead_wiremasks(
    state: EnvState,
    params: EnvParams,
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    macro_net_idx: jax.Array,
    macro_net_offset: jax.Array,
    macro_net_valid: jax.Array,
    sizes_array: jax.Array,
    horizon: int,
    cell_size: float = 1.0,
) -> jax.Array:
    """Returns wiremask() for each of the next `horizon` macros (a static Python int), sharing today's baseline."""
    # 1. The baseline is the same for every lookahead slot, so compute it once instead of once per slot.
    baseline_lo, baseline_hi, baseline_has_pins = _wiremask_baseline(
        state, params, padded_pin_idx, padded_pin_offset, valid_mask, cell_size
    )

    # 2. Clip macro indices past the last real macro to keep shapes fixed under jit, tracking which slots are in range.
    offsets = jnp.arange(horizon)
    in_range = (state.step + offsets) < params.n_macros
    macro_idxs = jnp.clip(state.step + offsets, 0, params.n_macros - 1)

    def one(macro_idx: jax.Array, keep: jax.Array) -> jax.Array:
        wm = _wiremask_from_baseline(
            baseline_lo, baseline_hi, baseline_has_pins, params,
            macro_net_idx, macro_net_offset, macro_net_valid, macro_idx,
            cell_size, sizes_array[macro_idx],
        )
        return wm * keep

    # 3. Compute every lookahead slot's wiremask at once; out-of-range slots come out zeroed.
    return jax.vmap(one)(macro_idxs, in_range)


def make_hpwl_reward(
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    dense: bool = False,
    reward_scale: float = 1.0,
    cell_size: float | None = None,
) -> RewardFn:
    """Builds a RewardFn over -HPWL (real units) * reward_scale; dense=True pays it out incrementally each step instead of at the end."""

    def value_of(positions: jax.Array, placed_mask: jax.Array) -> jax.Array:
        return -hpwl(positions, padded_pin_idx, padded_pin_offset, valid_mask, placed_mask, cell_size)

    def reward_fn(
        old_positions: jax.Array, new_positions: jax.Array, old_placed: jax.Array, new_placed: jax.Array
    ) -> jax.Array:
        # dense: pay the change in -HPWL every step; sparse: 0 until the episode ends, then the whole -HPWL at once.
        if dense:
            raw = value_of(new_positions, new_placed) - value_of(old_positions, old_placed)
        else:
            raw = jnp.where(new_placed.all(), value_of(new_positions, new_placed), 0.0)
        return raw * reward_scale

    return reward_fn
