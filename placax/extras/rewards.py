"""HPWL (half-perimeter wirelength) and wiremask(), its per-candidate
increase map used for guidance."""
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
    placed_mask: jax.Array | None = None,
) -> jax.Array:
    """Sum over nets of (bbox width + height); positions must be REAL units, not grid cells."""
    # 1. Default to "everything is placed" when the caller doesn't pass a partial-placement mask.
    if placed_mask is None:
        placed_mask = jnp.ones(positions.shape[0], dtype=bool)
    # 2. Compute every pin's absolute (x, y) center from its macro's position plus its offset.
    pin_xy = positions[padded_pin_idx].astype(jnp.float32) + padded_pin_offset
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
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-net (lo, hi, has_pins) bounds from already-placed macros only, computed once and reused
    across every macro previewed against this state (see lookahead_wiremasks)."""
    # 1. Only macros placed before the current step count toward the baseline.
    idx = state.step
    already_placed = jnp.arange(params.n_macros) < idx
    baseline_mask = valid_mask & already_placed[padded_pin_idx]

    # 2. Same min/max-bounds trick as hpwl(): mask out uncounted pins with +-_BIG.
    pin_xy = state.positions[padded_pin_idx].astype(jnp.float32) + padded_pin_offset
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
) -> jax.Array:
    """(grid_x, grid_y) HPWL-increase map for macro_idx against an already-computed baseline.

    Only ever touches macro_idx's own (small) net list, never the full netlist: a macro can
    have more than one pin on the same net, so its padded net list may contain duplicates -
    jnp.unique (fixed `size`, so the output shape stays static under jit/vmap) collapses those
    down to one local bucket per net this macro actually touches, and everything below operates
    on that small per-macro bucket set instead of a full-width (n_nets,) array. Nets this macro
    doesn't touch are unaffected by its placement, so their HPWL contribution is identical to the
    baseline and would cancel out of a (full total) - (baseline total) subtraction anyway -
    summing only the touched nets' own (new - baseline) deltas is exactly that same result,
    computed without ever materializing the full-width array."""
    my_nets = macro_net_idx[macro_idx]
    my_offsets = macro_net_offset[macro_idx]
    my_valid = macro_net_valid[macro_idx]
    n_slots = my_nets.shape[0]

    # 1. Collapse this macro's own padded (possibly duplicated) net list to unique local
    #    buckets - independent of the candidate cell, so this runs once per macro, not once
    #    per candidate cell. Invalid (padding) slots are marked -1 so they all collapse into
    #    one shared, excluded bucket instead of colliding with a real net indexed 0.
    local_net_ids, bucket_of = jnp.unique(
        jnp.where(my_valid, my_nets, -1), size=n_slots, fill_value=-1, return_inverse=True
    )
    bucket_is_real = local_net_ids >= 0
    safe_ids = jnp.where(bucket_is_real, local_net_ids, 0)
    bucket_baseline_lo = baseline_lo[safe_ids]
    bucket_baseline_hi = baseline_hi[safe_ids]
    bucket_baseline_has_pins = baseline_has_pins[safe_ids] & bucket_is_real

    def cost_at(xy: jax.Array) -> jax.Array:
        # This macro's own pins at candidate xy, folded into their local net buckets.
        my_positions = xy + my_offsets
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

    # 2. Evaluate cost_at for every grid cell at once - each call now only touches this
    #    macro's own small bucket set, not the whole netlist.
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
    macro_idx: jax.Array | int | None = None,
) -> jax.Array:
    """(grid_x, grid_y) array of HPWL(hypothetical placement at x, y) - HPWL(baseline) for macro_idx.

    macro_idx defaults to state.step; pass state.step + k to preview a macro further ahead
    (see lookahead_wiremasks) - the baseline always stays keyed to state.step, so a previewed
    macro never appears already placed."""
    idx = state.step if macro_idx is None else macro_idx
    baseline_lo, baseline_hi, baseline_has_pins = _wiremask_baseline(
        state, params, padded_pin_idx, padded_pin_offset, valid_mask
    )
    return _wiremask_from_baseline(
        baseline_lo, baseline_hi, baseline_has_pins, params,
        macro_net_idx, macro_net_offset, macro_net_valid, idx,
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
    horizon: int,
) -> jax.Array:
    """Returns wiremask() for each of the next `horizon` macros (a static Python int), sharing today's baseline."""
    # 1. The baseline (everything placed before state.step) is the same for every lookahead
    #    slot - compute it once here instead of once per slot (wiremask() would otherwise redo
    #    this full-netlist-scale computation `horizon` times over for identical results).
    baseline_lo, baseline_hi, baseline_has_pins = _wiremask_baseline(
        state, params, padded_pin_idx, padded_pin_offset, valid_mask
    )

    # 2. Clip macro indices past the last real macro to a valid index (so shapes stay fixed
    #    under jit), and separately track which slots are actually in range.
    offsets = jnp.arange(horizon)
    in_range = (state.step + offsets) < params.n_macros
    macro_idxs = jnp.clip(state.step + offsets, 0, params.n_macros - 1)

    def one(macro_idx: jax.Array, keep: jax.Array) -> jax.Array:
        wm = _wiremask_from_baseline(
            baseline_lo, baseline_hi, baseline_has_pins, params,
            macro_net_idx, macro_net_offset, macro_net_valid, macro_idx,
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
) -> RewardFn:
    """Builds a RewardFn over -HPWL (real units) * reward_scale; dense=True pays it out incrementally
    each step instead of at the end. reward_scale is a plain magnitude knob - PPO's clip_eps/entropy_coef/
    grad-norm-clip are all tuned against some assumed reward scale, so a caller matching a reference
    hyperparameter set tuned on a different position/reward unit convention should rescale here rather
    than expect the algorithm to be scale-invariant."""

    def value_of(positions: jax.Array, placed_mask: jax.Array) -> jax.Array:
        return -hpwl(positions, padded_pin_idx, padded_pin_offset, valid_mask, placed_mask)

    def reward_fn(
        old_positions: jax.Array, new_positions: jax.Array, old_placed: jax.Array, new_placed: jax.Array
    ) -> jax.Array:
        # dense: pay the change in -HPWL every step (sums to the same episode total as sparse).
        # sparse: 0 every step until the episode ends, then the whole -HPWL in one payout.
        if dense:
            raw = value_of(new_positions, new_placed) - value_of(old_positions, old_placed)
        else:
            raw = jnp.where(new_placed.all(), value_of(new_positions, new_placed), 0.0)
        return raw * reward_scale

    return reward_fn
