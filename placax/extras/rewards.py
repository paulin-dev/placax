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
    """Per-net (lo, hi) bounds and total HPWL from already-placed macros only, computed once and reused."""
    # 1. Only macros placed before the current step count toward the baseline.
    idx = state.step
    already_placed = jnp.arange(params.n_macros) < idx
    baseline_mask = valid_mask & already_placed[padded_pin_idx]

    # 2. Same min/max-bounds trick as hpwl(): mask out uncounted pins with +-_BIG.
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
    macro_idx: jax.Array | int | None = None,
) -> jax.Array:
    """(grid_x, grid_y) array of HPWL(hypothetical placement at x, y) - HPWL(baseline) for macro_idx.

    macro_idx defaults to state.step; pass state.step + k to preview a macro further ahead
    (see lookahead_wiremasks) - the baseline always stays keyed to state.step, so a previewed
    macro never appears already placed."""
    # 1. Get the HPWL contribution of everything already placed, shared across all candidate cells.
    idx = state.step if macro_idx is None else macro_idx
    baseline_lo, baseline_hi, baseline_total = _wiremask_baseline(
        state, params, padded_pin_idx, padded_pin_offset, valid_mask
    )
    # 2. Look up just this macro's own (small) net participation list via the reverse
    #    index - reprocessing the whole netlist per candidate cell wouldn't scale.
    my_nets = macro_net_idx[idx]
    my_offsets = macro_net_offset[idx]
    my_valid = macro_net_valid[idx]

    def cost_at(xy: jax.Array) -> jax.Array:
        # Extend just this macro's nets' baseline bounds with its pins at candidate xy.
        my_positions = xy + my_offsets
        full_lo = baseline_lo.at[my_nets].min(jnp.where(my_valid[:, None], my_positions, _BIG))
        full_hi = baseline_hi.at[my_nets].max(jnp.where(my_valid[:, None], my_positions, -_BIG))
        full_has_pins = full_lo[:, 0] < _BIG
        return jnp.where(full_has_pins, (full_hi - full_lo).sum(axis=1), 0.0).sum()

    # 3. Evaluate cost_at for every grid cell at once, then subtract the shared baseline
    #    to leave just the marginal cost of placing this macro at each cell.
    xs, ys = jnp.meshgrid(jnp.arange(params.grid_x), jnp.arange(params.effective_grid_y), indexing="ij")
    coords = jnp.stack([xs.ravel(), ys.ravel()], axis=1)
    return jax.vmap(cost_at)(coords).reshape(params.grid_x, params.effective_grid_y) - baseline_total


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
    # 1. Clip macro indices past the last real macro to a valid index (so shapes stay fixed
    #    under jit), and separately track which slots are actually in range.
    offsets = jnp.arange(horizon)
    in_range = (state.step + offsets) < params.n_macros
    macro_idxs = jnp.clip(state.step + offsets, 0, params.n_macros - 1)

    def one(macro_idx: jax.Array, keep: jax.Array) -> jax.Array:
        wm = wiremask(
            state, params, padded_pin_idx, padded_pin_offset, valid_mask,
            macro_net_idx, macro_net_offset, macro_net_valid, macro_idx=macro_idx,
        )
        return wm * keep

    # 2. Compute every lookahead slot's wiremask at once; out-of-range slots come out zeroed.
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
