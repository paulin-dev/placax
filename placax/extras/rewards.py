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
    """sum over nets of (bbox width + height); a net with 0 counted pins
    contributes 0 (guarded explicitly - sentinels alone would give a
    large negative value). positions: (n_macros, 2) float, REAL units -
    combined directly with padded_pin_offset (also real units), so
    passing grid-cell coordinates here silently gives a meaningless
    number rather than an error. placed_mask: (n_macros,) bool, which
    macros' positions are real - a pin whose macro isn't placed yet is
    excluded even if valid_mask says the slot is real, letting hpwl() be
    called on a partial assignment. Defaults to "every macro is placed",
    the historical full-assignment behavior."""
    if placed_mask is None:
        placed_mask = jnp.ones(positions.shape[0], dtype=bool)
    pin_xy = positions[padded_pin_idx].astype(jnp.float32) + padded_pin_offset  # pin centers
    counted = valid_mask & placed_mask[padded_pin_idx]
    # Padding/unplaced slots are masked to +-_BIG so they can never win the min/max.
    lo = jnp.where(counted[..., None], pin_xy, _BIG).min(axis=1)
    hi = jnp.where(counted[..., None], pin_xy, -_BIG).max(axis=1)
    net_has_pins = counted.any(axis=1)
    return jnp.where(net_has_pins[:, None], hi - lo, 0.0).sum()  # zero out empty nets, then sum


def _wiremask_baseline(
    state: EnvState, params: EnvParams, padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array, valid_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-net (lo, hi) bounds and total HPWL from already-placed macros
    only - computed once, reused by wiremask's per-candidate combine."""
    idx = state.step
    already_placed = jnp.arange(params.n_macros) < idx
    baseline_mask = valid_mask & already_placed[padded_pin_idx]  # only already-placed macros' pins count

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
    """(grid_x, grid_y) array: wiremask(x, y) = HPWL(hypothetical at
    x, y) - HPWL(baseline), for placing macro_idx there. macro_idx
    defaults to state.step (the macro actually about to be placed); pass
    state.step + k to preview a macro further ahead in a lookahead window
    (see lookahead_wiremasks below) - the baseline (which macros already
    have a real position) always stays keyed to state.step itself, not
    macro_idx, so previewing a later macro never pretends an earlier,
    still-undecided one has been placed."""
    idx = state.step if macro_idx is None else macro_idx
    baseline_lo, baseline_hi, baseline_total = _wiremask_baseline(
        state, params, padded_pin_idx, padded_pin_offset, valid_mask
    )
    # This macro's own (small) net participation list, via the reverse
    # index - the whole netlist per candidate cell wouldn't scale.
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

    # Evaluate cost_at for every grid cell at once.
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
    """(horizon, grid_x, grid_y) array: wiremask() for each of the next
    `horizon` macros (state.step, state.step+1, ...), all previewed
    against today's canvas - none of them are placed yet, so they all
    share the same baseline. horizon is a static Python int, so the
    shape stays fixed under jit; slots past the last macro are zeroed."""
    offsets = jnp.arange(horizon)
    in_range = (state.step + offsets) < params.n_macros
    macro_idxs = jnp.clip(state.step + offsets, 0, params.n_macros - 1)

    def one(macro_idx: jax.Array, keep: jax.Array) -> jax.Array:
        wm = wiremask(
            state, params, padded_pin_idx, padded_pin_offset, valid_mask,
            macro_net_idx, macro_net_offset, macro_net_valid, macro_idx=macro_idx,
        )
        return wm * keep

    return jax.vmap(one)(macro_idxs, in_range)


def make_hpwl_reward(
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    dense: bool = False,
) -> RewardFn:
    """Builds reward_fn(old_positions, new_positions, old_placed, new_placed)
    matching core.step()'s RewardFn contract - called every step.
    positions must be real units (see hpwl()) - this does NOT satisfy
    make_scaled_hpwl_reward's grid-unit contract on its own; use that to
    plug straight into training, or convert positions yourself.

    dense=False (default): 0 every step, -HPWL(new) once every macro is
    placed - the classic sparse/terminal reward.
    dense=True: -(HPWL(placed-so-far after) - HPWL(placed-so-far before))
    every step - a dense, per-action reward. Its episode sum telescopes
    to the exact same total as dense=False (HPWL of an empty placement is
    0), just paid out across steps instead of only at the end - sparse
    and dense are two credit-assignment choices over the same quantity,
    not two different mechanisms."""

    def value_of(positions: jax.Array, placed_mask: jax.Array) -> jax.Array:
        return -hpwl(positions, padded_pin_idx, padded_pin_offset, valid_mask, placed_mask)

    def reward_fn(
        old_positions: jax.Array, new_positions: jax.Array, old_placed: jax.Array, new_placed: jax.Array
    ) -> jax.Array:
        if dense:
            return value_of(new_positions, new_placed) - value_of(old_positions, old_placed)
        return jnp.where(new_placed.all(), value_of(new_positions, new_placed), 0.0)

    return reward_fn
