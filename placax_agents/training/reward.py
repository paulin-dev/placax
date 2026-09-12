"""Wraps make_hpwl_reward with the grid-to-real-unit conversion.

**Orientation enters here, as a transform on the inputs.** Every reward in this file turns grid
cells into real-unit centers with the design's macro footprints, and a quarter-turned macro is its
height by its width - so its center sits somewhere else, and its pins rotate with it.
`effective_sizes` and `oriented_pin_offsets` (extras/orientation.py) are both the identity on
`None`, which is what every un-oriented run passes, so this costs the historical path nothing and
keeps its numbers bit-identical.
"""
from placax.extras.masks import regularity_cost, regularity_max  # must precede jax imports
from placax.extras.orientation import effective_sizes
from placax.extras.congestion import make_congestion_cost
from placax.extras.density import make_density_cost
from placax.extras.rewards import make_hpwl_reward, make_smoothed_wirelength_reward
from placax.types import EnvParams, RewardFn
from placax_agents.policy.scale import to_grid_units, to_real_centers

import jax
import jax.numpy as jnp


def make_scaled_hpwl_reward(
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    sizes_array: jax.Array,
    cell_size: float,
    dense: bool = False,
    reward_scale: float = 1.0,
) -> RewardFn:
    """Wraps make_hpwl_reward so it scores real-unit macro centers instead of grid positions."""
    # Build the underlying -HPWL reward once; only its inputs get rescaled below. cell_size is
    # passed through so hpwl() quantizes pins to the nearest grid cell, matching the reference's
    # own per-step reward (place_env.py rounds every pin position it stores).
    base_reward_fn = make_hpwl_reward(
        padded_pin_idx, padded_pin_offset, valid_mask, dense=dense, reward_scale=reward_scale, cell_size=cell_size
    )

    def reward_fn(
        old_positions: jax.Array, new_positions: jax.Array, old_placed: jax.Array,
        new_placed: jax.Array, orientations: jax.Array | None = None,
    ) -> jax.Array:
        """Converts grid positions to real-unit centers, then delegates to the -HPWL reward."""
        sizes = effective_sizes(sizes_array, orientations)
        return base_reward_fn(
            to_real_centers(old_positions, sizes, cell_size),
            to_real_centers(new_positions, sizes, cell_size),
            old_placed,
            new_placed,
            orientations,
        )

    return reward_fn


def make_scaled_smoothed_reward(
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    sizes_array: jax.Array,
    cell_size: float,
    dense: bool = False,
    reward_scale: float = 1.0,
    gamma: float = 1.0,
) -> RewardFn:
    """make_scaled_hpwl_reward over the smooth log-sum-exp surrogate instead of raw HPWL.

    Same units and same shape, so it substitutes wherever the HPWL reward is used - the point
    being that the reward axis is the one place this project can already test a gradient-friendly
    objective without touching the kernel (docs/Action_Space_Decision.md, option D).
    """
    base_reward_fn = make_smoothed_wirelength_reward(
        padded_pin_idx, padded_pin_offset, valid_mask, dense=dense, reward_scale=reward_scale,
        cell_size=cell_size, gamma=gamma,
    )

    def reward_fn(
        old_positions: jax.Array, new_positions: jax.Array, old_placed: jax.Array,
        new_placed: jax.Array, orientations: jax.Array | None = None,
    ) -> jax.Array:
        sizes = effective_sizes(sizes_array, orientations)
        return base_reward_fn(
            to_real_centers(old_positions, sizes, cell_size),
            to_real_centers(new_positions, sizes, cell_size),
            old_placed,
            new_placed,
            orientations,
        )

    return reward_fn


def make_hpwl_congestion_reward(
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    sizes_array: jax.Array,
    cell_size: float,
    params: EnvParams,
    congestion_weight: float = 1.0,
    capacity: float = 1.0,
    dense: bool = False,
    reward_scale: float = 1.0,
) -> RewardFn:
    """-(HPWL + congestion_weight * RUDY overflow) - the second objective the reward comparison needs.

    The two terms are NOT auto-balanced, for the same reason make_expert_reward's aren't: HPWL is
    in real design units and overflow is in bins of excess wire demand, so `congestion_weight` has
    to be sized against the HPWL term's actual magnitude on YOUR benchmark. Run
    scripts/measure_reward_terms.py before picking one; a weight transplanted from another design
    is meaningless.

    Defaults to sparse. Congestion costs a grid-sized matmul where HPWL costs a reduction over
    pins, so paying it once at the end of an episode is the affordable choice unless you have
    specifically decided otherwise - see placax.extras.congestion for the cost discussion.
    """
    hpwl_reward_fn = make_scaled_hpwl_reward(
        padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size,
        dense=dense, reward_scale=1.0,
    )
    congestion_cost = make_congestion_cost(
        padded_pin_idx, padded_pin_offset, valid_mask, params, cell_size, capacity
    )

    def reward_fn(
        old_positions: jax.Array, new_positions: jax.Array, old_placed: jax.Array,
        new_placed: jax.Array, orientations: jax.Array | None = None,
    ) -> jax.Array:
        reward = hpwl_reward_fn(
            old_positions, new_positions, old_placed, new_placed, orientations
        )
        sizes = effective_sizes(sizes_array, orientations)
        if congestion_weight == 0.0:
            return reward * reward_scale
        # Congestion is a property of a layout, not of a step, so the dense form pays its DELTA -
        # which telescopes over an episode to the same total the sparse form pays once, exactly
        # as the HPWL term does.
        new_cost = congestion_cost(
            to_real_centers(new_positions, sizes, cell_size), new_placed
        )
        if dense:
            old_cost = congestion_cost(
                to_real_centers(old_positions, sizes, cell_size), old_placed
            )
            penalty = new_cost - old_cost
        else:
            penalty = jnp.where(new_placed.all(), new_cost, 0.0)
        return (reward - congestion_weight * penalty) * reward_scale

    return reward_fn


def make_differentiable_reward(
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    sizes_array: jax.Array,
    cell_size: float,
    params: EnvParams,
    density_weight: float = 1.0,
    target_density: float = 1.0,
    bounds_weight: float = 1.0,
    gamma: float = 1.0,
    dense: bool = True,
    reward_scale: float = 1.0,
) -> RewardFn:
    """-(smoothed wirelength + density overflow + out-of-bounds) - the objective SHAC can descend.

    The two halves `docs/Action_Space_Decision.md` says an analytic-gradient method needs, in one
    RewardFn:

      * **wirelength that every macro feels.** Raw HPWL is a sum of per-net `max - min`, so only
        the pins ON a net's bounding box receive gradient - 76% of adaptec1's connected macros got
        exactly zero. The log-sum-exp surrogate gives 100% coverage for 1% fidelity (measured; see
        `extras/rewards.smoothed_wirelength`).
      * **legality that has a gradient at all.** Under the discrete spaces legality is a MASK, and
        masks are comparisons - `d(overlap)/d(position)` is identically zero. A continuous action
        cannot be masked, so overlap and the canvas edge become costs here instead
        (`extras/density.py`).

    The weights are NOT auto-balanced, for the same reason `make_expert_reward`'s are not: the
    wirelength term is in real design units and the density term in grid bins, so
    `density_weight` has to be sized against the wirelength term's actual magnitude on YOUR
    benchmark. `scripts/measure_reward_terms.py` prints both.

    **`dense=True` by default, unlike every other reward here.** A short-horizon method
    differentiates a window of steps; a reward that pays out only at the end gives the first steps
    of that window nothing to differentiate.
    """
    wirelength = make_scaled_smoothed_reward(
        padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size,
        dense=dense, reward_scale=1.0, gamma=gamma,
    )
    legality_cost = make_density_cost(
        sizes_array, params, cell_size, target_density=target_density,
        bounds_weight=bounds_weight,
    )

    def reward_fn(
        old_positions: jax.Array, new_positions: jax.Array, old_placed: jax.Array,
        new_placed: jax.Array, orientations: jax.Array | None = None,
    ) -> jax.Array:
        reward = wirelength(old_positions, new_positions, old_placed, new_placed, orientations)
        if density_weight == 0.0:
            return reward * reward_scale
        sizes = effective_sizes(sizes_array, orientations)
        # Legality is a property of a LAYOUT, not of a step, so the dense form pays its delta -
        # which telescopes over an episode to the same total the sparse form pays once, exactly as
        # the wirelength and congestion terms do.
        new_cost = legality_cost(to_real_centers(new_positions, sizes, cell_size), new_placed)
        if dense:
            old_cost = legality_cost(to_real_centers(old_positions, sizes, cell_size), old_placed)
            penalty = new_cost - old_cost
        else:
            penalty = jnp.where(new_placed.all(), new_cost, 0.0)
        return (reward - density_weight * penalty) * reward_scale

    return reward_fn


def make_expert_reward(
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    sizes_array: jax.Array,
    cell_size: float,
    params: EnvParams,
    dense: bool = True,
    reward_scale: float = 1.0,
    regularity_weight: float = 0.0,
    regularity_mode: str = "corner",
) -> RewardFn:
    """Adds EXPlace's regularity (periphery) term to the -HPWL reward: `hpwl_reward - w * normalized_cost`.

    The regularity cost is normalized to [0, 1] by regularity_max() for the macro just placed, so
    `regularity_weight` is directly comparable across macros of different sizes. Note this deviates
    from EXPlace, which instead rescales by running per-episode min/max - a stateful scheme placax's
    pure RewardFn has nowhere to keep, and one that makes the reward non-stationary early in an
    episode. regularity_weight=0.0 reproduces make_scaled_hpwl_reward exactly.

    The two terms are NOT auto-balanced: the HPWL term keeps whatever units `reward_scale` leaves
    it in, while the regularity term is always [0, 1] * regularity_weight. Pick regularity_weight
    against the HPWL term's actual per-step magnitude for your benchmark, not against EXPlace's
    published coefficients - those are ratios between six terms that it has already normalized
    to a common scale, so they do not transfer directly.
    """
    hpwl_reward_fn = make_scaled_hpwl_reward(
        padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size,
        dense=dense, reward_scale=reward_scale,
    )

    def reward_fn(
        old_positions: jax.Array, new_positions: jax.Array, old_placed: jax.Array,
        new_placed: jax.Array, orientations: jax.Array | None = None,
    ) -> jax.Array:
        reward = hpwl_reward_fn(
            old_positions, new_positions, old_placed, new_placed, orientations
        )
        if regularity_weight == 0.0:
            return reward
        # Recover which macro this step placed, and where, straight from the RewardFn signature -
        # exactly one macro flips from unplaced to placed per step(), so no wider signature is needed.
        newly_placed = new_placed & ~old_placed
        idx = jnp.argmax(newly_placed)
        # The footprint AS PLACED: a turned macro reaches a different distance from the periphery.
        macro_size = to_grid_units(effective_sizes(sizes_array, orientations)[idx], cell_size)
        cost = regularity_cost(params, macro_size, new_positions[idx], cell_size, regularity_mode)
        # A macro too big to leave any interior has an all-zero cost map; guard that 0/0.
        scale = regularity_max(params, macro_size, cell_size, regularity_mode)
        normalized = jnp.where(scale > 0, cost / jnp.where(scale > 0, scale, 1.0), 0.0)
        # Nothing placed (never happens via step(), but keeps the term well-defined) costs nothing.
        return reward - regularity_weight * jnp.where(newly_placed.any(), normalized, 0.0)

    return reward_fn
