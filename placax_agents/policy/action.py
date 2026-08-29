"""Turns a policy's raw logits into a legal action."""
from placax.extras.masks import boundary_mask, occupancy_mask, quality_mask  # must precede jax imports
from placax.types import EnvParams
from placax_agents.policy.scale import to_grid_units
from placax_agents.types import ExtraIllegalFn

import jax
import jax.numpy as jnp


def legal_action_logits(
    logits: jax.Array,
    occupied: jax.Array,
    params: EnvParams,
    macro_size: tuple[int, int],
    extra_illegal: jax.Array | None = None,
) -> jax.Array:
    """Sets illegal cells' logits to -inf so they're never sampled/argmax'd."""
    # A cell is illegal if placing the macro there would overlap something or go out of
    # bounds - these are hard physical constraints, never relaxed below.
    base_illegal = occupancy_mask(occupied, macro_size) | boundary_mask(params, macro_size)
    illegal = base_illegal | extra_illegal if extra_illegal is not None else base_illegal
    # Safety valve: if the extra rule (e.g. wiremask quality) alone leaves zero legal
    # cells, relax back to just the physical constraints rather than allowing genuinely
    # illegal (overlapping/out-of-bounds) placements. Only if literally no cell is
    # physically legal either (grid fully occupied) do we bail out to no mask at all,
    # purely so log_softmax over all -inf doesn't produce NaNs.
    illegal = jnp.where(illegal.all(), base_illegal, illegal)
    illegal = jnp.where(illegal.all(), False, illegal)
    return jnp.where(illegal, -jnp.inf, logits)


def make_wiremask_quality_illegal(
    margin: float, cell_size: float, wiremask_key: str = "wiremask", lookahead_key: str = "lookahead_wiremasks"
) -> ExtraIllegalFn:
    """Builds an ExtraIllegalFn that rules out cells whose wiremask exceeds wiremask.min() + margin.

    Matches MaskPlace's own PPO2.py/place_env.py: margin (its --soft_coefficient) is applied
    after normalizing the wiremask into roughly [0, 1] by dividing by the larger of its own max
    and the next-macro lookahead map's max - not against the wiremask's raw HPWL-delta scale,
    which would make margin's usual ~1.0 default far too tight (near-zero tolerance instead of
    the intended soft one) since raw values run into the thousands for real netlists.

    MaskPlace's reference also excludes already-illegal (occupied/out-of-bounds) cells from that
    minimum - its Actor.forward adds a large constant to occupied cells before taking `.min()`
    (`net_img + mask * 10`), so an occupied cell's cheap-looking wiremask value can never win and
    silently lower the threshold below what's actually achievable at a legal cell. Reproduced
    here with +inf instead of a scale-dependent constant, using this same macro's own
    occupancy/boundary illegality (the same computation legal_action_logits does, derived
    independently since this function only ever sees `obs`, not legal_action_logits's own
    base_illegal). This exclusion is a MaskPlace-specific quirk - it isn't described in the
    MaskPlace paper itself, and no other placement-RL wiremask/position-mask scheme found in a
    broader search does this - so it lives here, not in the generic occupancy_mask/
    boundary_mask/quality_mask primitives in placax.extras.masks, which stay reusable as-is for
    any environment that wants a plain "cells above a cutoff" filter.
    """

    def extra_illegal_fn(obs: dict) -> jax.Array:
        lookahead = obs.get(lookahead_key)
        # Falls back to the lookahead map's own current-macro slice when there's no separate
        # wiremask_key entry - make_wiremask_observation deliberately doesn't buffer that as a
        # redundant extra channel (it's identical to lookahead_wiremasks[0]) since it would
        # otherwise double a full grid-resolution array's storage in every buffered trajectory.
        wiremask = obs[wiremask_key] if wiremask_key in obs else lookahead[0]
        next_wiremask = lookahead[1] if lookahead is not None and lookahead.shape[0] > 1 else wiremask
        scale = jnp.maximum(wiremask.max(), next_wiremask.max())
        normalized = jnp.where(scale > 0, wiremask / jnp.where(scale > 0, scale, 1.0), wiremask)

        grid_x, grid_y = obs["canvas"].shape
        params = EnvParams(grid=grid_x, grid_y=grid_y)
        macro_size = to_grid_units(obs["current_macro_size"], cell_size)
        position_illegal = occupancy_mask(obs["canvas"], macro_size) | boundary_mask(params, macro_size)
        min_source = jnp.where(position_illegal, jnp.inf, normalized)

        return quality_mask(normalized, min_source.min() + margin)

    return extra_illegal_fn


def sample_action(key: jax.Array, logits: jax.Array) -> jax.Array:
    """Samples one (x, y) from a (grid_x, grid_y) logits map."""
    grid = logits.shape[1]
    flat_idx = jax.random.categorical(key, logits.ravel())
    return jnp.array([flat_idx // grid, flat_idx % grid])


def action_log_prob(logits: jax.Array, action: jax.Array) -> jax.Array:
    """Log probability of `action` under logits - used at rollout time
    and recomputed under updated params for PPO's ratio."""
    grid = logits.shape[1]
    flat_idx = action[0] * grid + action[1]
    return jax.nn.log_softmax(logits.ravel())[flat_idx]
