"""Turns a policy's raw logits into a legal action."""
from placax.extras.masks import boundary_mask, occupancy_mask, quality_mask  # must precede jax imports
from placax.types import EnvParams
from placax_agents.policy.scale import to_grid_units
from placax_agents.types import ExtraIllegalFn

import jax
import jax.numpy as jnp


def illegal_cells(
    occupied: jax.Array,
    params: EnvParams,
    macro_size: tuple[int, int],
    extra_illegal: jax.Array | None = None,
) -> jax.Array:
    """The (grid_x, grid_y) bool map of cells this macro may not be placed in.

    THE one definition of legality in this project. Every agent reads it - the policy through
    `legal_action_logits` below, the non-learning baselines directly - because a baseline that
    computed its own legality would be running in a different environment while sharing the
    config that claims otherwise, which is exactly the incomparability the experiment machinery
    exists to prevent.

    Note the safety valve at the end: if a placement is impossible under the full rule set, the
    extra quality rule is dropped, and if it is still impossible, legality itself is. That second
    relaxation emits an overlapping macro, silently and by design - the alternative is an episode
    that cannot terminate. Its consequence is measurable rather than assumed: every evaluated
    placement is scored by `placax.extras.legality`, so a run that relaxed reports the overlap it
    caused instead of a quietly better HPWL.
    """
    # A cell is illegal if placing the macro there would overlap or go out of bounds.
    base_illegal = occupancy_mask(occupied, macro_size) | boundary_mask(params, macro_size)
    illegal = base_illegal | extra_illegal if extra_illegal is not None else base_illegal
    # Safety valve: relax to just physical constraints if the extra rule leaves zero legal cells.
    illegal = jnp.where(illegal.all(), base_illegal, illegal)
    return jnp.where(illegal.all(), False, illegal)


def legal_action_logits(
    logits: jax.Array,
    occupied: jax.Array,
    params: EnvParams,
    macro_size: tuple[int, int],
    extra_illegal: jax.Array | None = None,
) -> jax.Array:
    """Sets illegal cells' logits to -inf so they're never sampled/argmax'd."""
    illegal = illegal_cells(occupied, params, macro_size, extra_illegal)
    # Widen to float64 here, matching MaskPlace's own `x.double()` right before its softmax:
    # the CNN itself stays float32, but softmax/log_softmax downstream of this function need
    # the extra dynamic range so growing logit spread doesn't fully saturate (and zero out
    # PPO's gradient) as early in training as float32 alone would allow.
    return jnp.where(illegal, -jnp.inf, logits.astype(jnp.float64))


def make_wiremask_quality_illegal(
    margin: float, cell_size: float, wiremask_key: str = "wiremask", lookahead_key: str = "lookahead_wiremasks"
) -> ExtraIllegalFn:
    """Builds an ExtraIllegalFn ruling out cells whose normalized wiremask exceeds the legal minimum plus margin, matching MaskPlace's own soft-coefficient behavior."""

    def extra_illegal_fn(obs: dict) -> jax.Array:
        lookahead = obs.get(lookahead_key)
        # Falls back to the lookahead map's current-macro slice when there's no separate wiremask_key entry.
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
    """Log probability of `action` under logits, used at rollout time and for PPO's ratio."""
    grid = logits.shape[1]
    flat_idx = action[0] * grid + action[1]
    return jax.nn.log_softmax(logits.ravel())[flat_idx]
