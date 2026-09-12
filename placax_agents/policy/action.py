"""Turns a policy's raw logits into a legal action."""
from placax.extras.masks import boundary_mask, occupancy_mask, quality_mask  # must precede jax imports
from placax.extras.orientation import N_ORIENTATIONS
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
    """Sets illegal cells' logits to -inf so they're never sampled/argmax'd.

    `logits` is a `(grid_x, grid_y)` map, matching the shape of the legality map itself. A space
    whose action carries a further axis - `oriented_grid`'s quarter turn - needs a policy that
    emits that axis AND a legality map per turn, because a turned macro has a different footprint
    and therefore fits in different cells. Broadcasting this two-dimensional map across a third
    axis would call those placements legal without checking, so a rank mismatch is refused here
    rather than silently accepted.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"legal_action_logits masks a (grid_x, grid_y) logits map, got shape {logits.shape}. "
            f"An action space with an extra axis needs legality computed FOR that axis - see "
            f"placax/extras/orientation.py's effective_sizes - not this map stretched over it."
        )
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

    # This rule is ABOUT the macro being placed next: its footprint, and a wiremask whose baseline
    # is "the macros before it are down". A space that names no such macro (perturbation) cannot
    # supply either, and build() refuses the pairing rather than masking for an arbitrary macro.
    extra_illegal_fn.needs_current_macro = True
    return extra_illegal_fn


def oriented_illegal_actions(
    obs: dict, params: EnvParams, cell_size: float, extra_illegal_fn: ExtraIllegalFn | None = None
) -> jax.Array:
    """`(grid_x, grid_y, N_ORIENTATIONS)` bool: which (cell, turn) pairs this macro may not take.

    Legality per TURN, because a quarter-turned macro is its height by its width and therefore
    fits in different cells. That is the whole reason `legal_action_logits` refuses to broadcast
    its two-dimensional map over a turn axis: stretching it would mark placements legal that were
    never checked, which is the failure this project keeps finding rather than a shortcut.

    The relaxation valve fires over the WHOLE action set rather than per turn. Per turn, a turn
    with nowhere legal to go would relax to "everywhere legal" and become the attractive option;
    across the set, the quality rule is dropped only when no (cell, turn) pair survives it at all,
    and legality itself only if that is still empty - the same two-stage valve `illegal_cells`
    applies, over the actions this space actually has.

    **One documented approximation.** An extra rule that reads a wiremask - `wiremask_quality` -
    is handed this macro's ROTATED footprint, so the part of it that asks "does this macro fit
    here" is right; the wiremask preview it reads is still computed from the unrotated geometry by
    `make_wiremask_observation`, one level up. That makes the quality threshold slightly off for a
    turned macro, in the direction of ranking cells rather than of admitting illegal ones. Stated
    here because a legality rule is exactly where an unstated approximation does damage.
    """
    size = obs["current_macro_size"]

    def per_turn(turn: jax.Array) -> tuple[jax.Array, jax.Array]:
        # A quarter turn (WEST or EAST) swaps width and height; a half turn does not.
        turned = jnp.where(turn % 2 == 1, size[::-1], size)
        macro_size = to_grid_units(turned, cell_size)
        base = occupancy_mask(obs["canvas"], macro_size) | boundary_mask(params, macro_size)
        if extra_illegal_fn is None:
            return base, base
        extra = extra_illegal_fn({**obs, "current_macro_size": turned})
        return base | extra, base

    full, base = jax.vmap(per_turn)(jnp.arange(N_ORIENTATIONS))
    full = jnp.where(full.all(), base, full)
    full = jnp.where(full.all(), False, full)
    # (n_turns, grid_x, grid_y) -> (grid_x, grid_y, n_turns), matching the action's own order.
    return jnp.moveaxis(full, 0, -1)


def masked_action_logits(
    logits: jax.Array, obs: dict, params: EnvParams, cell_size: float,
    extra_illegal_fn: ExtraIllegalFn | None = None,
) -> jax.Array:
    """A policy's logits with the run's legality applied, whatever action shape the policy emits.

    THE one place a logits map meets the environment's rules, called by the rollout, the greedy
    evaluation and PPO's loss alike. That last one is not a convenience: `ppo_loss` recomputes the
    mask from the stored observation to get the probability ratio right, so if the rollout and the
    loss built their masks from two copies of this logic, a change to one would make PPO train on
    a ratio between two different distributions - silently, and only under the space nobody tests.
    """
    if logits.ndim == 2:
        macro_size = to_grid_units(obs["current_macro_size"], cell_size)
        extra_illegal = extra_illegal_fn(obs) if extra_illegal_fn is not None else None
        return legal_action_logits(logits, obs["canvas"], params, macro_size, extra_illegal)
    if logits.ndim == 3:
        illegal = oriented_illegal_actions(obs, params, cell_size, extra_illegal_fn)
        # float64 for the same reason legal_action_logits widens - see there.
        return jnp.where(illegal, -jnp.inf, logits.astype(jnp.float64))
    raise ValueError(
        f"a policy emits either a (grid_x, grid_y) logits map or a (grid_x, grid_y, "
        f"{N_ORIENTATIONS}) one; got shape {logits.shape}. A wider action needs legality computed "
        f"for its extra axis, which is what oriented_illegal_actions does for the turn."
    )


def sample_action(key: jax.Array, logits: jax.Array) -> jax.Array:
    """Samples one action from a logits map of ANY rank - `(x, y)`, or `(x, y, turn)`.

    Unravelled rather than divided by `logits.shape[1]`, so the shape of an action is the shape
    of the logits a policy emits and nothing here has to know which space is in play. Identical
    to the old arithmetic for a two-dimensional map.
    """
    flat_idx = jax.random.categorical(key, logits.ravel())
    return jnp.stack(jnp.unravel_index(flat_idx, logits.shape))


def action_log_prob(logits: jax.Array, action: jax.Array) -> jax.Array:
    """Log probability of `action` under logits, used at rollout time and for PPO's ratio."""
    flat_idx = jnp.ravel_multi_index(
        tuple(action[i] for i in range(logits.ndim)), logits.shape, mode="clip"
    )
    return jax.nn.log_softmax(logits.ravel())[flat_idx]
