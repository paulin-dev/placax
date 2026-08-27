"""Turns a policy's raw logits into a legal action."""
from placax.extras.masks import boundary_mask, occupancy_mask, quality_mask  # must precede jax imports
from placax.types import EnvParams
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


def make_wiremask_quality_illegal(margin: float, wiremask_key: str = "wiremask") -> ExtraIllegalFn:
    """Builds an ExtraIllegalFn that rules out cells whose wiremask exceeds wiremask.min() + margin."""

    def extra_illegal_fn(obs: dict) -> jax.Array:
        wiremask = obs[wiremask_key]
        return quality_mask(wiremask, wiremask.min() + margin)

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
