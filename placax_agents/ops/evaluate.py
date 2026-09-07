"""Evaluates a policy's placement quality via a greedy (argmax) rollout, reporting real HPWL."""
from placax.core import reset, step  # must precede jax imports
from placax.extras.rewards import hpwl
from placax.types import EnvParams
from placax_agents.policy.action import legal_action_logits
from placax_agents.policy.observation import observation
from placax_agents.policy.scale import to_grid_units, to_real_centers
from placax_agents.types import AlgorithmFn, ExtraIllegalFn, StateFn

import jax
import jax.numpy as jnp


def _no_reward(_old_positions, _new_positions, _old_placed, _new_placed) -> jax.Array:
    """The evaluation rollout scores its own final placement, so per-step reward is unused.

    Passed explicitly rather than letting evaluate() reimplement the state transition inline:
    `step()` is the one place a macro gets placed, and a second copy of that line here is how the
    "one shared kernel" property quietly stops being true. See placax/core.py.
    """
    return jnp.array(0.0)


def evaluate(
    variables,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    sizes_array: jax.Array,
    cell_size: float,
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    state_fn: StateFn = observation,
    extra_illegal_fn: ExtraIllegalFn | None = None,
    initial_positions: jax.Array | None = None,
    n_placed: int = 0,
):
    """Places every remaining macro greedily (argmax over legal cells) and returns (final_positions, real_hpwl)."""
    state = reset(params, initial_positions)

    def scan_step(state, _macro_idx):
        # 1. Ask the policy for action scores at this state (we don't need the value estimate here).
        # The bare `observation` default takes cell_size as a keyword the plain StateFn signature
        # doesn't carry - bind it here from what evaluate() was already given, rather than silently
        # falling back to observation's own cell_size=1.0 default (wrong for any real benchmark; a
        # custom state_fn, e.g. make_wiremask_observation's closure, already binds its own cell_size
        # and is called exactly as given).
        obs = observation(state, params, sizes_array, cell_size=cell_size) if state_fn is observation \
            else state_fn(state, params, sizes_array)
        logits, _value = policy_apply_fn(variables, obs)

        # 2. Mask out illegal cells (occupied/out-of-bounds, plus any extra rule) before choosing.
        macro_size = to_grid_units(obs["current_macro_size"], cell_size)
        extra_illegal = extra_illegal_fn(obs) if extra_illegal_fn is not None else None
        masked_logits = legal_action_logits(logits, obs["canvas"], params, macro_size, extra_illegal)

        # 3. Greedily take the single best legal cell (no sampling, unlike training rollouts).
        flat_idx = jnp.argmax(masked_logits.ravel())
        grid_y = masked_logits.shape[1]
        action = jnp.array([flat_idx // grid_y, flat_idx % grid_y])

        # 4. Hand the action to the kernel - the same step() a training rollout drives.
        new_state, _reward, _done = step(state, action, _no_reward, params)
        return new_state, None

    # One scan step per macro still to place: a warm start shortens the episode rather than
    # scanning past the end of the position array.
    final_state, _ = jax.lax.scan(scan_step, state, jnp.arange(params.n_macros - n_placed))

    # Convert grid positions to real-unit centers to score the final layout with true HPWL.
    real_centers = to_real_centers(final_state.positions, sizes_array, cell_size)
    return final_state.positions, hpwl(real_centers, padded_pin_idx, padded_pin_offset, valid_mask)


# Built once at import to avoid retracing/recompiling on every call (same fix as buffered_train.py's jitted fns).
_jitted_evaluate = jax.jit(
    evaluate, static_argnames=("policy_apply_fn", "state_fn", "extra_illegal_fn", "n_placed")
)
