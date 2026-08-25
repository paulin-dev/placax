"""Evaluates a policy's placement quality: greedy (argmax) rollout,
reporting real HPWL."""
from placax.core import reset  # noqa: F401  must precede jax imports
from placax.extras.rewards import hpwl  # noqa: F401
from placax.types import EnvParams, RewardFn  # noqa: F401
from placax_agents.policy.action import legal_action_logits  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.policy.scale import to_grid_units, to_real_centers  # noqa: F401
from placax_agents.types import AlgorithmFn, StateFn  # noqa: F401

import jax
import jax.numpy as jnp


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
):
    """Places every macro greedily. Returns (final_positions, real_hpwl)."""
    state = reset(params)

    def scan_step(state, _macro_idx):
        # legal_action_logits masks illegal cells; argmax picks the best legal one.
        obs = state_fn(state, params, sizes_array)
        logits, _value = policy_apply_fn(variables, obs)
        macro_size = to_grid_units(obs["current_macro_size"], cell_size)
        masked_logits = legal_action_logits(logits, obs["canvas"], params, macro_size)

        flat_idx = jnp.argmax(masked_logits.ravel())
        grid_y = masked_logits.shape[1]
        action = jnp.array([flat_idx // grid_y, flat_idx % grid_y])

        positions = state.positions.at[state.step].set(action)
        return state.replace(positions=positions, step=state.step + 1), None

    final_state, _ = jax.lax.scan(scan_step, state, jnp.arange(params.n_macros))

    real_centers = to_real_centers(final_state.positions, sizes_array, cell_size)
    return final_state.positions, hpwl(real_centers, padded_pin_idx, padded_pin_offset, valid_mask)
