"""Evaluates a policy's placement quality: greedy (argmax) rollout,
reporting real HPWL."""
from placax.core import reset  # must precede jax imports
from placax.extras.rewards import hpwl
from placax.types import EnvParams
from placax_agents.policy.action import legal_action_logits
from placax_agents.policy.observation import observation
from placax_agents.policy.scale import to_grid_units, to_real_centers
from placax_agents.types import AlgorithmFn, ExtraIllegalFn, StateFn

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
    extra_illegal_fn: ExtraIllegalFn | None = None,
):
    """Places every macro greedily (argmax over legal cells) and returns (final_positions, real_hpwl)."""
    state = reset(params)

    def scan_step(state, _macro_idx):
        # 1. Ask the policy for action scores at this state (we don't need the value estimate here).
        obs = state_fn(state, params, sizes_array)
        logits, _value = policy_apply_fn(variables, obs)

        # 2. Mask out illegal cells (occupied/out-of-bounds, plus any extra rule) before choosing.
        macro_size = to_grid_units(obs["current_macro_size"], cell_size)
        extra_illegal = extra_illegal_fn(obs) if extra_illegal_fn is not None else None
        masked_logits = legal_action_logits(logits, obs["canvas"], params, macro_size, extra_illegal)

        # 3. Greedily take the single best legal cell (no sampling, unlike training rollouts).
        flat_idx = jnp.argmax(masked_logits.ravel())
        grid_y = masked_logits.shape[1]
        action = jnp.array([flat_idx // grid_y, flat_idx % grid_y])

        # 4. Record the placement and advance to the next macro.
        positions = state.positions.at[state.step].set(action)
        return state.replace(positions=positions, step=state.step + 1), None

    # Run scan_step once per macro to place the whole netlist.
    final_state, _ = jax.lax.scan(scan_step, state, jnp.arange(params.n_macros))

    # Convert grid positions to real-unit centers to score the final layout with true HPWL.
    real_centers = to_real_centers(final_state.positions, sizes_array, cell_size)
    return final_state.positions, hpwl(real_centers, padded_pin_idx, padded_pin_offset, valid_mask)
