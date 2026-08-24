"""Evaluates a policy's actual placement quality - greedy (argmax, no
sampling noise), reporting the real hpwl(), not the training loss.
Training loss is a normalized, PPO-internal number; this is the number
that's actually comparable to a published result."""
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
    """Places every macro greedily (highest-logit legal cell, not
    sampled), then returns (final_positions, real_hpwl) - real_hpwl
    computed on real-unit centers (via to_real_centers), not raw grid
    indices directly: hpwl() mixing tiny grid indices with real-unit
    offsets was a confirmed bug (see scale.py, reward.py).

    state_fn defaults to observation(), swappable the same way as
    collect_rollout - whatever state_fn a policy was trained against
    should be the same one used to evaluate it."""
    state = reset(params)

    def scan_step(state, macro_idx):
        obs = state_fn(state, params, sizes_array)
        logits, _value = policy_apply_fn(variables, obs)
        macro_size = to_grid_units(obs["current_macro_size"], cell_size)
        masked_logits = legal_action_logits(logits, obs["canvas"], params, macro_size)

        flat_idx = jnp.argmax(masked_logits.ravel())
        grid_y = masked_logits.shape[1]
        action = jnp.array([flat_idx // grid_y, flat_idx % grid_y])

        idx = state.step
        new_positions = state.positions.at[idx].set(action)
        new_state = state.replace(positions=new_positions, step=idx + 1)
        return new_state, None

    final_state, _ = jax.lax.scan(scan_step, state, jnp.arange(params.n_macros))

    real_centers = to_real_centers(final_state.positions, sizes_array, cell_size)
    real_hpwl = hpwl(real_centers, padded_pin_idx, padded_pin_offset, valid_mask)
    return final_state.positions, real_hpwl
