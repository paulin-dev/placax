"""Runs one full episode as a single lax.scan."""
from placax.core import random_action, reset, step  # noqa: F401  must precede jax imports
from placax.types import EnvParams, RewardFn  # noqa: F401
from placax_agents.policy.action import action_log_prob, legal_action_logits, sample_action  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.policy.scale import to_grid_units  # noqa: F401
from placax_agents.types import AlgorithmFn, StateFn  # noqa: F401

import jax


def collect_rollout(
    key: jax.Array,
    variables,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    reward_fn: RewardFn,
    sizes_array: jax.Array,
    cell_size: float,
    state_fn: StateFn = observation,
):
    """Samples one full episode. Returns (trajectory, final_state);
    trajectory maps obs/action/reward/log_prob/value/done to arrays
    with a leading n_macros dimension. reward is 0 until placement."""
    initial_state = reset(params)

    def scan_step(state, step_key):
        # obs -> policy -> mask illegal cells -> sample -> apply -> record.
        obs = state_fn(state, params, sizes_array)
        logits, value = policy_apply_fn(variables, obs)

        macro_size = to_grid_units(obs["current_macro_size"], cell_size)
        masked_logits = legal_action_logits(logits, obs["canvas"], params, macro_size)

        action = sample_action(step_key, masked_logits)
        log_prob = action_log_prob(masked_logits, action)

        new_state, reward, done = step(state, action, reward_fn, params)

        transition = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "log_prob": log_prob,
            "value": value,
            "done": done,
        }
        return new_state, transition

    step_keys = jax.random.split(key, params.n_macros)
    final_state, trajectory = jax.lax.scan(scan_step, initial_state, step_keys)
    return trajectory, final_state
