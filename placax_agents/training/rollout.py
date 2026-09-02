"""Runs one full episode as a single lax.scan."""
from placax.core import reset, step  # must precede jax imports
from placax.types import EnvParams, RewardFn
from placax_agents.policy.action import action_log_prob, legal_action_logits, sample_action
from placax_agents.policy.observation import observation
from placax_agents.policy.scale import to_grid_units
from placax_agents.types import AlgorithmFn, ExtraIllegalFn, StateFn

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
    extra_illegal_fn: ExtraIllegalFn | None = None,
):
    """Samples one full episode, returning (trajectory, final_state) with per-step obs/action/reward/log_prob/value/done arrays."""
    initial_state = reset(params)

    def scan_step(state, step_key):
        # obs -> policy -> mask illegal cells -> sample -> apply -> record.
        # The bare `observation` default takes cell_size as a keyword the plain StateFn signature
        # doesn't carry - bind it here from what collect_rollout() was already given, rather than
        # silently falling back to observation's own cell_size=1.0 default (wrong for any real
        # benchmark; a custom state_fn, e.g. make_wiremask_observation's closure, already binds its
        # own cell_size and is called exactly as given).
        obs = observation(state, params, sizes_array, cell_size=cell_size) if state_fn is observation \
            else state_fn(state, params, sizes_array)
        logits, value = policy_apply_fn(variables, obs)

        macro_size = to_grid_units(obs["current_macro_size"], cell_size)
        extra_illegal = extra_illegal_fn(obs) if extra_illegal_fn is not None else None
        masked_logits = legal_action_logits(logits, obs["canvas"], params, macro_size, extra_illegal)

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
