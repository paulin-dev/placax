"""Runs one full episode as a single lax.scan."""
from placax.action_space import DISCRETE_GRID  # must precede jax imports
from placax.core import reset, step
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
    initial_positions: jax.Array | None = None,
    n_placed: int = 0,
    action_space=DISCRETE_GRID,
):
    """Samples one full episode, returning (trajectory, final_state) with per-step obs/action/reward/log_prob/value/done arrays.

    `action_space` is the environment's, not the loop's: what an action means and how long an
    episode runs belong to the task every agent is solving. It used to be absent here, which made
    the whole PPO path silently `discrete_grid`-only however a config was written - so the one
    axis this project generalized the kernel for could never be studied with the learner.

    `initial_positions`/`n_placed` are the environment's warm start: a prefix of macros already
    placed, and how many. The episode is then the REMAINING placements, so a warm-started run is
    a shorter episode rather than one that scans past the end of the position array and silently
    drops its last updates. `n_placed` is static (it comes from the config, resolved once per
    run) precisely so that length is a compile-time shape.
    """
    initial_state = reset(params, initial_positions, action_space)

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

        new_state, reward, done = step(state, action, reward_fn, params, action_space)

        transition = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "log_prob": log_prob,
            "value": value,
            "done": done,
        }
        return new_state, transition

    # The space's own answer, not an assumed macro count: a perturbation episode is a move
    # budget and has nothing to do with how many macros the design has.
    step_keys = jax.random.split(key, action_space.episode_length(params, n_placed))
    final_state, trajectory = jax.lax.scan(scan_step, initial_state, step_keys)
    return trajectory, final_state
