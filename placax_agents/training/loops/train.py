"""Sequential training: one episode, one gradient update, repeated.
Ties rollout collection, GAE, the PPO loss, and an optax optimizer
together. See parallel_train.py for the batched (n_envs at once)
version - the rollout/loss computation genuinely differs (vmapped or
not - a real, measured ~28% overhead even at n_envs=1, not just
historical duplication), but the optimizer-update tail is identical and
shared via optimizer_step.py."""
import pathlib

from placax.types import EnvParams, RewardFn  # noqa: F401  must precede jax imports
from placax_agents.ops.checkpoint import load_checkpoint, save_checkpoint  # noqa: F401
from placax_agents.training.algorithm.config import PPOConfig  # noqa: F401
from placax_agents.training.algorithm.gae import compute_gae  # noqa: F401
from placax_agents.training.algorithm.loss import ppo_loss  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.training.algorithm.optimizer_step import apply_gradient_update  # noqa: F401
from placax_agents.training.rollout import collect_rollout  # noqa: F401
from placax_agents.training.algorithm.running_stats import RunningStats, init_running_stats  # noqa: F401
from placax_agents.types import AlgorithmFn, StateFn  # noqa: F401

import jax
import jax.numpy as jnp
import optax


def train_step(
    key: jax.Array,
    variables,
    opt_state,
    running_stats: RunningStats,
    optimizer: optax.GradientTransformation,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    reward_fn: RewardFn,
    sizes_array: jax.Array,
    cell_size: float,
    state_fn: StateFn = observation,
    ppo_config: PPOConfig = PPOConfig(),
):
    """One full episode + one gradient update. Returns (variables,
    opt_state, running_stats, loss, final_state) - loss returned for
    logging, not used internally beyond this call."""
    trajectory, final_state = collect_rollout(
        key, variables, policy_apply_fn, params, reward_fn, sizes_array, cell_size, state_fn
    )
    advantages, returns = compute_gae(
        trajectory["reward"], trajectory["value"], trajectory["done"], next_value=jnp.array(0.0),
        gamma=ppo_config.gamma, lam=ppo_config.lam,
    )

    def loss_fn(policy_params, normalized_advantages, normalized_returns):
        return ppo_loss(
            policy_params, policy_apply_fn, trajectory, normalized_advantages, normalized_returns,
            sizes_array, cell_size, params,
            clip_eps=ppo_config.clip_eps, value_coef=ppo_config.value_coef,
            entropy_coef=ppo_config.entropy_coef,
        )

    new_variables, new_opt_state, new_running_stats, loss = apply_gradient_update(
        variables, opt_state, running_stats, optimizer, loss_fn, advantages, returns
    )
    return new_variables, new_opt_state, new_running_stats, loss, final_state


def _build_jitted_train_step():
    """Builds the jitted train_step function - callers should build this
    once and reuse it across many calls rather than rebuilding it every
    call, since each distinct jax.jit(...) wrapper object pays its own
    one-time compilation cost on first use. Kept as its own function so
    that cost is paid exactly once per training run, not once per
    caller that happens to need a jitted train_step."""
    return jax.jit(
        train_step,
        static_argnames=("optimizer", "policy_apply_fn", "reward_fn", "state_fn", "ppo_config"),
    )


def _train_n_steps(
    jitted_train_step,
    key: jax.Array,
    variables,
    opt_state,
    running_stats: RunningStats,
    optimizer: optax.GradientTransformation,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    reward_fn: RewardFn,
    sizes_array: jax.Array,
    cell_size: float,
    n_iterations: int,
    state_fn: StateFn,
    ppo_config: PPOConfig,
):
    """Runs n_iterations of an ALREADY-BUILT jitted_train_step (from
    _build_jitted_train_step(), built once by the caller and reused -
    never rebuilt here) from a GIVEN starting opt_state/running_stats
    (not always freshly initialized), returning the FULL final state -
    (variables, opt_state, running_stats, key, losses). The shared
    primitive train_sequential (which discards opt_state/running_stats
    for a simple public API) and ops.resumable_train (which needs them
    preserved to actually resume) both build on, instead of each
    keeping its own copy of this loop."""
    losses = []
    for _ in range(n_iterations):
        key, rollout_key = jax.random.split(key)
        variables, opt_state, running_stats, loss, _final_state = jitted_train_step(
            rollout_key, variables, opt_state, running_stats, optimizer, policy_apply_fn,
            params, reward_fn, sizes_array, cell_size, state_fn, ppo_config,
        )
        losses.append(float(loss))

    return variables, opt_state, running_stats, key, losses


def train_sequential(
    key: jax.Array,
    variables,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    reward_fn: RewardFn,
    sizes_array: jax.Array,
    cell_size: float,
    n_iterations: int,
    learning_rate: float = 3e-4,
    state_fn: StateFn = observation,
    optimizer: optax.GradientTransformation | None = None,
    ppo_config: PPOConfig = PPOConfig(),
    checkpoint_path: pathlib.Path | None = None,
):
    """Runs n_iterations of train_step. Returns (final_variables, losses).
    Called train_sequential, not train: train() itself (run.py) is the
    unified entry point that dispatches between this and
    parallel_train.train_parallel - this is the underlying
    implementation, still callable directly if you want to force
    sequential specifically.

    optimizer defaults to optax.adam(learning_rate) if not given - any
    optax.GradientTransformation works (SGD, gradient clipping chains,
    a schedule, etc.), the same swappable pattern as reward_fn/
    policy_apply_fn/state_fn. ppo_config bundles gamma/lam/clip_eps/
    value_coef/entropy_coef - found by auditing that these were
    correctly parameterized in compute_gae/ppo_loss themselves but
    never actually reachable from here.

    checkpoint_path, if given, saves the full training state (weights,
    optimizer state, running stats, PRNG key) after every iteration,
    and resumes from it automatically if it already exists - the simple
    "save progress every lap" case. For periodic (not every-iteration)
    checkpointing, evaluation snapshots, or a persisted log, use
    ops.resumable_train instead."""
    if optimizer is None:
        optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(variables)
    running_stats = init_running_stats()
    start_iteration = 0

    if checkpoint_path is not None:
        template = {
            "variables": variables, "opt_state": opt_state, "running_stats": running_stats,
            "iteration": jnp.array(0), "key": key,
        }
        if checkpoint_path.exists():
            state = load_checkpoint(template, checkpoint_path)
            variables, opt_state, running_stats, key = (
                state["variables"], state["opt_state"], state["running_stats"], state["key"]
            )
            start_iteration = int(state["iteration"])

    jitted_train_step = _build_jitted_train_step()
    losses = []
    for i in range(n_iterations):
        variables, opt_state, running_stats, key, step_losses = _train_n_steps(
            jitted_train_step, key, variables, opt_state, running_stats, optimizer,
            policy_apply_fn, params, reward_fn, sizes_array, cell_size, 1, state_fn, ppo_config,
        )
        losses.append(step_losses[0])

        if checkpoint_path is not None:
            bundle = {
                "variables": variables, "opt_state": opt_state, "running_stats": running_stats,
                "iteration": jnp.array(start_iteration + i + 1), "key": key,
            }
            save_checkpoint(bundle, checkpoint_path)

    return variables, losses
