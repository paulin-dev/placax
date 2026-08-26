"""Buffer + minibatch-epoch PPO training, matching MaskPlace's own PPO2.py procedure."""
import pathlib

from placax.types import EnvParams, RewardFn  # must precede jax imports
from placax_agents.policy.observation import observation
from placax_agents.training.algorithm.config import PPOConfig
from placax_agents.training.algorithm.gae import compute_gae
from placax_agents.training.algorithm.loss import ppo_loss
from placax_agents.training.algorithm.optimizer_step import apply_gradient_update
from placax_agents.training.algorithm.running_stats import RunningStats
from placax_agents.training.loops.common import checkpoint_every_n, make_step_input, open_train_state
from placax_agents.training.rollout import collect_rollout
from placax_agents.types import AlgorithmFn, ExtraIllegalFn, StateFn

import jax
import jax.numpy as jnp
import optax


def collect_buffer(
    key: jax.Array,
    variables,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    reward_fn: RewardFn,
    sizes_array: jax.Array,
    cell_size: float,
    n_episodes: int,
    state_fn: StateFn = observation,
    extra_illegal_fn: ExtraIllegalFn | None = None,
):
    """Collects n_episodes of rollout (vmapped) and flattens them into one buffer of transitions."""
    # 1. One fresh random key per episode, then run all episodes at once via vmap
    #    instead of a slow Python loop.
    keys = jax.random.split(key, n_episodes)
    trajectories, _ = jax.vmap(
        collect_rollout, in_axes=(0, None, None, None, None, None, None, None, None)
    )(keys, variables, policy_apply_fn, params, reward_fn, sizes_array, cell_size, state_fn, extra_illegal_fn)
    # 2. Collapse the (n_episodes, n_macros, ...) batch dimensions into one flat buffer
    #    axis; episodes stay in temporal order internally, which is what compute_gae
    #    needs since it relies on each episode's done flag to reset advantages.
    flatten = lambda x: x.reshape((-1,) + x.shape[2:])  # noqa: E731
    return jax.tree_util.tree_map(flatten, trajectories)


# Built once at import: without this, the vmapped rollout above (which runs the entire
# policy network, backbone included, n_episodes times) gets retraced and recompiled from
# scratch on every call instead of being cached - both far slower than it should be, and,
# on tightly memory-constrained hardware, a source of GPU memory growth every iteration
# until the process OOMs.
_jitted_collect_buffer = jax.jit(
    collect_buffer, static_argnames=("policy_apply_fn", "reward_fn", "state_fn", "extra_illegal_fn", "n_episodes")
)

# Same reasoning as _jitted_collect_buffer above: without this, compute_gae's small scan
# also recompiles from scratch on every call instead of being cached.
_jitted_compute_gae = jax.jit(compute_gae)


def _minibatch_update(
    variables, opt_state, running_stats, optimizer, policy_apply_fn, params,
    batch_trajectory, batch_advantages, batch_returns, cell_size, ppo_config, extra_illegal_fn,
):
    """One gradient step over one minibatch slice of an already-built buffer."""

    def loss_fn(policy_params, normalized_advantages, normalized_returns):
        return ppo_loss(
            policy_params, policy_apply_fn, batch_trajectory, normalized_advantages, normalized_returns,
            cell_size, params,
            clip_eps=ppo_config.clip_eps, value_coef=ppo_config.value_coef,
            entropy_coef=ppo_config.entropy_coef, value_loss_fn=ppo_config.value_loss_fn,
            extra_illegal_fn=extra_illegal_fn,
        )

    return apply_gradient_update(
        variables, opt_state, running_stats, optimizer, loss_fn, batch_advantages, batch_returns
    )


_jitted_minibatch_update = jax.jit(
    _minibatch_update, static_argnames=("optimizer", "policy_apply_fn", "ppo_config", "extra_illegal_fn")
)


def _shuffled_batches(key: jax.Array, buffer_size: int, batch_size: int) -> jax.Array:
    """Returns (n_batches, batch_size) shuffled buffer indices, dropping any short remainder."""
    # Shuffle all buffer indices, then chop off any leftover that doesn't fill a full
    # batch (matches MaskPlace's BatchSampler(drop_last=True)) before reshaping into batches.
    n_batches = buffer_size // batch_size
    perm = jax.random.permutation(key, buffer_size)
    return perm[: n_batches * batch_size].reshape(n_batches, batch_size)


def buffered_train_step(
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
    n_episodes: int,
    ppo_epochs: int,
    batch_size: int,
    state_fn: StateFn = observation,
    ppo_config: PPOConfig = PPOConfig(),
    extra_illegal_fn: ExtraIllegalFn | None = None,
):
    """One full buffer-collect + multi-epoch-minibatch update cycle, returning updated
    (variables, opt_state, running_stats, last_minibatch_loss)."""
    # 1. Fill the buffer with n_episodes of fresh rollout data using the current policy.
    key, buffer_key = jax.random.split(key)
    buffer = _jitted_collect_buffer(
        buffer_key, variables, policy_apply_fn, params, reward_fn, sizes_array, cell_size, n_episodes,
        state_fn, extra_illegal_fn,
    )
    # 2. Compute advantages/returns once for the whole buffer - cheaper than per-minibatch,
    #    and valid because GAE resets naturally at each episode's done flag.
    advantages, returns = _jitted_compute_gae(
        buffer["reward"], buffer["value"], buffer["done"], next_value=jnp.array(0.0),
        gamma=ppo_config.gamma, lam=ppo_config.lam,
    )

    # 3. Re-use this same buffer for several epochs, each time reshuffling into fresh
    #    minibatches, so every transition contributes to several gradient steps.
    loss = jnp.array(0.0)
    for _epoch in range(ppo_epochs):
        key, shuffle_key = jax.random.split(key)
        for batch_idx in _shuffled_batches(shuffle_key, advantages.shape[0], batch_size):
            batch_trajectory = jax.tree_util.tree_map(lambda x: x[batch_idx], buffer)
            variables, opt_state, running_stats, loss = _jitted_minibatch_update(
                variables, opt_state, running_stats, optimizer, policy_apply_fn, params,
                batch_trajectory, advantages[batch_idx], returns[batch_idx], cell_size, ppo_config,
                extra_illegal_fn,
            )
    return variables, opt_state, running_stats, loss


def train_buffered(
    key: jax.Array,
    variables,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    reward_fn: RewardFn,
    sizes_array: jax.Array,
    cell_size: float,
    n_iterations: int,
    n_episodes: int = 10,
    ppo_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    state_fn: StateFn = observation,
    optimizer: optax.GradientTransformation | None = None,
    ppo_config: PPOConfig = PPOConfig(),
    checkpoint_path: pathlib.Path | None = None,
    extra_illegal_fn: ExtraIllegalFn | None = None,
):
    """Runs n_iterations of buffered_train_step and returns (final_variables, losses)."""
    # Default optimizer, and either a fresh training state or one resumed from checkpoint_path.
    if optimizer is None:
        optimizer = optax.adam(learning_rate)
    variables, opt_state, running_stats, key, start_iteration = open_train_state(
        variables, key, optimizer, checkpoint_path
    )

    losses = []
    for i in range(n_iterations):
        # Each iteration: fresh rollout buffer, several epochs of minibatch updates over
        # it, then checkpoint so a crash never loses more than one iteration of progress.
        key, step_key = make_step_input(key)
        variables, opt_state, running_stats, loss = buffered_train_step(
            step_key, variables, opt_state, running_stats, optimizer, policy_apply_fn, params,
            reward_fn, sizes_array, cell_size, n_episodes, ppo_epochs, batch_size, state_fn, ppo_config,
            extra_illegal_fn,
        )
        losses.append(float(loss))
        checkpoint_every_n(checkpoint_path, 1, start_iteration + i + 1, variables, opt_state, running_stats, key)

    return variables, losses
