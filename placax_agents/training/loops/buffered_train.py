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
    # 1. One fresh random key per episode, run all episodes at once via vmap.
    keys = jax.random.split(key, n_episodes)
    trajectories, _ = jax.vmap(
        collect_rollout, in_axes=(0, None, None, None, None, None, None, None, None)
    )(keys, variables, policy_apply_fn, params, reward_fn, sizes_array, cell_size, state_fn, extra_illegal_fn)
    # 2. Flatten (n_episodes, n_macros, ...) into one buffer axis, keeping episodes in temporal order for compute_gae.
    flatten = lambda x: x.reshape((-1,) + x.shape[2:])  # noqa: E731
    return jax.tree_util.tree_map(flatten, trajectories)


# Built once at import so the vmapped rollout isn't retraced/recompiled (and GPU memory doesn't grow) on every call.
_jitted_collect_buffer = jax.jit(
    collect_buffer, static_argnames=("policy_apply_fn", "reward_fn", "state_fn", "extra_illegal_fn", "n_episodes")
)

# Same reasoning as above; donate_argnums=(0, 1) lets XLA reuse the reward/value buffers for the (advantages, returns) output.
_jitted_compute_gae = jax.jit(compute_gae, donate_argnums=(0, 1))


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
        variables, opt_state, running_stats, optimizer, loss_fn, batch_advantages, batch_returns, ppo_config
    )


def _run_epochs(
    variables, opt_state, running_stats, optimizer, policy_apply_fn, params,
    policy_trajectory, advantages, returns, flat_batch_idx, cell_size, ppo_config, extra_illegal_fn,
):
    """Runs every (epoch, minibatch) gradient step as one jax.lax.scan over flat_batch_idx, letting XLA fuse gather and update into a single compiled program."""

    def step(carry, batch_idx):
        variables, opt_state, running_stats = carry
        batch_trajectory = jax.tree_util.tree_map(lambda x: x[batch_idx], policy_trajectory)
        variables, opt_state, running_stats, loss = _minibatch_update(
            variables, opt_state, running_stats, optimizer, policy_apply_fn, params,
            batch_trajectory, advantages[batch_idx], returns[batch_idx], cell_size, ppo_config,
            extra_illegal_fn,
        )
        return (variables, opt_state, running_stats), loss

    (variables, opt_state, running_stats), losses = jax.lax.scan(
        step, (variables, opt_state, running_stats), flat_batch_idx
    )
    return variables, opt_state, running_stats, losses[-1]


# donate_argnums=(0, 1, 2): lets XLA reuse variables/opt_state/running_stats memory for the scan's output instead of holding both copies.
# ppo_config is static since hyperparameters are fixed per run; a future annealing schedule would need to make those fields dynamic instead.
_jitted_run_epochs = jax.jit(
    _run_epochs, static_argnames=("optimizer", "policy_apply_fn", "ppo_config", "extra_illegal_fn"),
    donate_argnums=(0, 1, 2),
)


def _shuffled_batches(key: jax.Array, buffer_size: int, batch_size: int) -> jax.Array:
    """Returns (n_batches, batch_size) shuffled buffer indices, dropping any short remainder."""
    # Shuffle indices, drop any leftover remainder (matches MaskPlace's BatchSampler(drop_last=True)).
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
    """One full buffer-collect + multi-epoch-minibatch update cycle, returning updated (variables, opt_state, running_stats, loss)."""
    # 1. Fill the buffer with n_episodes of fresh rollout data using the current policy.
    key, buffer_key = jax.random.split(key)
    buffer = _jitted_collect_buffer(
        buffer_key, variables, policy_apply_fn, params, reward_fn, sizes_array, cell_size, n_episodes,
        state_fn, extra_illegal_fn,
    )
    # 2. Compute advantages/returns once for the whole buffer; valid since GAE resets at each episode's done flag.
    advantages, returns = _jitted_compute_gae(
        buffer["reward"], buffer["value"], buffer["done"], next_value=jnp.array(0.0),
        gamma=ppo_config.gamma, lam=ppo_config.lam,
    )

    # 3. Reuse the buffer for several epochs of reshuffled minibatches; skip in plain Python if the schedule is empty.
    n_batches = advantages.shape[0] // batch_size
    if ppo_epochs == 0 or n_batches == 0:
        return variables, opt_state, running_stats, jnp.array(0.0)

    # 4. Build every epoch's shuffled minibatch indices up front (one vmapped call), then flatten into one schedule.
    key, epochs_key = jax.random.split(key)
    epoch_keys = jax.random.split(epochs_key, ppo_epochs)
    per_epoch_batch_idx = jax.vmap(_shuffled_batches, in_axes=(0, None, None))(
        epoch_keys, advantages.shape[0], batch_size
    )
    flat_batch_idx = per_epoch_batch_idx.reshape((-1, batch_size))

    # Only the fields ppo_loss reads; buffer["reward"]/["value"]/["done"] were donated above and must not be touched again.
    policy_trajectory = {
        "obs": buffer["obs"], "action": buffer["action"], "log_prob": buffer["log_prob"],
    }
    variables, opt_state, running_stats, loss = _jitted_run_epochs(
        variables, opt_state, running_stats, optimizer, policy_apply_fn, params,
        policy_trajectory, advantages, returns, flat_batch_idx, cell_size, ppo_config, extra_illegal_fn,
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
        # Each iteration: fresh rollout buffer, several epochs of minibatch updates, then checkpoint.
        key, step_key = make_step_input(key)
        variables, opt_state, running_stats, loss = buffered_train_step(
            step_key, variables, opt_state, running_stats, optimizer, policy_apply_fn, params,
            reward_fn, sizes_array, cell_size, n_episodes, ppo_epochs, batch_size, state_fn, ppo_config,
            extra_illegal_fn,
        )
        losses.append(float(loss))
        checkpoint_every_n(checkpoint_path, 1, start_iteration + i + 1, variables, opt_state, running_stats, key)

    return variables, losses
