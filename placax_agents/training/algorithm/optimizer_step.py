"""The optimizer-update tail shared by sequential and parallel training."""
from collections.abc import Callable

from placax_agents.training.algorithm.config import PPOConfig
from placax_agents.training.algorithm.normalize import normalize_advantages  # must precede jax imports
from placax_agents.training.algorithm.running_stats import (
    RunningStats, normalize_with_stats, update_running_stats,
)

import jax
import optax


def apply_gradient_update(
    variables,
    opt_state,
    running_stats: RunningStats,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable[[object, jax.Array, jax.Array], jax.Array],
    advantages: jax.Array,
    returns: jax.Array,
    ppo_config: PPOConfig = PPOConfig(),
):
    """One gradient step: optionally normalizes advantages/returns (per ppo_config), then applies loss_fn's gradient via optimizer, training only variables["params"]."""
    # 1. Fold this batch's returns into the running mean/var, then optionally use the updated stats
    #    to standardize returns - keeps the value loss's scale stable as training progresses.
    new_running_stats = update_running_stats(running_stats, returns)
    normalized_returns = normalize_with_stats(new_running_stats, returns) if ppo_config.normalize_returns else returns
    # 2. Advantages are optionally normalized per-batch instead (no running stats needed for them).
    normalized_advantages = normalize_advantages(advantages) if ppo_config.normalize_advantages else advantages

    # 3. Split out the trainable params from any frozen collection, so only
    #    params get differentiated and optimized below.
    params = variables["params"]
    frozen_collections = {k: v for k, v in variables.items() if k != "params"}

    def params_loss_fn(params, na, nr):
        return loss_fn({**frozen_collections, "params": params}, na, nr)

    # 4. Compute the loss and its gradient w.r.t. params only.
    loss, grads = jax.value_and_grad(params_loss_fn)(params, normalized_advantages, normalized_returns)
    # 5. Let the optimizer (e.g. Adam) turn raw gradients into parameter deltas, then apply them.
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_variables = {**frozen_collections, "params": new_params}

    return new_variables, new_opt_state, new_running_stats, loss
