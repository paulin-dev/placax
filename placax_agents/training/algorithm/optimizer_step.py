"""The optimizer-update tail shared by sequential and parallel training."""
from collections.abc import Callable

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
):
    """One gradient step: normalizes advantages/returns, then applies loss_fn's gradient via optimizer."""
    # 1. Fold this batch's returns into the running mean/var, then use the updated stats to
    #    standardize returns - keeps the value loss's scale stable as training progresses.
    new_running_stats = update_running_stats(running_stats, returns)
    normalized_returns = normalize_with_stats(new_running_stats, returns)
    # 2. Advantages are normalized per-batch instead (no running stats needed for them).
    normalized_advantages = normalize_advantages(advantages)

    # 3. Compute the loss and its gradient w.r.t. the policy/value network parameters.
    loss, grads = jax.value_and_grad(loss_fn)(variables, normalized_advantages, normalized_returns)
    # 4. Let the optimizer (e.g. Adam) turn raw gradients into parameter deltas, then apply them.
    updates, new_opt_state = optimizer.update(grads, opt_state, variables)
    new_variables = optax.apply_updates(variables, updates)

    return new_variables, new_opt_state, new_running_stats, loss
