"""The optimizer-update tail shared by sequential and parallel training."""
from placax_agents.training.algorithm.normalize import normalize_advantages  # noqa: F401  must precede jax imports
from placax_agents.training.algorithm.running_stats import (  # noqa: F401
    RunningStats, normalize_with_stats, update_running_stats,
)

import jax
import optax


def apply_gradient_update(
    variables,
    opt_state,
    running_stats: RunningStats,
    optimizer: optax.GradientTransformation,
    loss_fn,
    advantages: jax.Array,
    returns: jax.Array,
):
    """One gradient step. loss_fn(policy_params, normalized_advantages,
    normalized_returns) -> scalar. Returns (variables, opt_state,
    running_stats, loss)."""
    new_running_stats = update_running_stats(running_stats, returns)
    normalized_returns = normalize_with_stats(new_running_stats, returns)
    normalized_advantages = normalize_advantages(advantages)

    loss, grads = jax.value_and_grad(loss_fn)(variables, normalized_advantages, normalized_returns)
    updates, new_opt_state = optimizer.update(grads, opt_state, variables)
    new_variables = optax.apply_updates(variables, updates)

    return new_variables, new_opt_state, new_running_stats, loss
