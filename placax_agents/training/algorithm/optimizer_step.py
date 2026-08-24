"""The part of a training step that's genuinely identical whether the
rollout/loss computation was sequential or vmapped: normalize, compute
gradients, apply the optax update. Extracted once both train.py and
parallel_train.py were found duplicating this exact tail."""
from placax_agents.training.algorithm.normalize import normalize_advantages  # noqa: F401  must precede jax imports
from placax_agents.training.algorithm.running_stats import RunningStats, normalize_with_stats  # noqa: F401
from placax_agents.training.algorithm.running_stats import update_running_stats  # noqa: F401

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
    """Normalizes advantages/returns, computes grad(loss_fn), applies one
    optax update. loss_fn must accept (policy_params, normalized_advantages,
    normalized_returns) and return a scalar - the caller decides whether
    that loss_fn is a single-episode call or a vmapped, averaged one."""
    new_running_stats = update_running_stats(running_stats, returns)
    normalized_returns = normalize_with_stats(new_running_stats, returns)
    normalized_advantages = normalize_advantages(advantages)

    loss, grads = jax.value_and_grad(loss_fn)(variables, normalized_advantages, normalized_returns)
    updates, new_opt_state = optimizer.update(grads, opt_state, variables)
    new_variables = optax.apply_updates(variables, updates)

    return new_variables, new_opt_state, new_running_stats, loss
