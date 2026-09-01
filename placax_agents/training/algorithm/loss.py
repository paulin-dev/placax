"""PPO's loss: clipped surrogate objective + value loss - entropy bonus."""
from typing import Callable

from placax.types import EnvParams  # must precede jax imports
from placax_agents.policy.action import action_log_prob, legal_action_logits
from placax_agents.policy.scale import to_grid_units
from placax_agents.types import AlgorithmFn, ExtraIllegalFn

import jax
import jax.numpy as jnp


def _entropy(masked_logits: jax.Array) -> jax.Array:
    """Entropy of the action distribution: -sum(probs * log_probs)."""
    # Illegal cells are masked to -inf logits, so their prob is exactly 0 but
    # their log_prob is -inf; zero them out explicitly before multiplying,
    # otherwise 0 * -inf produces a NaN that poisons the gradient.
    log_probs = jax.nn.log_softmax(masked_logits.ravel())
    probs = jnp.exp(log_probs)
    safe_log_probs = jnp.where(probs > 0, log_probs, 0.0)
    return -jnp.sum(probs * safe_log_probs)


def mse_value_loss(value: jax.Array, ret: jax.Array) -> jax.Array:
    """Default value loss: (value - return)^2."""
    return (value - ret) ** 2


def huber_value_loss(value: jax.Array, ret: jax.Array, delta: float = 1.0) -> jax.Array:
    """Smooth-L1/Huber value loss (quadratic near 0, linear beyond delta), less sensitive to outliers than MSE."""
    error = value - ret
    abs_error = jnp.abs(error)
    return jnp.where(abs_error <= delta, 0.5 * error**2, delta * (abs_error - 0.5 * delta))


ValueLossFn = Callable[[jax.Array, jax.Array], jax.Array]


def ppo_loss(
    policy_params,
    policy_apply_fn: AlgorithmFn,
    trajectory: dict,
    advantages: jax.Array,
    returns: jax.Array,
    cell_size: float,
    params: EnvParams,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    value_loss_fn: ValueLossFn = mse_value_loss,
    extra_illegal_fn: ExtraIllegalFn | None = None,
) -> jax.Array:
    """PPO loss: mean(policy_loss) + value_coef*mean(value_loss) - entropy_coef*mean(entropy)."""

    def per_step(obs, action, old_log_prob, advantage, ret):
        # 1. Re-run the current (being-trained) policy on the saved observation to get
        #    fresh logits/value - these will differ from rollout time as params update.
        logits, value = policy_apply_fn(policy_params, obs)

        # 2. Recompute the same legal-action mask used at rollout time, so the
        #    probability ratio below compares apples to apples.
        macro_size = to_grid_units(obs["current_macro_size"], cell_size)
        extra_illegal = extra_illegal_fn(obs) if extra_illegal_fn is not None else None
        masked_logits = legal_action_logits(logits, obs["canvas"], params, macro_size, extra_illegal)
        new_log_prob = action_log_prob(masked_logits, action)

        # 3. PPO's clipped surrogate objective: reward the new policy for increasing
        #    probability on actions with positive advantage, but clip the ratio so a
        #    single update can't move the policy too far from what collected the data.
        ratio = jnp.exp(new_log_prob - old_log_prob)
        clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -jnp.minimum(ratio * advantage, clipped_ratio * advantage)

        # 4. Value loss trains the critic head to predict the return, and entropy is a
        #    bonus (subtracted as a loss) that keeps the policy exploring instead of
        #    collapsing onto one action too early.
        value_loss = value_loss_fn(value, ret)
        return policy_loss, value_loss, _entropy(masked_logits)

    # Apply per_step to every transition in the trajectory at once, then combine into one scalar.
    policy_losses, value_losses, entropies = jax.vmap(per_step)(
        trajectory["obs"], trajectory["action"], trajectory["log_prob"], advantages, returns,
    )
    return policy_losses.mean() + value_coef * value_losses.mean() - entropy_coef * entropies.mean()
