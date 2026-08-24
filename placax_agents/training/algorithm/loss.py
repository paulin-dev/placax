"""PPO's loss: clipped surrogate objective + value loss - entropy bonus,
over one full trajectory. Each step's legal mask is recomputed from its
saved obs so the ratio compares probabilities under the same
distribution used at rollout time."""
from placax.types import EnvParams  # noqa: F401  must precede jax imports
from placax_agents.policy.action import action_log_prob, legal_action_logits  # noqa: F401
from placax_agents.policy.scale import to_grid_units  # noqa: F401
from placax_agents.types import AlgorithmFn  # noqa: F401

import jax
import jax.numpy as jnp


def _entropy(masked_logits: jax.Array) -> jax.Array:
    """Entropy of the masked categorical distribution - illegal actions
    contribute exactly 0. Guards the *input* to probs * log_probs, not
    just the output: masking only the output still differentiates
    through the discarded -inf branch and yields NaN gradients."""
    log_probs = jax.nn.log_softmax(masked_logits.ravel())
    probs = jnp.exp(log_probs)
    safe_log_probs = jnp.where(probs > 0, log_probs, 0.0)
    return -jnp.sum(probs * safe_log_probs)


def ppo_loss(
    policy_params,
    policy_apply_fn: AlgorithmFn,
    trajectory: dict,
    advantages: jax.Array,
    returns: jax.Array,
    sizes_array: jax.Array,
    cell_size: float,
    params: EnvParams,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> jax.Array:
    """Scalar PPO loss over one trajectory dict (as produced by
    collect_rollout): policy surrogate + value_coef * value MSE -
    entropy_coef * entropy, each averaged over steps."""
    macro_sizes = to_grid_units(sizes_array, cell_size)

    def per_step(obs, action, old_log_prob, advantage, ret, macro_size):
        logits, value = policy_apply_fn(policy_params, obs)
        masked_logits = legal_action_logits(logits, obs["canvas"], params, macro_size)
        new_log_prob = action_log_prob(masked_logits, action)

        ratio = jnp.exp(new_log_prob - old_log_prob)
        clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -jnp.minimum(ratio * advantage, clipped_ratio * advantage)

        value_loss = (value - ret) ** 2
        return policy_loss, value_loss, _entropy(masked_logits)

    policy_losses, value_losses, entropies = jax.vmap(per_step)(
        trajectory["obs"], trajectory["action"], trajectory["log_prob"],
        advantages, returns, macro_sizes,
    )
    return policy_losses.mean() + value_coef * value_losses.mean() - entropy_coef * entropies.mean()
