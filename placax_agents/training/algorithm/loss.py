"""PPO's loss: clipped surrogate objective + value loss - entropy bonus."""
from placax.types import EnvParams  # must precede jax imports
from placax_agents.policy.action import action_log_prob, legal_action_logits
from placax_agents.policy.scale import to_grid_units
from placax_agents.types import AlgorithmFn

import jax
import jax.numpy as jnp


def _entropy(masked_logits: jax.Array) -> jax.Array:
    """-sum(probs * log_probs). Guards probs==0 explicitly - masking
    only the product still NaNs the gradient through the -inf branch."""
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
    """mean(policy_loss) + value_coef*mean(value_loss) - entropy_coef*mean(entropy),
    over one trajectory dict (as produced by collect_rollout)."""
    macro_sizes = to_grid_units(sizes_array, cell_size)

    def per_step(obs, action, old_log_prob, advantage, ret, macro_size):
        # Recompute the legal mask from the saved obs so the ratio compares
        # probabilities under the same distribution used at rollout time.
        logits, value = policy_apply_fn(policy_params, obs)
        masked_logits = legal_action_logits(logits, obs["canvas"], params, macro_size)
        new_log_prob = action_log_prob(masked_logits, action)

        ratio = jnp.exp(new_log_prob - old_log_prob)  # new/old probability ratio
        clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -jnp.minimum(ratio * advantage, clipped_ratio * advantage)  # PPO's clipped surrogate

        value_loss = (value - ret) ** 2  # critic MSE against the actual return
        return policy_loss, value_loss, _entropy(masked_logits)

    policy_losses, value_losses, entropies = jax.vmap(per_step)(
        trajectory["obs"], trajectory["action"], trajectory["log_prob"],
        advantages, returns, macro_sizes,
    )
    return policy_losses.mean() + value_coef * value_losses.mean() - entropy_coef * entropies.mean()
