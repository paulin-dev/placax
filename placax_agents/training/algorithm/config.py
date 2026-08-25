"""PPO's tunable hyperparameters, passed through as a static jit arg."""
from dataclasses import dataclass

from placax_agents.training.algorithm.loss import ValueLossFn, huber_value_loss, mse_value_loss
from placax_agents.training.algorithm.split_optimizer import make_grouped_optimizer

import optax


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99  # GAE discount
    lam: float = 0.95  # GAE smoothing - lam=1.0 gives plain Monte-Carlo advantage, no smoothing
    clip_eps: float = 0.2  # PPO ratio clip
    value_coef: float = 0.5  # value-loss weight
    entropy_coef: float = 0.01  # entropy-bonus weight - 0 disables the bonus entirely
    value_loss_fn: ValueLossFn = mse_value_loss  # e.g. loss.huber_value_loss


MASKPLACE_LEARNING_RATE = 2.5e-3
"""MaskPlace's own --lr default (PPO2.py) - not a PPOConfig field, pass
directly as train_sequential/train_parallel's learning_rate, or bake it
into maskplace_optimizer() below."""

MASKPLACE_MAX_GRAD_NORM = 0.5
"""MaskPlace's own PPO.max_grad_norm class attribute - the global-norm
clip applied to each network separately, not jointly (see
maskplace_optimizer())."""


def maskplace_ppo_config() -> PPOConfig:
    """PPOConfig matching MaskPlace's own PPO2.py: gamma=0.95 (its
    --gamma default), no GAE smoothing (lam=1.0 - plain discounted
    return, matching its buffer-based return-to-go), no entropy bonus,
    Huber value loss. Pair with MASKPLACE_LEARNING_RATE/maskplace_optimizer()."""
    return PPOConfig(gamma=0.95, lam=1.0, clip_eps=0.2, entropy_coef=0.0, value_loss_fn=huber_value_loss)


def maskplace_optimizer(
    learning_rate: float = MASKPLACE_LEARNING_RATE,
    max_grad_norm: float = MASKPLACE_MAX_GRAD_NORM,
    critic_param_prefix: str = "critic_",
) -> optax.GradientTransformation:
    """MaskPlace's own optimizer setup: Adam(learning_rate) for the actor
    and Adam(learning_rate) for the critic, each with its own
    clip-by-global-norm(max_grad_norm) applied to only its own network's
    gradients - matching MaskPlace's two separate optimizers and two
    separate .backward() calls (PPO2.py), without needing a second
    backward pass (see split_optimizer.make_grouped_optimizer for why
    that's an exact equivalence, not an approximation).

    Requires critic parameters to be named under critic_param_prefix and
    share no parameters with the actor - true for
    policy.architectures.resnet_cnn.ResNetCoarseFineActorCritic's
    critic_style="step_embedding", not for critic_style="canvas" (whose
    value head shares an upstream trunk with the actor - clipping it
    separately there wouldn't reproduce MaskPlace's independence, since
    the two aren't actually independent to begin with)."""
    per_network = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate))
    return make_grouped_optimizer(per_network, per_network, critic_param_prefix)
