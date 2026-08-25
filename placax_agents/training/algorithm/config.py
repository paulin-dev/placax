"""PPO's tunable hyperparameters, passed through as a static jit arg."""
from dataclasses import dataclass

from placax_agents.training.algorithm.loss import ValueLossFn, huber_value_loss, mse_value_loss


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
directly as train_sequential/train_parallel's learning_rate."""


def maskplace_ppo_config() -> PPOConfig:
    """PPOConfig matching MaskPlace's own PPO2.py: gamma=0.95 (its
    --gamma default), no GAE smoothing (lam=1.0 - plain discounted
    return, matching its buffer-based return-to-go), no entropy bonus,
    Huber value loss. Pair with MASKPLACE_LEARNING_RATE above.

    Not covered by this preset, and not closable by a PPOConfig alone:
    MaskPlace collects a multi-episode replay buffer
    (buffer_capacity = 10 * placed_num_macro) and runs ppo_epoch=10
    minibatch (batch_size=64) passes over it per update; placax's
    train_sequential/train_parallel each do one full-batch gradient step
    per rollout - a training-loop-shape difference, not a hyperparameter,
    so this preset can't close it."""
    return PPOConfig(gamma=0.95, lam=1.0, clip_eps=0.2, entropy_coef=0.0, value_loss_fn=huber_value_loss)
