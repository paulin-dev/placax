"""PPO's tunable hyperparameters, passed through as a static jit arg."""
from dataclasses import dataclass

from placax_agents.training.algorithm.loss import ValueLossFn, mse_value_loss


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99  # GAE discount
    lam: float = 0.95  # GAE smoothing - lam=1.0 gives plain Monte-Carlo advantage, no smoothing
    clip_eps: float = 0.2  # PPO ratio clip
    value_coef: float = 0.5  # value-loss weight
    entropy_coef: float = 0.01  # entropy-bonus weight - 0 disables the bonus entirely
    value_loss_fn: ValueLossFn = mse_value_loss  # e.g. loss.huber_value_loss
    normalize_advantages: bool = True  # zero-mean/unit-std advantages per batch before the loss
    normalize_returns: bool = True  # standardize returns with a running mean/std before the value loss
