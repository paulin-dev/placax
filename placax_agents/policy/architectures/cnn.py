"""CNN policy architecture - one of possibly several (a GNN is a real
candidate too, given placax's own data is naturally graph-shaped; this
directory exists so a second architecture has an obvious home)."""
import jax
import jax.numpy as jnp
from flax import linen as nn


class CNNActorCritic(nn.Module):
    """obs (a dict with 'canvas') -> (action_logits (grid_x, grid_y), value
    scalar). Shared conv trunk, two heads - standard actor-critic
    shape, needed for PPO's advantage estimation (value predicts
    expected return).

    features/kernel_size/num_conv_layers are Flax fields, not hardcoded -
    a wider or deeper network is a construction-time choice
    (CNNActorCritic(features=32)), not a code edit.

    Takes obs (the whole dict from observation() or any other state_fn),
    not canvas directly: rollout/evaluate/loss all pass the same obs
    dict to any policy uniformly, and each architecture picks out what
    it needs - CNNActorCritic wants canvas, a future GNN would want
    obs['positions']/obs['sizes_array'] instead, without rollout.py or
    evaluate.py needing to know which."""

    features: int = 16
    kernel_size: tuple[int, int] = (3, 3)
    num_conv_layers: int = 2

    @nn.compact
    def __call__(self, obs: dict) -> tuple[jax.Array, jax.Array]:
        canvas = obs["canvas"]
        x = canvas[..., None].astype(jnp.float32)
        for _ in range(self.num_conv_layers):
            x = nn.relu(nn.Conv(features=self.features, kernel_size=self.kernel_size, padding="SAME")(x))

        action_logits = nn.Conv(features=1, kernel_size=self.kernel_size, padding="SAME")(x)[..., 0]

        pooled = x.mean(axis=(0, 1))
        value = nn.Dense(features=1)(pooled)[0]

        return action_logits, value
