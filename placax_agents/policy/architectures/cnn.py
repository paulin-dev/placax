"""CNN actor-critic policy - one architecture among possible several (a
GNN is a natural candidate; this directory is its home too)."""
import jax
import jax.numpy as jnp
from flax import linen as nn


class CNNActorCritic(nn.Module):
    """obs dict (uses 'canvas') -> (action_logits (grid_x, grid_y),
    value scalar). Shared conv trunk, two heads."""

    features: int = 16
    kernel_size: tuple[int, int] = (3, 3)
    num_conv_layers: int = 2

    @nn.compact
    def __call__(self, obs: dict) -> tuple[jax.Array, jax.Array]:
        x = obs["canvas"][..., None].astype(jnp.float32)  # add a channel dim
        for _ in range(self.num_conv_layers):  # shared conv trunk
            x = nn.relu(nn.Conv(features=self.features, kernel_size=self.kernel_size, padding="SAME")(x))

        # Policy head: one logit per grid cell.
        action_logits = nn.Conv(features=1, kernel_size=self.kernel_size, padding="SAME")(x)[..., 0]
        # Value head: global-average-pool the trunk, then a scalar.
        value = nn.Dense(features=1)(x.mean(axis=(0, 1)))[0]
        return action_logits, value
