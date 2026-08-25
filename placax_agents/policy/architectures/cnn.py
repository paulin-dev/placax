"""CNN actor-critic policy - one architecture among possible several (a
GNN is a natural candidate; this directory is its home too)."""
import jax
import jax.numpy as jnp
from flax import linen as nn


class CNNActorCritic(nn.Module):
    """Simple actor-critic over just the canvas: a shared conv trunk feeding a policy head and a value head."""

    features: int = 16
    kernel_size: tuple[int, int] = (3, 3)
    num_conv_layers: int = 2

    @nn.compact
    def __call__(self, obs: dict) -> tuple[jax.Array, jax.Array]:
        # 1. Canvas is (grid_x, grid_y) bool; Conv layers need an explicit channel dim, as float.
        x = obs["canvas"][..., None].astype(jnp.float32)
        # 2. Shared conv trunk: extract spatial features both heads below will read.
        for _ in range(self.num_conv_layers):
            x = nn.relu(nn.Conv(features=self.features, kernel_size=self.kernel_size, padding="SAME")(x))

        # 3. Policy head: one logit per grid cell.
        action_logits = nn.Conv(features=1, kernel_size=self.kernel_size, padding="SAME")(x)[..., 0]
        # 4. Value head: global-average-pool the trunk down to one feature vector, then a scalar.
        value = nn.Dense(features=1)(x.mean(axis=(0, 1)))[0]
        return action_logits, value
