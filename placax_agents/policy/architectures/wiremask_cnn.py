"""CNN actor-critic that also conditions on a wiremask channel - one more
architecture option (see cnn.py), pairs with
policy.observation.make_wiremask_observation as its state_fn."""
import jax
import jax.numpy as jnp
from flax import linen as nn


class WiremaskCNNActorCritic(nn.Module):
    """obs dict (uses 'canvas' and 'wiremask') -> (action_logits
    (grid_x, grid_y), value scalar). Both channels stacked into one
    shared conv trunk, two heads."""

    features: int = 16
    kernel_size: tuple[int, int] = (3, 3)
    num_conv_layers: int = 2

    @nn.compact
    def __call__(self, obs: dict) -> tuple[jax.Array, jax.Array]:
        canvas = obs["canvas"].astype(jnp.float32)
        wiremask = obs["wiremask"]
        wiremask = wiremask / (wiremask.max() + 1e-6)  # keep both channels on a similar scale
        x = jnp.stack([canvas, wiremask], axis=-1)

        for _ in range(self.num_conv_layers):
            x = nn.relu(nn.Conv(features=self.features, kernel_size=self.kernel_size, padding="SAME")(x))

        action_logits = nn.Conv(features=1, kernel_size=self.kernel_size, padding="SAME")(x)[..., 0]
        value = nn.Dense(features=1)(x.mean(axis=(0, 1)))[0]
        return action_logits, value
