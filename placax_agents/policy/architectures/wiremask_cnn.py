"""CNN actor-critic that also conditions on a wiremask channel; pairs with make_wiremask_observation."""
import jax
import jax.numpy as jnp
from flax import linen as nn


class WiremaskCNNActorCritic(nn.Module):
    """Like CNNActorCritic, but stacks a wiremask channel alongside canvas before the shared conv trunk."""

    features: int = 16
    kernel_size: tuple[int, int] = (3, 3)
    num_conv_layers: int = 2

    @nn.compact
    def __call__(self, obs: dict) -> tuple[jax.Array, jax.Array]:
        # 1. Canvas as float, and wiremask rescaled to roughly [0, 1] to match canvas's scale.
        canvas = obs["canvas"].astype(jnp.float32)
        # Falls back to the lookahead slice when there's no standalone "wiremask" key buffered.
        wiremask = obs["wiremask"] if "wiremask" in obs else obs["lookahead_wiremasks"][0]
        wiremask = wiremask / (wiremask.max() + 1e-6)
        # 2. Stack both as separate channels of one input image.
        x = jnp.stack([canvas, wiremask], axis=-1)

        # 3. Shared conv trunk over both channels together.
        for _ in range(self.num_conv_layers):
            x = nn.relu(nn.Conv(features=self.features, kernel_size=self.kernel_size, padding="SAME")(x))

        # 4. Policy head (one logit per cell) and value head (pooled trunk -> scalar).
        action_logits = nn.Conv(features=1, kernel_size=self.kernel_size, padding="SAME")(x)[..., 0]
        value = nn.Dense(features=1)(x.mean(axis=(0, 1)))[0]
        return action_logits, value
