"""An actor-critic that reads raw coordinates instead of an image - the non-CNN arm of §12.

The spec's state-representation experiment is "raw coordinates vs. image (CNN) vs. graph (GNN),
algorithm held fixed". Until now only the image arm existed: every shipped architecture consumed
`obs["canvas"]`, so the comparison could not be set up at all, whatever the config said.

**Which axis that comparison actually varies here is worth being precise about.** `observation()`
returns a superset - a canvas *and* `positions`/`sizes_array`/`placed_mask`/`step` - and each
architecture reads the subset it wants. So the representation under test is chosen by the POLICY,
not by the StateFn, and two arms of the study differ in `agent.policy` while sharing an
environment exactly. That is a stronger comparison than the spec anticipated: it holds at
`environment_hash`, not merely at `task_hash`, because nothing about the environment changes.

The canvas is still read, once, by the machinery that masks illegal actions - legality is a
property of the environment, not of how the agent chose to look at it, and an agent does not get
to opt out of it by picking a different representation.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn


class MLPActorCritic(nn.Module):
    """Flattens the placement itself - positions, sizes, what is placed, how far in - into an MLP.

    Emits the same `(grid_x, grid_y)` logits map every other architecture does, so it drops into
    the identical masking, sampling and loss path. That the head is a `Dense` over the whole grid
    is the honest cost of the coordinate representation: it has no spatial prior at all, which is
    exactly the thing the comparison is supposed to measure rather than assume.
    """

    grid_x: int
    grid_y: int
    size_scale: float = 1.0
    """Largest macro dimension in the design, used to normalize the real-unit size inputs. A plain
    float rather than the sizes array itself: macro geometry is static per design, and putting a
    per-macro array into the observation would carry a copy of it in every stored transition."""

    features: int = 256
    num_layers: int = 2

    @nn.compact
    def __call__(self, obs: dict) -> tuple[jax.Array, jax.Array]:
        # 1. Normalize every geometric input to roughly [0, 1] so one Dense layer sees comparable
        #    scales. Positions carry a -1 sentinel for "not yet placed", which would otherwise be
        #    the largest-magnitude signal in the vector; the placed mask says the same thing
        #    without the outlier, so unplaced rows are zeroed and the mask carries the meaning.
        scale = jnp.array([self.grid_x, self.grid_y], dtype=jnp.float32)
        placed = obs["placed_mask"].astype(jnp.float32)
        positions = jnp.where(
            obs["placed_mask"][:, None], obs["positions"].astype(jnp.float32) / scale, 0.0
        )
        size_scale = max(self.size_scale, 1e-6)

        n_macros = positions.shape[0]
        step = jnp.asarray(obs["step"], dtype=jnp.float32).reshape(()) / max(n_macros, 1)
        x = jnp.concatenate([
            positions.reshape(-1), placed.reshape(-1), step[None],
            # The macro being placed now, and the ones queued behind it - the same lookahead the
            # canvas policies get, so the arms differ in representation and not in information.
            obs["current_macro_size"].astype(jnp.float32).reshape(-1) / size_scale,
            obs["lookahead_sizes"].astype(jnp.float32).reshape(-1) / size_scale,
        ])

        # 2. Shared trunk.
        for _ in range(self.num_layers):
            x = nn.relu(nn.Dense(features=self.features)(x))

        # 3. Heads: one logit per grid cell, and a scalar value. The critic is named so the
        #    grouped optimizer can find it, matching the convention resnet_cnn.py established.
        action_logits = nn.Dense(features=self.grid_x * self.grid_y)(x)
        action_logits = action_logits.reshape(self.grid_x, self.grid_y)
        value = nn.Dense(features=1, name="critic_value")(x)[0]
        return action_logits, value


__all__ = ["MLPActorCritic"]
