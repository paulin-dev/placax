"""A CNN actor-critic that chooses a macro's TURN as well as its cell - PPO under `oriented_grid`.

Orientation existed as an axis before this: the space recorded a turn, the GA searched over one,
legality and wirelength and the written `.pl`/DEF all respected it. What could not happen was
*learning* it. Every shipped architecture emitted a `(grid_x, grid_y)` logits map, which is
exactly one grid cell and therefore exactly `discrete_grid` - so `_require_space` refused PPO the
oriented space, correctly, and the axis was searchable only by a method with no policy.

**The output is `(grid_x, grid_y, N_ORIENTATIONS)`, and that is the whole of it.** Sampling,
log-probabilities and PPO's ratio are all rank-agnostic (`sample_action` unravels against the
logits' own shape), so nothing downstream needed a special case. The one thing that DID need
building is legality: a quarter-turned macro is its height by its width and fits in different
cells, so a mask has to be computed per turn - `policy.action.oriented_illegal_actions` - rather
than a two-dimensional map stretched over the turn axis, which would call placements legal that
were never checked.

**Why the turn head is a Dense over the pooled trunk rather than a fourth conv channel.** A conv
channel would make the turn a per-cell function of local canvas features only, which is most of
what decides it - but the useful signal for "should this macro be laid on its side" is the shape
of the macro against the shape of the remaining space, and the macro's own size is not in the
canvas at all. So the turn logits come from the trunk pooled together with the macro's footprint,
and are broadcast across cells before being added to the per-cell map. The two heads therefore
compose: where to put it is spatial, which way round to put it is about the macro.
"""
from placax.extras.orientation import N_ORIENTATIONS  # must precede jax imports

import jax
import jax.numpy as jnp
from flax import linen as nn


class OrientedCNNActorCritic(nn.Module):
    """Conv trunk over the canvas, a per-cell placement head and a per-turn orientation head."""

    action_spaces = ("oriented_grid",)
    """What this policy's output can express, read by `build._require_space`.

    Only the oriented space: a rank-3 logits map has no meaning under a space whose action is two
    numbers, and saying so here is what keeps the refusal a property of the architecture rather
    than a table in `build.py` that every new policy would have to be added to."""

    features: int = 16
    kernel_size: tuple[int, int] = (3, 3)
    num_conv_layers: int = 2
    size_scale: float = 1.0
    """Largest macro dimension in the design, normalizing the footprint fed to the turn head."""

    @nn.compact
    def __call__(self, obs: dict) -> tuple[jax.Array, jax.Array]:
        # 1. Shared conv trunk over the canvas, exactly as CNNActorCritic's.
        x = obs["canvas"][..., None].astype(jnp.float32)
        for _ in range(self.num_conv_layers):
            x = nn.relu(
                nn.Conv(features=self.features, kernel_size=self.kernel_size, padding="SAME")(x)
            )

        # 2. Placement head: one logit per grid cell, the same map every other architecture emits.
        cell_logits = nn.Conv(features=1, kernel_size=self.kernel_size, padding="SAME")(x)[..., 0]

        # 3. Turn head: the pooled trunk plus THIS macro's footprint, which the canvas does not
        #    carry. Broadcast across cells so the two heads add rather than compete for capacity.
        pooled = x.mean(axis=(0, 1))
        footprint = obs["current_macro_size"].astype(jnp.float32) / max(self.size_scale, 1e-6)
        # The aspect ratio explicitly: whether turning a macro helps is mostly a question about
        # how far from square it is, and asking a Dense layer to derive that from two normalized
        # lengths is work it does not need to do.
        aspect = jnp.array([footprint[0] / jnp.maximum(footprint[1], 1e-6)])
        turn_features = jnp.concatenate([pooled, footprint, aspect])
        turn_logits = nn.Dense(features=N_ORIENTATIONS)(turn_features)

        action_logits = cell_logits[..., None] + turn_logits[None, None, :]

        # 4. Value head, named so the grouped optimizer can find it.
        value = nn.Dense(features=1, name="critic_value")(pooled)[0]
        return action_logits, value


__all__ = ["OrientedCNNActorCritic"]
