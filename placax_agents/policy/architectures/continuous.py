"""A continuous actor-critic: the policy an analytic-gradient method differentiates through.

Every other architecture here emits a `(grid_x, grid_y)` logits map and is sampled from with
`categorical`. That is a discrete choice, and `d(action)/d(parameters)` through it does not exist -
which is the reason SHAC needed an action space of its own and not merely a different agent.

This one emits the parameters of a **continuous** distribution instead: a mean coordinate and a
log standard deviation. An action is drawn by reparameterization, `a = mean + exp(log_std) * eps`,
so the sample is a differentiable function of the parameters and the gradient of a reward computed
at that action reaches them.

**It reads coordinates, not the canvas, and that is load-bearing here in a way it is not for
`mlp`.** The canvas is produced by `render`, which is a comparison - so a policy reading it has no
gradient path from the placement it sees back to the actions that produced it. For PPO that costs
nothing (the reward is a black box either way). For SHAC it would silently truncate
backpropagation through time at every step, leaving only the direct action-to-reward term and
none of the "this macro is where it is because of where I put the last one" signal that the method
exists to exploit. So the observation this consumes is the differentiable half: positions, which
macros are down, how far through the episode, and the footprint being placed.
"""
from placax import _device  # noqa: F401  must run before any `import jax` below

import jax
import jax.numpy as jnp
from flax import linen as nn


class ContinuousActorCritic(nn.Module):
    """Mean + log-std over a real-valued grid coordinate, and a value head, from raw coordinates."""

    action_spaces = ("continuous",)
    """The only space a real-valued action fits - see `build.policy_action_spaces`."""

    grid_x: int
    grid_y: int
    cell_size: float = 1.0
    """Real units per grid cell - needed to express a macro's footprint in the same units as the
    canvas, so the actor head can keep the whole macro on it rather than just its corner."""

    size_scale: float = 1.0
    features: int = 256
    num_layers: int = 2
    init_log_std: float = -1.0
    """Starting exploration width, in grid cells: exp(-1) is about a third of a cell.

    Learned from here rather than fixed, because the useful noise scale changes over a run - wide
    while the layout is being roughed out, narrow once macros are landing in specific places - and
    a hand-set schedule would be one more thing to tune per design."""

    @nn.compact
    def __call__(self, obs: dict) -> tuple[dict, jax.Array]:
        # 1. The same normalized coordinate vector MLPActorCritic reads, for the same reason: one
        #    Dense layer should see inputs on comparable scales. The -1 sentinel is zeroed and the
        #    placed mask carries "not yet down" instead, so an unplaced macro is not the largest
        #    magnitude in the vector.
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
            obs["current_macro_size"].astype(jnp.float32).reshape(-1) / size_scale,
            obs["lookahead_sizes"].astype(jnp.float32).reshape(-1) / size_scale,
        ])

        for _ in range(self.num_layers):
            x = nn.relu(nn.Dense(features=self.features)(x))

        # 2. Actor head: a coordinate whose MACRO fits on the canvas, not merely a coordinate on
        #    it. `sigmoid * (canvas - footprint)` is the continuous counterpart of boundary_mask -
        #    the discrete spaces forbid the cells a macro would overhang, and this makes the
        #    policy's mean unable to ask for them. Without the footprint term the corner is bounded
        #    and the macro is not, so a policy could saturate the sigmoid and place a macro exactly
        #    at the far edge with its whole body outside; measured, that is precisely what a
        #    continuous policy on a crowded canvas learns to do, because leaving is cheaper than
        #    overlapping. The density term's out-of-bounds cost then only has to handle the
        #    exploration noise, which is what a penalty is good at.
        footprint = obs["current_macro_size"].astype(jnp.float32) / self.cell_size
        room = jnp.clip(scale - footprint, 0.0, None)
        mean = nn.sigmoid(nn.Dense(features=2)(x)) * room
        log_std = self.param(
            "log_std", lambda _key: jnp.full((2,), self.init_log_std, dtype=jnp.float32)
        )

        # 3. Value head, named so the grouped optimizer can find it.
        value = nn.Dense(features=1, name="critic_value")(x)[0]
        return {"mean": mean, "log_std": log_std}, value


def sample_continuous(key: jax.Array, action_params: dict) -> jax.Array:
    """One action, drawn so that the gradient reaches the parameters that produced it.

    The reparameterization trick, and the whole reason this architecture exists: `mean + sigma *
    eps` is a differentiable function of `mean` and `log_std`, where drawing from a categorical
    over grid cells is not a differentiable function of anything.
    """
    noise = jax.random.normal(key, action_params["mean"].shape)
    return action_params["mean"] + jnp.exp(action_params["log_std"]) * noise


def mean_action(action_params: dict) -> jax.Array:
    """The distribution's mean - what a greedy evaluation rollout places at."""
    return action_params["mean"]


__all__ = ["ContinuousActorCritic", "mean_action", "sample_continuous"]
