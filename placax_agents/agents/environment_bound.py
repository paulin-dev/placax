"""The environment half of an agent: everything it must honor, however it picks actions.

Lifted out of `baselines.py` once a third agent family needed it. An agent chooses WHERE to put
the next macro and nothing else - the observation it sees, the cells it may legally use, the warm
start it begins from and the reward it is judged by all belong to the environment, and an agent
that reconstructs any of them for itself is running a different experiment while sharing a config
that claims otherwise. That was a real bug once (the baseline builders dropped the action mask),
so the wiring lives in one place and every agent takes it whole.
"""
from placax.action_space import DISCRETE_GRID  # must precede jax imports
from placax.core import replay, step
from placax.types import EnvParams
from placax_agents.policy.action import illegal_cells
from placax_agents.policy.scale import to_grid_units

import jax
import jax.numpy as jnp


class _EnvironmentBound:
    """Shared wiring: the environment pieces every agent must honor, however it picks actions."""

    def __init__(self, benchmark, state_fn=None, extra_illegal_fn=None,
                 initial_positions=None, n_placed: int = 0, action_space=None):
        self.benchmark = benchmark
        # The environment's action space, not the agent's: what an action means is a property of
        # the task every agent in a comparison is solving, not of how one of them picks.
        self.action_space = action_space if action_space is not None else DISCRETE_GRID
        # Fall back to the benchmark's own correctly-bound observation rather than the unbound
        # `observation` default, whose cell_size=1.0 is wrong for any real benchmark.
        self.state_fn = state_fn if state_fn is not None else benchmark.state_fn
        self.extra_illegal_fn = extra_illegal_fn
        self.initial_positions = initial_positions
        self.n_placed = n_placed
        self._grid_sizes = to_grid_units(benchmark.sizes_array, benchmark.cell_size)

    @property
    def _episode_length(self) -> int:
        """Actions in one episode - the action space's answer, not an assumed macro count."""
        return self.action_space.episode_length(self.benchmark.params, self.n_placed)

    def _illegal(self, state, orientation=None) -> tuple[jax.Array, jax.Array]:
        """(illegal-cell map, this macro's grid size) at `state`, under the run's full rule set.

        `orientation` is the turn the action is about to give this macro, when the space lets an
        agent choose one. It has to be applied HERE and not after the fact: a macro laid on its
        side is its height by its width, so the cells it may legally occupy are different, and
        asking with the unrotated footprint would accept placements that do not fit.
        """
        obs = self.state_fn(state, self.benchmark.params, self.benchmark.sizes_array)
        size = obs["current_macro_size"]
        if orientation is not None:
            # A quarter turn (WEST or EAST) swaps width and height; a half turn does not.
            size = jnp.where(orientation % 2 == 1, size[::-1], size)
        macro_size = to_grid_units(size, self.benchmark.cell_size)
        extra = self.extra_illegal_fn(obs) if self.extra_illegal_fn is not None else None
        return illegal_cells(obs["canvas"], self.benchmark.params, macro_size, extra), macro_size

    def _illegal_for_macro(self, state, macro_idx) -> jax.Array:
        """Cells macro `macro_idx` may not move to, with the macro itself LIFTED off the canvas.

        A perturbation-only need, and a trap without it: a macro still standing on the canvas
        blocks every cell it occupies, so the only "legal" move it could ever find is back to
        exactly where it already is. Lifting it first asks the real question - where could this
        macro go, given everything else.
        """
        lifted = state.replace(positions=state.positions.at[macro_idx].set(-1))
        obs = self.state_fn(lifted, self.benchmark.params, self.benchmark.sizes_array)
        extra = self.extra_illegal_fn(obs) if self.extra_illegal_fn is not None else None
        return illegal_cells(
            obs["canvas"], self.benchmark.params, self._grid_sizes[macro_idx], extra
        )

    def score(self, positions: jax.Array, orientations: jax.Array | None = None) -> jax.Array:
        """This placement's episode return under the run's configured reward.

        Not HPWL: swapping the reward has to move what every agent optimizes, or the reward axis
        is only swappable for the one agent that happens to read it.

        `orientations` are the turns this placement was built with, for a space that has them. The
        space encodes the pair back into the actions that produced it, so a search over turns is
        scored on the placement it actually made rather than on an all-north reading of it.
        """
        return replay(positions, self.benchmark.reward_fn, self.benchmark.params, self.n_placed,
                      self.action_space, orientations)

    def converged(self, _state) -> bool:
        return False


def _take_action(state, action, params: EnvParams, action_space=DISCRETE_GRID):
    """Applies one action through the shared kernel, discarding the per-step reward.

    Baselines rank whole placements rather than steps, so they ignore the per-step value - but
    they still go through `step()`, because a second copy of the state transition is how the
    one-kernel property stops being true.
    """
    new_state, _reward, _done = step(state, action, _zero_reward, params, action_space)
    return new_state


def _zero_reward(_old_positions, _new_positions, _old_placed, _new_placed,
                 _orientations=None) -> jax.Array:
    """Takes the orientation argument `step()` passes under an orientation-bearing space, and
    ignores it like everything else here: these agents rank whole placements, not steps."""
    return jnp.array(0.0)
