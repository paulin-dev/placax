"""The environment half of an agent: everything it must honor, however it picks actions.

Lifted out of `baselines.py` once a third agent family needed it. An agent chooses WHERE to put
the next macro and nothing else - the observation it sees, the cells it may legally use, the warm
start it begins from and the reward it is judged by all belong to the environment, and an agent
that reconstructs any of them for itself is running a different experiment while sharing a config
that claims otherwise. That was a real bug once (the baseline builders dropped the action mask),
so the wiring lives in one place and every agent takes it whole.
"""
from placax.core import replay, step  # must precede jax imports
from placax.types import EnvParams
from placax_agents.policy.action import illegal_cells
from placax_agents.policy.scale import to_grid_units

import jax
import jax.numpy as jnp


class _EnvironmentBound:
    """Shared wiring: the environment pieces every agent must honor, however it picks actions."""

    def __init__(self, benchmark, state_fn=None, extra_illegal_fn=None,
                 initial_positions=None, n_placed: int = 0):
        self.benchmark = benchmark
        # Fall back to the benchmark's own correctly-bound observation rather than the unbound
        # `observation` default, whose cell_size=1.0 is wrong for any real benchmark.
        self.state_fn = state_fn if state_fn is not None else benchmark.state_fn
        self.extra_illegal_fn = extra_illegal_fn
        self.initial_positions = initial_positions
        self.n_placed = n_placed
        self._grid_sizes = to_grid_units(benchmark.sizes_array, benchmark.cell_size)

    @property
    def _episode_length(self) -> int:
        """Macros this agent still has to place, after the environment's warm-start prefix."""
        return self.benchmark.params.n_macros - self.n_placed

    def _illegal(self, state) -> tuple[jax.Array, jax.Array]:
        """(illegal-cell map, this macro's grid size) at `state`, under the run's full rule set."""
        obs = self.state_fn(state, self.benchmark.params, self.benchmark.sizes_array)
        macro_size = to_grid_units(obs["current_macro_size"], self.benchmark.cell_size)
        extra = self.extra_illegal_fn(obs) if self.extra_illegal_fn is not None else None
        return illegal_cells(obs["canvas"], self.benchmark.params, macro_size, extra), macro_size

    def score(self, positions: jax.Array) -> jax.Array:
        """This placement's episode return under the run's configured reward.

        Not HPWL: swapping the reward has to move what every agent optimizes, or the reward axis
        is only swappable for the one agent that happens to read it.
        """
        return replay(positions, self.benchmark.reward_fn, self.benchmark.params, self.n_placed)

    def converged(self, _state) -> bool:
        return False


def _take_action(state, action, params: EnvParams):
    """Applies one action through the shared kernel, discarding the per-step reward.

    Baselines rank whole placements rather than steps, so they ignore the per-step value - but
    they still go through `step()`, because a second copy of the state transition is how the
    one-kernel property stops being true.
    """
    new_state, _reward, _done = step(state, action, _zero_reward, params)
    return new_state


def _zero_reward(_old_positions, _new_positions, _old_placed, _new_placed) -> jax.Array:
    return jnp.array(0.0)
