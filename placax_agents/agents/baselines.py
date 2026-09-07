"""Two non-learning agents: the floor any method must clear, and a strong classical heuristic.

The project had no baselines at all, so "better than X" could not be stated even against a
trivial reference. These two are also what prove the Agent seam is real rather than a PPO shape
wearing a protocol: neither has parameters, an optimizer, a gradient, or a policy network, and
one of them has no learnable state whatsoever.

RandomSearchAgent is the honest floor, and the one most worth reporting. Given a compute budget,
it draws that many uniformly-random legal placements and keeps the best. Almost no paper in this
field reports it, and a method that does not clearly beat compute-matched random search has not
demonstrated anything - which is only checkable now that both can be given the same env_step
budget.

GreedyWiremaskAgent is the classical strong baseline: place each macro at the legal cell that
adds least HPWL, in the same order the RL agent uses. It is deterministic, spends exactly one
episode, and is essentially the greedy oracle the wiremask observation exists to inform - so it
is the reference that says how much of a learned policy's score comes from learning rather than
from the observation it was handed.

**Both run inside the environment the config describes, not beside it.** They build the run's own
observation with its `state_fn`, mask actions with its `extra_illegal_fn` through the same
`illegal_cells` the policy uses, start from its warm-start placement, drive its `step()`, and
rank candidates by its `reward_fn`. That was not previously true: the builders dropped the
action mask and both agents scored with bare HPWL, so under any config declaring an action mask
or a non-HPWL reward these baselines were quietly playing a different game while
`assert_comparable` reported the environments as identical. Everything an agent is allowed to
choose is how it picks an action; nothing else.
"""
from placax.core import replay, reset, step  # must precede jax imports
from placax.types import EnvParams
from placax_agents.agents.base import UpdateResult
from placax_agents.policy.action import illegal_cells
from placax_agents.policy.observation import observation
from placax_agents.policy.scale import to_grid_units, to_real_centers
from placax.extras.rewards import wiremask
from placax.netlist.padding import build_macro_net_index

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


class GreedyWiremaskAgent(_EnvironmentBound):
    """Places every macro at the legal cell that adds the least wirelength. No learning at all."""

    name = "greedy_wiremask"

    def __init__(self, benchmark, state_fn=None, extra_illegal_fn=None,
                 initial_positions=None, n_placed: int = 0):
        super().__init__(benchmark, state_fn, extra_illegal_fn, initial_positions, n_placed)
        # The reverse index of which nets touch each macro, built once - the same precomputation
        # the wiremask observation does.
        self._macro_nets = build_macro_net_index(
            benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
            n_macros=benchmark.params.n_macros,
        )
        self._placement = None

    def init(self, key: jax.Array):
        """No state to carry: the placement is a pure function of the netlist and the order."""
        return {}

    def update(self, key: jax.Array, state):
        """Nothing to learn. One episode is spent so the budget is charged for the work done."""
        return state, UpdateResult(episodes=1, loss=None)

    def converged(self, _state) -> bool:
        """Deterministic and stateless: a second iteration cannot produce a different placement.

        Saying so lets the runner stop instead of spending the rest of a large budget replaying
        the same answer, writing an identical checkpoint and log line each time.
        """
        return True

    def best_positions(self, state) -> jax.Array:
        # Deterministic, so compute once and reuse - `update` cannot change the answer.
        if self._placement is None:
            self._placement = _greedy_wiremask_placement(self)
        return self._placement


def _greedy_wiremask_placement(agent: GreedyWiremaskAgent) -> jax.Array:
    """One left-to-right pass placing each macro at its minimum-HPWL-increase legal cell."""
    benchmark = agent.benchmark
    params = benchmark.params
    macro_net_idx, macro_net_offset, macro_net_valid = agent._macro_nets

    def scan_step(state, _unused):
        # wiremask() scores candidate cells against the macros already placed, and wants REAL-unit
        # centers; unplaced macros are excluded by its own step-based baseline mask.
        real_state = state.replace(
            positions=to_real_centers(state.positions, benchmark.sizes_array, benchmark.cell_size)
        )
        cost = wiremask(
            real_state, params, benchmark.padded_pin_idx, benchmark.padded_pin_offset,
            benchmark.valid_mask, macro_net_idx, macro_net_offset, macro_net_valid,
            benchmark.sizes_array, benchmark.cell_size,
        )
        illegal, _macro_size = agent._illegal(state)
        scored = jnp.where(illegal, jnp.inf, cost)

        flat_idx = jnp.argmin(scored.ravel())
        action = jnp.array([flat_idx // scored.shape[1], flat_idx % scored.shape[1]])
        return _take_action(state, action, params), None

    final_state, _ = jax.lax.scan(
        scan_step, reset(params, agent.initial_positions), jnp.arange(agent._episode_length)
    )
    return final_state.positions


class RandomSearchAgent(_EnvironmentBound):
    """Draws uniformly-random legal placements and keeps the best. The compute floor.

    `population` placements per update, so a budget of N env steps buys N / n_macros samples -
    the same environment work a learning agent gets for the same budget, which is the only
    setting in which "beats random search" means anything.
    """

    name = "random_search"

    def __init__(self, benchmark, population: int = 16, state_fn=None, extra_illegal_fn=None,
                 initial_positions=None, n_placed: int = 0):
        super().__init__(benchmark, state_fn, extra_illegal_fn, initial_positions, n_placed)
        self.population = population

    def init(self, key: jax.Array):
        """Start with no incumbent: the first update's best is unconditionally an improvement."""
        return {
            "best_positions": jnp.full((self.benchmark.params.n_macros, 2), -1, dtype=jnp.int32),
            # Return, so higher is better - the reward is -HPWL, not HPWL.
            "best_return": jnp.array(-jnp.inf),
        }

    def update(self, key: jax.Array, state):
        positions, scores = _random_population(key, self, self.population)
        winner = jnp.argmax(scores)
        improved = scores[winner] > state["best_return"]
        new_state = {
            "best_positions": jnp.where(improved, positions[winner], state["best_positions"]),
            "best_return": jnp.where(improved, scores[winner], state["best_return"]),
        }
        return new_state, UpdateResult(
            episodes=self.population,
            loss=None,
            metrics={"population_best_return": float(scores[winner]),
                     "population_mean_return": float(scores.mean())},
        )

    def best_positions(self, state) -> jax.Array:
        return state["best_positions"]


def _random_placement(key: jax.Array, agent: RandomSearchAgent) -> jax.Array:
    """One placement, each macro sampled uniformly over its own legal cells given the ones before."""
    params = agent.benchmark.params

    def scan_step(state, step_key):
        illegal, _macro_size = agent._illegal(state)
        # Uniform over legal cells: equal logits everywhere, -inf on the illegal ones.
        logits = jnp.where(illegal, -jnp.inf, 0.0)
        flat_idx = jax.random.categorical(step_key, logits.ravel())
        action = jnp.array([flat_idx // illegal.shape[1], flat_idx % illegal.shape[1]])
        return _take_action(state, action, params), None

    final_state, _ = jax.lax.scan(
        scan_step, reset(params, agent.initial_positions),
        jax.random.split(key, agent._episode_length),
    )
    return final_state.positions


def _random_population(key: jax.Array, agent: RandomSearchAgent, population: int):
    """`population` independent random placements and their episode returns, all at once."""
    keys = jax.random.split(key, population)
    # The agent is closed over rather than passed as a vmap argument: it is an ordinary Python
    # object, not a pytree, and closing over it keeps that fact out of the transform entirely.
    positions = jax.vmap(lambda k: _random_placement(k, agent))(keys)
    scores = jax.vmap(agent.score)(positions)
    return positions, scores


__all__ = ["GreedyWiremaskAgent", "RandomSearchAgent"]
