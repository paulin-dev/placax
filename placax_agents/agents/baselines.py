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
"""
from placax.core import reset  # must precede jax imports
from placax.extras.masks import boundary_mask, occupancy_mask
from placax.extras.render import render
from placax.extras.rewards import hpwl, wiremask
from placax.netlist.padding import build_macro_net_index
from placax.types import EnvParams, EnvState
from placax_agents.agents.base import UpdateResult
from placax_agents.policy.scale import to_grid_units, to_real_centers

import jax
import jax.numpy as jnp


def _illegal_mask(canvas: jax.Array, params: EnvParams, macro_size: jax.Array) -> jax.Array:
    """Cells where this macro cannot legally go: overlapping, or hanging off the canvas."""
    illegal = occupancy_mask(canvas, (macro_size[0], macro_size[1])) | boundary_mask(
        params, (macro_size[0], macro_size[1])
    )
    # A macro with nowhere legal to go would make argmin/sampling meaningless. Relaxing to "all
    # cells legal" keeps the scan total; it can only happen on a canvas already too full, and the
    # resulting overlap shows up honestly in the HPWL the runner computes.
    return jnp.where(illegal.all(), False, illegal)


class GreedyWiremaskAgent:
    """Places every macro at the legal cell that adds the least wirelength. No learning at all."""

    name = "greedy_wiremask"

    def __init__(self, benchmark):
        self.benchmark = benchmark
        # The reverse index of which nets touch each macro, built once - the same precomputation
        # the wiremask observation does.
        self._macro_nets = build_macro_net_index(
            benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
            n_macros=benchmark.params.n_macros,
        )
        self._grid_sizes = to_grid_units(benchmark.sizes_array, benchmark.cell_size)
        self._placement = None

    def init(self, key: jax.Array):
        """No state to carry: the placement is a pure function of the netlist and the order."""
        return {}

    def update(self, key: jax.Array, state):
        """Nothing to learn. One episode is spent so the budget is charged for the work done."""
        return state, UpdateResult(episodes=1, loss=None)

    def best_positions(self, state) -> jax.Array:
        # Deterministic, so compute once and reuse - `update` cannot change the answer.
        if self._placement is None:
            self._placement = _greedy_wiremask_placement(self.benchmark, self._macro_nets,
                                                         self._grid_sizes)
        return self._placement


def _greedy_wiremask_placement(benchmark, macro_nets, grid_sizes) -> jax.Array:
    """One left-to-right pass placing each macro at its minimum-HPWL-increase legal cell."""
    params = benchmark.params
    macro_net_idx, macro_net_offset, macro_net_valid = macro_nets

    def scan_step(state: EnvState, _unused):
        # wiremask() scores candidate cells against the macros already placed, and wants REAL-unit
        # centers; unplaced macros are excluded by its own step-based baseline mask.
        real_state = EnvState(
            positions=to_real_centers(state.positions, benchmark.sizes_array, benchmark.cell_size),
            step=state.step,
        )
        cost = wiremask(
            real_state, params, benchmark.padded_pin_idx, benchmark.padded_pin_offset,
            benchmark.valid_mask, macro_net_idx, macro_net_offset, macro_net_valid,
            benchmark.sizes_array, benchmark.cell_size,
        )
        canvas = render(state.positions, grid_sizes, params.grid_x, params.effective_grid_y)
        macro_size = grid_sizes[state.step]
        scored = jnp.where(_illegal_mask(canvas, params, macro_size), jnp.inf, cost)

        flat_idx = jnp.argmin(scored.ravel())
        action = jnp.array([flat_idx // scored.shape[1], flat_idx % scored.shape[1]])
        return state.replace(
            positions=state.positions.at[state.step].set(action), step=state.step + 1
        ), None

    final_state, _ = jax.lax.scan(scan_step, reset(params), jnp.arange(params.n_macros))
    return final_state.positions


class RandomSearchAgent:
    """Draws uniformly-random legal placements and keeps the best HPWL seen. The compute floor.

    `population` placements per update, so a budget of N env steps buys N / n_macros samples -
    the same environment work a learning agent gets for the same budget, which is the only
    setting in which "beats random search" means anything.
    """

    name = "random_search"

    def __init__(self, benchmark, population: int = 16):
        self.benchmark = benchmark
        self.population = population
        self._grid_sizes = to_grid_units(benchmark.sizes_array, benchmark.cell_size)

    def init(self, key: jax.Array):
        """Start with no incumbent: the first update's best is unconditionally an improvement."""
        return {
            "best_positions": jnp.full((self.benchmark.params.n_macros, 2), -1, dtype=jnp.int32),
            "best_hpwl": jnp.array(jnp.inf),
        }

    def update(self, key: jax.Array, state):
        positions, scores = _random_population(
            key, self.benchmark, self._grid_sizes, self.population
        )
        winner = jnp.argmin(scores)
        improved = scores[winner] < state["best_hpwl"]
        new_state = {
            "best_positions": jnp.where(improved, positions[winner], state["best_positions"]),
            "best_hpwl": jnp.where(improved, scores[winner], state["best_hpwl"]),
        }
        return new_state, UpdateResult(
            episodes=self.population,
            loss=None,
            metrics={"population_best_hpwl": float(scores[winner]),
                     "population_mean_hpwl": float(scores.mean())},
        )

    def best_positions(self, state) -> jax.Array:
        return state["best_positions"]


def _random_placement(key: jax.Array, benchmark, grid_sizes) -> jax.Array:
    """One placement, each macro sampled uniformly over its own legal cells given the ones before."""
    params = benchmark.params

    def scan_step(state: EnvState, step_key):
        canvas = render(state.positions, grid_sizes, params.grid_x, params.effective_grid_y)
        illegal = _illegal_mask(canvas, params, grid_sizes[state.step])
        # Uniform over legal cells: equal logits everywhere, -inf on the illegal ones.
        logits = jnp.where(illegal, -jnp.inf, 0.0)
        flat_idx = jax.random.categorical(step_key, logits.ravel())
        action = jnp.array([flat_idx // illegal.shape[1], flat_idx % illegal.shape[1]])
        return state.replace(
            positions=state.positions.at[state.step].set(action), step=state.step + 1
        ), None

    final_state, _ = jax.lax.scan(
        scan_step, reset(params), jax.random.split(key, params.n_macros)
    )
    return final_state.positions


def _score(benchmark, positions: jax.Array) -> jax.Array:
    """Real HPWL of one placement - the same metric the runner uses to compare agents."""
    centers = to_real_centers(positions, benchmark.sizes_array, benchmark.cell_size)
    return hpwl(centers, benchmark.padded_pin_idx, benchmark.padded_pin_offset,
                benchmark.valid_mask)


def _random_population(key: jax.Array, benchmark, grid_sizes, population: int):
    """`population` independent random placements and their HPWLs, all at once."""
    keys = jax.random.split(key, population)
    positions = jax.vmap(_random_placement, in_axes=(0, None, None))(keys, benchmark, grid_sizes)
    scores = jax.vmap(_score, in_axes=(None, 0))(benchmark, positions)
    return positions, scores
