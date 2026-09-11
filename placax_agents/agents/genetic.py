"""A genetic algorithm over placements - the first agent here from a different family entirely.

Three agents shipped before this and all three were sequential: PPO samples an action per step, and
both baselines choose one per step too. The spec's algorithm-family comparison ("PPO/SHAC vs. ACO
vs. GA, reward and benchmark held fixed - does BBOPlace-Bench's finding that evolutionary methods
beat RL on several benchmarks extend to these?") therefore could not be set up at all. It can now.

**What a genome is, and why it is not a list of coordinates.** The obvious encoding - one (x, y)
per macro, mutated freely - produces illegal placements, and the environment would then either
reject them or, worse, let them overlap. Overlapping macros have shorter wires, so a GA given a
wirelength objective and no legality constraint optimizes straight into an unrealizable answer.

So a genome is a vector of *preferences* in [0, 1)², one per macro, decoded through the run's own
legality mask: at each step the macro goes to the legal cell nearest its preferred one. Every
placement the GA can express is therefore legal by construction, exactly as the policy's is,
because both go through the same `illegal_cells`. This is the shape WireMask-BBO and
BBOPlace-Bench use - search over a continuous encoding, decode with a placement heuristic - which
is what makes the comparison the spec asks for a real one rather than a strawman.

The kernel needed no change for any of this, which was the design claim all along and is now
tested rather than asserted: `step()` cannot tell whether an action came from a policy's sample or
a genome's decode.
"""
from placax_agents.agents.base import UpdateResult  # must precede jax imports
from placax_agents.agents.environment_bound import _EnvironmentBound, _take_action
from placax.core import reset
from placax.extras.orientation import N_ORIENTATIONS

import jax
import jax.numpy as jnp


class GeneticAgent(_EnvironmentBound):
    """Evolves a population of placement preferences under the run's configured reward."""

    name = "genetic"

    def __init__(self, benchmark, population: int = 32, elite_fraction: float = 0.25,
                 mutation_rate: float = 0.1, mutation_scale: float = 0.15,
                 state_fn=None, extra_illegal_fn=None, initial_positions=None, n_placed: int = 0,
                 action_space=None):
        super().__init__(benchmark, state_fn, extra_illegal_fn, initial_positions, n_placed,
                         action_space)
        if population < 2:
            raise ValueError(f"a population needs at least 2 genomes, got {population}")
        self.population = population
        self.n_elite = max(1, int(round(population * elite_fraction)))
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        # Three genes per macro instead of two when the space lets an agent turn a macro: the
        # orientation is part of the placement being searched, so it belongs in the genome rather
        # than being fixed at north behind the search's back.
        self.chooses_orientation = getattr(self.action_space, "name", "") == "oriented_grid"
        self.gene_width = 3 if self.chooses_orientation else 2

    # ------------------------------------------------------------------ Agent

    def init(self, key: jax.Array) -> dict:
        """A uniformly random population, and no incumbent yet."""
        genomes = jax.random.uniform(
            key, (self.population, self._episode_length, self.gene_width)
        )
        return {
            "genomes": genomes,
            "best_positions": jnp.full((self.benchmark.params.n_macros, 2), -1, dtype=jnp.int32),
            "best_orientations": jnp.zeros((self.benchmark.params.n_macros,), dtype=jnp.int32),
            "best_return": jnp.array(-jnp.inf),
        }

    def update(self, key: jax.Array, state: dict) -> tuple[dict, UpdateResult]:
        """One generation: decode and score every genome, keep the elites, breed the rest."""
        positions, orientations, returns = _decode_population(state["genomes"], self)

        # 1. Track the best placement ever seen, not merely the best in this generation - a GA
        #    can lose its incumbent to mutation, and the runner asks for the agent's best answer.
        champion = jnp.argmax(returns)
        improved = returns[champion] > state["best_return"]
        best_positions = jnp.where(improved, positions[champion], state["best_positions"])
        best_orientations = jnp.where(
            improved, orientations[champion], state["best_orientations"]
        )
        best_return = jnp.where(improved, returns[champion], state["best_return"])

        # 2. Elites survive untouched; everything else is bred from them.
        elite_idx = jnp.argsort(-returns)[: self.n_elite]
        elites = state["genomes"][elite_idx]
        breed_key, mutate_key = jax.random.split(key)
        children = _breed(breed_key, elites, self.population - self.n_elite)
        children = _mutate(mutate_key, children, self.mutation_rate, self.mutation_scale)
        genomes = jnp.concatenate([elites, children], axis=0)

        new_state = {"genomes": genomes, "best_positions": best_positions,
                     "best_orientations": best_orientations, "best_return": best_return}
        return new_state, UpdateResult(
            episodes=self.population,
            loss=None,
            # A GA has no gradient, so env_steps is the whole of what it spends - which is exactly
            # the case the budget's sample-matched unit exists to make comparable.
            gradient_steps=0,
            metrics={
                "population_best_return": float(returns[champion]),
                "population_mean_return": float(returns.mean()),
                # Spread across the population: a GA that has converged has nothing left to
                # explore, and reporting it distinguishes "finished" from "stuck".
                "population_std_return": float(returns.std()),
            },
        )

    def best_positions(self, state: dict) -> jax.Array:
        return state["best_positions"]

    def best_orientations(self, state: dict):
        """The turns that came with the best placement, or None when the space has no such axis.

        None rather than an all-north array so that a run without orientation scores through the
        exact same code path it always did - see `experiment.run.best_orientations`.
        """
        return state["best_orientations"] if self.chooses_orientation else None


def _decode(genome: jax.Array, agent: GeneticAgent) -> jax.Array:
    """One genome into one placement: each macro at the LEGAL cell nearest its preference.

    Going through `agent._illegal` rather than clipping into range is what keeps the search inside
    the environment: the same occupancy, boundary and quality rules the policy plays under, from
    the same `illegal_cells`.
    """
    params = agent.benchmark.params

    def scan_step(state, preference):
        # The third gene, where there is one, is the quarter turn - and it has to be decided
        # BEFORE legality, since a turned macro has a different footprint and therefore a
        # different set of cells it fits in.
        if agent.chooses_orientation:
            preferred = jnp.clip(
                (preference[2] * N_ORIENTATIONS).astype(jnp.int32), 0, N_ORIENTATIONS - 1
            )
            turn, illegal = _turn_that_fits(state, preferred, agent)
        else:
            turn = None
            illegal, _macro_size = agent._illegal(state, turn)
        grid_x, grid_y = illegal.shape
        target_x = preference[0] * grid_x
        target_y = preference[1] * grid_y
        xs = jnp.arange(grid_x, dtype=jnp.float32)[:, None]
        ys = jnp.arange(grid_y, dtype=jnp.float32)[None, :]
        distance = (xs - target_x) ** 2 + (ys - target_y) ** 2
        scored = jnp.where(illegal, jnp.inf, distance)
        flat_idx = jnp.argmin(scored.ravel())
        cell = jnp.array([flat_idx // grid_y, flat_idx % grid_y])
        action = cell if turn is None else jnp.concatenate([cell, turn[None]])
        return _take_action(state, action, params, agent.action_space), None

    final_state, _ = jax.lax.scan(
        scan_step, reset(params, agent.initial_positions, agent.action_space), genome
    )
    return final_state.positions, final_state.orientations


def _turn_that_fits(state, preferred: jax.Array, agent: GeneticAgent):
    """The genome's preferred turn if the macro fits anywhere in it, else the first turn it does.

    Rotation makes a placement reachable that the search would otherwise have to back out of: a
    macro turned onto its long side can leave the next one with nowhere legal to go. Without this
    the mask's own relaxation valve fires instead - it drops legality rather than deadlock - and
    the GA quietly discovers that overlapping macros have shorter wires. The fallback keeps the
    search where the constructive decode always was: inside the legal set.

    All four turns are evaluated rather than searched, because four maps under `vmap` is cheaper
    than a data-dependent loop inside a jitted scan, and the shape stays static.
    """
    turns = jnp.arange(N_ORIENTATIONS)
    illegal_per_turn = jax.vmap(lambda turn: agent._illegal(state, turn)[0])(turns)
    # A turn is usable when it leaves at least one legal cell.
    usable = ~illegal_per_turn.reshape(N_ORIENTATIONS, -1).all(axis=1)
    # The preferred turn wins when it is usable; otherwise the lowest-numbered turn that is. If
    # none is, argmax returns 0 and the mask's valve handles it exactly as it does elsewhere.
    fallback = jnp.argmax(usable)
    turn = jnp.where(usable[preferred], preferred, fallback)
    return turn, illegal_per_turn[turn]


def _decode_population(genomes: jax.Array, agent: GeneticAgent):
    """Every genome decoded and scored at once - the population method `replay` was built for."""
    positions, orientations = jax.vmap(lambda genome: _decode(genome, agent))(genomes)
    # Scored by the run's CONFIGURED reward, replayed through the same step() a policy drives, so
    # swapping the reward moves what the GA optimizes exactly as it moves what PPO optimizes -
    # WITH the turns each genome chose, or selection would run on a fitness that cannot see the
    # third gene at all and orientation would drift on legality alone.
    returns = (
        jax.vmap(agent.score)(positions, orientations) if orientations is not None
        else jax.vmap(agent.score)(positions)
    )
    if orientations is None:
        orientations = jnp.zeros(
            (genomes.shape[0], agent.benchmark.params.n_macros), dtype=jnp.int32
        )
    return positions, orientations, returns


def _breed(key: jax.Array, elites: jax.Array, n_children: int) -> jax.Array:
    """Uniform crossover: each gene taken from one of two randomly chosen elite parents."""
    n_elite = elites.shape[0]
    parent_key, mix_key = jax.random.split(key)
    parents = jax.random.randint(parent_key, (2, n_children), 0, n_elite)
    first, second = elites[parents[0]], elites[parents[1]]
    take_first = jax.random.bernoulli(mix_key, 0.5, first.shape)
    return jnp.where(take_first, first, second)


def _mutate(key: jax.Array, genomes: jax.Array, rate: float, scale: float) -> jax.Array:
    """Gaussian jitter on a random subset of genes, wrapped back into the unit interval.

    Wrapped rather than clipped: clipping piles probability mass onto 0 and 1, which decode to the
    canvas corners, so a mutation-heavy run would drift every macro towards the edges for a reason
    that has nothing to do with the objective.
    """
    mask_key, noise_key = jax.random.split(key)
    mutate = jax.random.bernoulli(mask_key, rate, genomes.shape)
    noise = jax.random.normal(noise_key, genomes.shape) * scale
    return jnp.mod(jnp.where(mutate, genomes + noise, genomes), 1.0)


__all__ = ["GeneticAgent"]
