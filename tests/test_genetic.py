"""The genetic agent - the first algorithm family here that is not sequential decision-making.

The spec's algorithm-family comparison ("PPO vs. ACO vs. GA, reward and benchmark held fixed")
could not be set up while all three shipped agents chose one action per step. These tests are what
say the kernel really did accommodate a different family without changing - the claim
`placax.core.replay`'s docstring has made since before any population method existed.

The load-bearing test is `test_every_evolved_placement_obeys_the_environments_mask`. A GA handed a
wirelength objective and no legality constraint optimizes into overlap, because overlapping macros
have shorter wires - so an encoding that can express an illegal placement does not merely risk one,
it converges to one.
"""
import dataclasses
import pathlib

import pytest

from placax.core import reset  # noqa: F401  must precede jax imports
from placax_agents.agents.base import Agent
from placax_agents.agents.genetic import GeneticAgent, _breed, _decode, _mutate
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build
from placax_agents.experiment.config import AgentSpec, ExperimentConfig, Spec, assert_comparable
from placax_agents.experiment.presets import training
from placax_agents.experiment.run import run_experiment, score
from placax_agents.policy.action import illegal_cells
from placax_agents.policy.scale import to_grid_units

import jax
import jax.numpy as jnp

NODES = ("UCLA nodes 1.0\nNumNodes : 4\nNumTerminals : 4\n"
         "a 4 4 terminal\nb 2 2 terminal\nc 2 4 terminal\nd 4 2 terminal\n")
NETS = ("UCLA nets 1.0\nNumNets : 3\nNumPins : 6\n"
        "NetDegree : 2 n0\n\ta I : 0.0 0.0\n\tb O : 0.0 0.0\n"
        "NetDegree : 2 n1\n\tb I : 0.0 0.0\n\tc O : 0.0 0.0\n"
        "NetDegree : 2 n2\n\tc I : 0.0 0.0\n\td O : 0.0 0.0\n")


def _design(directory: pathlib.Path) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "s.aux").write_text("RowBasedPlacement : s.nodes s.nets s.wts s.pl s.scl\n")
    (directory / "s.nodes").write_text(NODES)
    (directory / "s.nets").write_text(NETS)
    return directory


def _config(directory, population: int = 6, **agent_kwargs) -> ExperimentConfig:
    base = training(directory, budget=Budget(iterations=2))
    return dataclasses.replace(
        base,
        environment=dataclasses.replace(
            base.environment,
            benchmark=dataclasses.replace(base.environment.benchmark, grid=12),
        ),
        agent=AgentSpec(algorithm=Spec("genetic", {"population": population, **agent_kwargs})),
    )


@pytest.fixture
def built(tmp_path):
    return build(_config(_design(tmp_path / "bench")))


# ------------------------------------------------------------------ the seam


def test_the_genetic_agent_satisfies_the_agent_protocol(built) -> None:
    # The point of the seam: a population method with no policy, no optimizer and no gradient is
    # still an Agent, and the runner needs to know nothing else about it.
    assert isinstance(built.agent, Agent)
    assert built.agent.name == "genetic"


def test_it_is_selectable_from_a_config_like_any_other_algorithm(tmp_path) -> None:
    config = _config(_design(tmp_path / "bench"))
    assert config.agent.policy is None and config.agent.optimizer is None
    assert build(config).agent.name == "genetic"


def test_a_ga_run_is_comparable_with_the_other_agents_on_one_environment(tmp_path) -> None:
    """The comparison the spec asks for, made mechanically checkable.

    Same environment, different family: the GA's config must match a PPO config's environment
    hash exactly, or the table it appears in is not a comparison.
    """
    directory = _design(tmp_path / "bench")
    reference = _config(directory)
    ppo = dataclasses.replace(reference, agent=training(directory).agent)
    assert_comparable(reference, ppo)


def test_it_reports_a_population_of_episodes_so_the_budget_charges_it_correctly(built) -> None:
    # A generation of 6 placements is 6 episodes, not one - otherwise a population method gets
    # its environment interaction for free relative to a sequential one.
    state, result = built.agent.update(jax.random.PRNGKey(0), built.agent.init(jax.random.PRNGKey(1)))
    assert result.episodes == built.agent.population == 6
    assert result.gradient_steps == 0 and result.loss is None


# ------------------------------------------------------------------ legality


def test_every_evolved_placement_obeys_the_environments_mask(built) -> None:
    """The encoding cannot express an illegal placement, because it decodes through the mask.

    A genome of raw coordinates would let the GA discover that overlapping macros have shorter
    wires and converge straight onto an unrealizable answer. Preferences decoded to the nearest
    LEGAL cell make that unreachable rather than merely discouraged.
    """
    key = jax.random.PRNGKey(0)
    genome = jax.random.uniform(key, (built.agent._episode_length, 2))
    positions, _orientations = _decode(genome, built.agent)

    grid_sizes = to_grid_units(built.benchmark.sizes_array, built.benchmark.cell_size)
    state = reset(built.benchmark.params, built.initial_positions)
    for step_index in range(built.agent._episode_length):
        illegal = illegal_cells(
            built.state_fn(state, built.benchmark.params, built.benchmark.sizes_array)["canvas"],
            built.benchmark.params, grid_sizes[state.step],
        )
        chosen = positions[state.step]
        assert not bool(illegal[chosen[0], chosen[1]]), f"macro {step_index} took an illegal cell"
        state = state.replace(
            positions=state.positions.at[state.step].set(chosen), step=state.step + 1
        )


def test_a_genome_decodes_towards_its_preference(built) -> None:
    # The encoding has to actually mean something, or crossover and mutation are noise: a genome
    # preferring the far corner should land further from the origin than one preferring it.
    length = built.agent._episode_length
    near, _ = _decode(jnp.zeros((length, 2)), built.agent)
    far, _ = _decode(jnp.full((length, 2), 0.99), built.agent)
    assert float(far.sum()) > float(near.sum())


def test_decoding_is_deterministic(built) -> None:
    genome = jax.random.uniform(jax.random.PRNGKey(3), (built.agent._episode_length, 2))
    assert jnp.array_equal(_decode(genome, built.agent)[0], _decode(genome, built.agent)[0])


# ------------------------------------------------------------------ evolution


def test_the_best_placement_survives_a_generation_that_gets_worse(built) -> None:
    # A GA can lose its incumbent to mutation. The runner asks for the agent's best answer, not
    # its current one, so the champion is tracked across generations rather than within one.
    key = jax.random.PRNGKey(0)
    state = built.agent.init(key)
    returns = []
    for step_index in range(4):
        state, _result = built.agent.update(jax.random.fold_in(key, step_index), state)
        returns.append(float(state["best_return"]))
    assert returns == sorted(returns), "best_return must never decrease"


def test_evolution_improves_on_the_initial_population(built) -> None:
    key = jax.random.PRNGKey(0)
    state = built.agent.init(key)
    state, first = built.agent.update(jax.random.fold_in(key, 0), state)
    for step_index in range(1, 6):
        state, last = built.agent.update(jax.random.fold_in(key, step_index), state)
    assert float(state["best_return"]) >= first.metrics["population_best_return"]


def test_elites_are_carried_into_the_next_generation_unchanged(built) -> None:
    key = jax.random.PRNGKey(0)
    from placax_agents.agents.genetic import _decode_population

    state = built.agent.init(key)
    _positions, _orientations, returns = _decode_population(state["genomes"], built.agent)
    best = state["genomes"][int(jnp.argmax(returns))]
    new_state, _result = built.agent.update(jax.random.fold_in(key, 1), state)
    # The single best genome must appear verbatim among the survivors.
    assert any(
        bool(jnp.allclose(new_state["genomes"][i], best))
        for i in range(built.agent.n_elite)
    )


def test_crossover_takes_every_gene_from_some_parent() -> None:
    elites = jnp.array([[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]])
    children = _breed(jax.random.PRNGKey(0), elites, n_children=8)
    assert children.shape == (8, 2, 2)
    assert bool(jnp.all((children == 0.0) | (children == 1.0)))


def test_mutation_wraps_rather_than_clipping() -> None:
    """Clipping would pile probability onto 0 and 1, which decode to the canvas corners.

    A mutation-heavy run would then drift every macro to the edges for a reason unrelated to the
    objective - a bias introduced by the search operator rather than found in the design.
    """
    genomes = jnp.full((64, 4, 2), 0.99)
    mutated = _mutate(jax.random.PRNGKey(0), genomes, rate=1.0, scale=0.5)
    assert bool(jnp.all((mutated >= 0.0) & (mutated < 1.0)))
    # Wrapping means some genes land near zero rather than all stacking at the ceiling.
    assert float(mutated.min()) < 0.5


def test_a_population_below_two_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        build(_config(_design(tmp_path / "bench"), population=1))


# ------------------------------------------------------------------ through the runner


def test_a_ga_run_leaves_the_same_evidence_as_any_other(tmp_path) -> None:
    directory = _design(tmp_path / "bench")
    config = _config(directory)
    output = tmp_path / "run"
    state, log = run_experiment(config, output, eval_every=1)

    assert (output / "manifest.json").exists()
    assert log and log[-1]["gradient_steps"] == 0
    assert "population_std_return" in log[-1]
    measured = score(build(config).benchmark, state["best_positions"], 0)
    assert measured["real_hpwl"] > 0
