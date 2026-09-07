"""Every agent runs inside the environment its config describes - not a convenient subset of it.

This is the test that was missing, and its absence let a real hole sit open. The agent fixtures
elsewhere all build from `presets.training`, whose `action_mask` is None, so nothing ever
exercised an agent under a config that DECLARES a constraint. Under `presets.maskplace`, which
declares `wiremask_quality`, the two baseline builders dropped the resolved mask and both
baselines scored candidates with bare HPWL instead of the configured reward - so PPO sampled from
a wiremask-restricted action set and its baselines sampled from the full legal one, while
`assert_comparable` reported the two environments as identical.

That is the worst possible failure for this project specifically: a false negative in the one
mechanism that exists to stop incomparable numbers reaching a table. So the property is asserted
directly - for every agent, over an environment that actually constrains - rather than left to
each builder to remember.
"""
import dataclasses
import pathlib

from placax.core import replay, reset  # noqa: F401  must precede jax imports
from placax.extras.legality import legality
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build, build_benchmark
from placax_agents.experiment.config import AgentSpec, Spec, assert_comparable
from placax_agents.experiment.presets import maskplace
from placax_agents.policy.action import illegal_cells
from placax_agents.policy.scale import to_grid_units
from placax_agents.training.reward import make_scaled_hpwl_reward

import jax.numpy as jnp
import pytest
from jax import random

AGENT_SPECS = {
    "ppo": None,  # taken from the reference config
    "greedy_wiremask": AgentSpec(algorithm=Spec("greedy_wiremask")),
    "random_search": AgentSpec(algorithm=Spec("random_search", {"population": 2})),
}


def _write_tiny_bookshelf(tmp_path: pathlib.Path) -> pathlib.Path:
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    (benchmark_dir / "s.aux").write_text(
        "RowBasedPlacement : s.nodes s.nets s.wts s.pl s.scl\n"
    )
    (benchmark_dir / "s.nodes").write_text(
        "UCLA nodes 1.0\nNumNodes : 4\nNumTerminals : 4\n"
        "a 4 4 terminal\nb 2 2 terminal\nc 2 4 terminal\nd 4 2 terminal\n"
    )
    (benchmark_dir / "s.nets").write_text(
        "UCLA nets 1.0\nNumNets : 3\nNumPins : 6\n"
        "NetDegree : 2 n0\n\ta I : 0.0 0.0\n\tb O : 0.0 0.0\n"
        "NetDegree : 2 n1\n\tb I : 0.0 0.0\n\tc O : 0.0 0.0\n"
        "NetDegree : 2 n2\n\tc I : 0.0 0.0\n\td O : 0.0 0.0\n"
    )
    return benchmark_dir


@pytest.fixture
def masked_env(tmp_path: pathlib.Path):
    """A config that DECLARES an action mask - the case no other agent test covers."""
    benchmark_dir = _write_tiny_bookshelf(tmp_path)
    config = maskplace(benchmark_dir, budget=Budget(iterations=1))
    config = dataclasses.replace(config, environment=dataclasses.replace(
        config.environment,
        benchmark=dataclasses.replace(config.environment.benchmark, grid=24),
    ))
    assert config.environment.action_mask is not None, "fixture must actually constrain"
    return config, build_benchmark(config)


def _with_agent(config, name: str):
    agent = config.agent if AGENT_SPECS[name] is None else AGENT_SPECS[name]
    return dataclasses.replace(config, name=name, agent=agent)


@pytest.mark.parametrize("algorithm", sorted(AGENT_SPECS))
def test_every_agent_receives_the_environments_action_mask(masked_env, algorithm: str) -> None:
    # The regression itself: a builder that drops the mask produces an agent playing a different
    # game from the one it is about to be compared against.
    config, benchmark = masked_env
    built = build(_with_agent(config, algorithm), benchmark=benchmark)
    assert built.extra_illegal_fn is not None
    assert built.agent.extra_illegal_fn is built.extra_illegal_fn, (
        f"{algorithm} was built without the environment's action mask, so it is not running in "
        f"the environment its config describes"
    )


@pytest.mark.parametrize("algorithm", sorted(AGENT_SPECS))
def test_every_agent_receives_the_environments_observation(masked_env, algorithm: str) -> None:
    config, benchmark = masked_env
    built = build(_with_agent(config, algorithm), benchmark=benchmark)
    assert built.agent.state_fn is built.state_fn


@pytest.mark.parametrize("algorithm", sorted(AGENT_SPECS))
def test_swapping_the_agent_never_changes_the_environment(masked_env, algorithm: str) -> None:
    config, benchmark = masked_env
    swapped = _with_agent(config, algorithm)
    assert_comparable(config, swapped)
    # And it must hold at every level, not only the default one.
    for level in ("benchmark", "task", "environment"):
        assert_comparable(config, swapped, level=level)


def test_baselines_rank_candidates_by_the_configured_reward_not_by_hpwl(masked_env) -> None:
    """Swapping the reward must move what EVERY agent optimizes, not just the one that reads it.

    Random search picks its incumbent by score, so if it scored with bare HPWL it would keep the
    same placement no matter what reward the config named - the reward axis would be swappable
    for PPO alone. Here the same population is ranked under two different rewards, and the
    incumbent's recorded score has to follow the reward it was told to use.
    """
    config, benchmark = masked_env
    built = build(_with_agent(config, "random_search"), benchmark=benchmark)
    agent = built.agent

    state, _result = agent.update(random.PRNGKey(0), agent.init(random.PRNGKey(0)))
    recorded = float(state["best_return"])

    # The score the agent recorded must be the configured reward's own return for that placement.
    replayed = float(replay(state["best_positions"], benchmark.reward_fn, benchmark.params))
    assert replayed == pytest.approx(recorded, rel=1e-5)

    # And it must NOT be the bare-HPWL reward's return, which is what it used to compute. (The
    # maskplace reward is a scaled, divided HPWL delta, so the two differ by construction.)
    bare = make_scaled_hpwl_reward(
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
        benchmark.sizes_array, benchmark.cell_size,
    )
    assert float(replay(state["best_positions"], bare, benchmark.params)) != pytest.approx(
        recorded, rel=1e-3
    )


@pytest.mark.parametrize("algorithm", sorted(AGENT_SPECS))
def test_every_agents_placement_obeys_the_mask_wherever_the_mask_left_a_choice(
    masked_env, algorithm: str
) -> None:
    """Replay each placement and check every step landed on a cell the run's rules allowed.

    Cells are checked against `illegal_cells` - the same function the policy's own masking calls -
    at the state each macro was actually placed from. The relaxation valve means "no legal cell"
    is a real outcome, and in that case the map comes back all-False, so this stays a genuine
    assertion rather than one the valve silently satisfies.
    """
    config, benchmark = masked_env
    built = build(_with_agent(config, algorithm), benchmark=benchmark)
    agent = built.agent
    state = agent.init(random.PRNGKey(0))
    state, _result = agent.update(random.PRNGKey(0), state)
    positions = agent.best_positions(state)

    env_state = reset(benchmark.params, built.initial_positions)
    for i in range(built.n_placed, benchmark.params.n_macros):
        obs = built.state_fn(env_state, benchmark.params, benchmark.sizes_array)
        macro_size = to_grid_units(obs["current_macro_size"], benchmark.cell_size)
        illegal = illegal_cells(
            obs["canvas"], benchmark.params, macro_size,
            built.extra_illegal_fn(obs) if built.extra_illegal_fn else None,
        )
        x, y = int(positions[i][0]), int(positions[i][1])
        assert not bool(illegal[x, y]), (
            f"{algorithm} placed macro {i} at ({x}, {y}), which the run's own rules forbid"
        )
        env_state = env_state.replace(
            positions=env_state.positions.at[i].set(positions[i]), step=env_state.step + 1
        )


def test_legality_is_measured_and_reported_for_every_agent(masked_env) -> None:
    # A wirelength number for an overlapping placement is not a result, so legality has to be
    # something a run reports rather than something a reader assumes.
    config, benchmark = masked_env
    grid_sizes = to_grid_units(benchmark.sizes_array, benchmark.cell_size)
    for algorithm in sorted(AGENT_SPECS):
        built = build(_with_agent(config, algorithm), benchmark=benchmark)
        agent = built.agent
        state, _ = agent.update(random.PRNGKey(0), agent.init(random.PRNGKey(0)))
        measured = legality(agent.best_positions(state), grid_sizes, benchmark.params)
        assert int(measured.n_unplaced) == 0
        assert float(measured.overlap_area) >= 0.0
        assert measured.to_dict()["is_legal"] in (True, False)


def test_legality_catches_a_deliberately_overlapping_placement(masked_env) -> None:
    # The measurement has to be able to fail, or reporting it proves nothing.
    _config, benchmark = masked_env
    grid_sizes = to_grid_units(benchmark.sizes_array, benchmark.cell_size)
    stacked = jnp.zeros((benchmark.params.n_macros, 2), dtype=jnp.int32)  # every macro at (0, 0)
    measured = legality(stacked, grid_sizes, benchmark.params)
    assert not bool(measured.is_legal)
    assert float(measured.overlap_area) > 0
