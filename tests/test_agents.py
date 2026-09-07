"""The Agent seam, and the two non-learning agents that prove it isn't just PPO in a costume."""
import pathlib

from placax.core import reset, step  # noqa: F401  must precede jax imports
from placax.extras.masks import boundary_mask, occupancy_mask
from placax.extras.render import render
from placax_agents.agents import Agent, GreedyWiremaskAgent, PPOAgent, RandomSearchAgent
from placax_agents.agents.base import UpdateResult
from placax_agents.experiment import Budget, assert_comparable, presets, run_experiment
from placax_agents.experiment.build import build, build_benchmark
from placax_agents.experiment.run import score_placement
from placax_agents.policy.scale import to_grid_units

import jax.numpy as jnp
import pytest
from jax import random


def _write_tiny_bookshelf(tmp_path: pathlib.Path) -> pathlib.Path:
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    (benchmark_dir / "sample.aux").write_text(
        "RowBasedPlacement : sample.nodes sample.nets sample.wts sample.pl sample.scl\n"
    )
    (benchmark_dir / "sample.nodes").write_text(
        "UCLA nodes 1.0\nNumNodes : 4\nNumTerminals : 4\n"
        "a 4 4 terminal\nb 2 2 terminal\nc 2 4 terminal\nd 4 2 terminal\n"
    )
    (benchmark_dir / "sample.nets").write_text(
        "UCLA nets 1.0\nNumNets : 3\nNumPins : 6\n"
        "NetDegree : 2 n0\n\ta I : 0.0 0.0\n\tb O : 0.0 0.0\n"
        "NetDegree : 2 n1\n\tb I : 0.0 0.0\n\tc O : 0.0 0.0\n"
        "NetDegree : 2 n2\n\tc I : 0.0 0.0\n\td O : 0.0 0.0\n"
    )
    return benchmark_dir


@pytest.fixture
def env(tmp_path: pathlib.Path):
    """A small shared environment plus its loaded benchmark, reused by every agent in a test."""
    benchmark_dir = _write_tiny_bookshelf(tmp_path)
    config = presets.training(benchmark_dir, budget=Budget(iterations=2))
    small = type(config.environment.benchmark)(
        benchmark_dir=str(benchmark_dir), grid=8, macro_budget=None,
        order=config.environment.benchmark.order,
    )
    config = type(config)(
        name=config.name, seed=config.seed, agent=config.agent,
        environment=type(config.environment)(
            benchmark=small, reward=config.environment.reward, state=config.environment.state,
            action_mask=config.environment.action_mask, budget=config.environment.budget,
        ),
    )
    return config, build_benchmark(config)


def _no_overlap(benchmark, positions) -> bool:
    """Whether a placement puts every macro on cells no earlier macro already occupies."""
    grid_sizes = to_grid_units(benchmark.sizes_array, benchmark.cell_size)
    params = benchmark.params
    occupied = jnp.zeros((params.grid_x, params.effective_grid_y), dtype=bool)
    for i in range(params.n_macros):
        size = grid_sizes[i]
        footprint = render(
            positions[i][None, :], size[None, :], params.grid_x, params.effective_grid_y
        )
        if bool((occupied & footprint).any()):
            return False
        occupied = occupied | footprint
    return True


def _within_canvas(benchmark, positions) -> bool:
    grid_sizes = to_grid_units(benchmark.sizes_array, benchmark.cell_size)
    params = benchmark.params
    upper = positions + grid_sizes
    return bool(
        (positions >= 0).all()
        and (upper[:, 0] <= params.grid_x).all()
        and (upper[:, 1] <= params.effective_grid_y).all()
    )


# --------------------------------------------------------------------------- protocol


@pytest.mark.parametrize("algorithm", ["ppo", "greedy_wiremask", "random_search"])
def test_every_agent_satisfies_the_protocol(env, algorithm: str) -> None:
    config, benchmark = env
    swapped = _with_agent(config, algorithm)
    built = build(swapped, benchmark=benchmark)
    assert isinstance(built.agent, Agent)


def _with_agent(config, algorithm: str, **kwargs):
    """The same config with a different agent - the only axis a comparison is allowed to vary."""
    from placax_agents.experiment.config import AgentSpec, Spec

    if algorithm == "ppo":
        return config
    return type(config)(
        name=f"{algorithm}-test", seed=config.seed, environment=config.environment,
        agent=AgentSpec(algorithm=Spec(algorithm, kwargs)),
    )


@pytest.mark.parametrize("algorithm", ["ppo", "greedy_wiremask", "random_search"])
def test_every_agent_produces_a_legal_placement(env, algorithm: str) -> None:
    # The kernel's legality guarantee has to hold whatever chose the action - a policy argmax, a
    # uniform draw over legal cells, or a greedy wirelength scan.
    config, benchmark = env
    built = build(_with_agent(config, algorithm, **({"population": 4} if algorithm == "random_search" else {})),
                  benchmark=benchmark)
    state = built.agent.init(random.PRNGKey(0))
    state, _result = built.agent.update(random.PRNGKey(1), state)
    positions = built.agent.best_positions(state)

    assert positions.shape == (benchmark.params.n_macros, 2)
    assert _within_canvas(benchmark, positions)
    assert _no_overlap(benchmark, positions)


@pytest.mark.parametrize("algorithm", ["ppo", "greedy_wiremask", "random_search"])
def test_every_agent_reports_the_episodes_it_spent(env, algorithm: str) -> None:
    # The budget is charged from this number, so an agent under-reporting it would silently buy
    # itself extra compute in a comparison.
    config, benchmark = env
    built = build(_with_agent(config, algorithm), benchmark=benchmark)
    state = built.agent.init(random.PRNGKey(0))
    _state, result = built.agent.update(random.PRNGKey(1), state)
    assert isinstance(result, UpdateResult)
    assert result.episodes >= 1
    assert result.episodes == built.episodes_per_iteration


# --------------------------------------------------------------------------- baselines


def test_greedy_wiremask_is_deterministic(env) -> None:
    config, benchmark = env
    a = GreedyWiremaskAgent(benchmark)
    b = GreedyWiremaskAgent(benchmark)
    assert (a.best_positions(a.init(random.PRNGKey(0)))
            == b.best_positions(b.init(random.PRNGKey(99)))).all()


def test_greedy_wiremask_beats_a_single_random_draw(env) -> None:
    # Not a tuning claim - just that the heuristic is actually doing something. If greedy ever
    # scored worse than one random placement, the wiremask scoring would be wired up backwards.
    config, benchmark = env
    greedy = GreedyWiremaskAgent(benchmark)
    greedy_score = score_placement(benchmark, greedy.best_positions(greedy.init(random.PRNGKey(0))))

    rand = RandomSearchAgent(benchmark, population=1)
    scores = []
    for seed in range(8):
        state, _ = rand.update(random.PRNGKey(seed), rand.init(random.PRNGKey(0)))
        scores.append(score_placement(benchmark, state["best_positions"]))
    assert greedy_score <= min(scores)


def test_random_search_keeps_the_best_placement_across_updates(env) -> None:
    config, benchmark = env
    agent = RandomSearchAgent(benchmark, population=4)
    state = agent.init(random.PRNGKey(0))
    best = float("inf")
    for seed in range(4):
        state, _result = agent.update(random.PRNGKey(seed), state)
        score = float(state["best_hpwl"])
        assert score <= best  # monotonically non-increasing: an incumbent is never given up
        best = score
    # And the recorded incumbent must be the placement that actually earned that score.
    assert score_placement(benchmark, state["best_positions"]) == pytest.approx(best, rel=1e-5)


def test_random_search_population_sets_its_episode_cost(env) -> None:
    config, benchmark = env
    agent = RandomSearchAgent(benchmark, population=7)
    _state, result = agent.update(random.PRNGKey(0), agent.init(random.PRNGKey(0)))
    assert result.episodes == 7


def test_baselines_report_no_loss(env) -> None:
    # They don't train, and reporting a fake 0.0 loss would make a log look like training.
    config, benchmark = env
    for agent in (GreedyWiremaskAgent(benchmark), RandomSearchAgent(benchmark, population=2)):
        _state, result = agent.update(random.PRNGKey(0), agent.init(random.PRNGKey(0)))
        assert result.loss is None


# --------------------------------------------------------------------------- comparability


def test_swapping_the_agent_leaves_the_environment_identical(env) -> None:
    # The property every comparison in this project rests on.
    config, _benchmark = env
    variants = [config, _with_agent(config, "greedy_wiremask"), _with_agent(config, "random_search")]
    assert_comparable(*variants)
    assert len({c.full_hash() for c in variants}) == 3   # but they are still distinct runs


def test_three_agents_run_to_the_same_env_step_budget(env, tmp_path: pathlib.Path) -> None:
    # The headline capability: different methods, one environment, one budget, one scoring path.
    config, benchmark = env
    budget = Budget(env_steps=benchmark.params.n_macros * 8)
    configs = [
        _with_agent(_rebudget(config, budget), "greedy_wiremask"),
        _with_agent(_rebudget(config, budget), "random_search", population=2),
        _rebudget(config, budget),
    ]
    spends = []
    for cfg in configs:
        built = build(cfg, benchmark=benchmark)
        state, log = run_experiment(
            cfg, tmp_path / cfg.name, built=built, eval_every=10 ** 9, log_every=10 ** 9
        )
        spends.append(log[-1]["env_steps"])
        # Every agent's placement is scored by the runner, not by itself.
        assert score_placement(benchmark, built.agent.best_positions(state)) > 0
    assert len(set(spends)) == 1
    assert spends[0] <= budget.env_steps


def _rebudget(config, budget: Budget):
    return type(config)(
        name=config.name, seed=config.seed, agent=config.agent,
        environment=type(config.environment)(
            benchmark=config.environment.benchmark, reward=config.environment.reward,
            state=config.environment.state, action_mask=config.environment.action_mask,
            budget=budget,
        ),
    )


def test_a_baseline_run_writes_the_same_evidence_as_a_training_run(env, tmp_path: pathlib.Path) -> None:
    # A baseline that skipped the manifest would be exactly the unattributable result this
    # project is trying to stop producing.
    config, benchmark = env
    cfg = _with_agent(config, "greedy_wiremask")
    built = build(cfg, benchmark=benchmark)
    output_dir = tmp_path / "baseline"
    _state, log = run_experiment(cfg, output_dir, built=built, eval_every=1)

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "training_log.jsonl").exists()
    assert log[-1]["full_hash"] == cfg.full_hash()
    assert log[-1]["real_hpwl"] is not None
    # No policy weights exist, so no best_checkpoint should be invented for one.
    assert not (output_dir / "best_checkpoint.bin").exists()


def test_init_from_is_refused_for_an_agent_without_weights(env, tmp_path: pathlib.Path) -> None:
    config, benchmark = env
    cfg = _with_agent(config, "greedy_wiremask")
    built = build(cfg, benchmark=benchmark)
    with pytest.raises(ValueError, match="does not have"):
        run_experiment(cfg, tmp_path / "x", built=built, init_from=tmp_path / "nope.bin")


def test_ppo_agent_state_carries_exactly_what_ppo_needs_to_resume(env) -> None:
    config, benchmark = env
    built = build(config, benchmark=benchmark)
    assert isinstance(built.agent, PPOAgent)
    state = built.agent.init(random.PRNGKey(0))
    assert set(state) == {"variables", "opt_state", "running_stats"}
