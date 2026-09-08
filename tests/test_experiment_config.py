"""ExperimentConfig: JSON round-tripping, hashing, and the comparability contract."""
import json
import pathlib

import pytest

from placax_agents.experiment import (  # noqa: F401  must precede jax imports
    AgentSpec, BenchmarkSpec, Budget, EnvironmentSpec, ExperimentConfig, Spec, assert_comparable,
)
from placax_agents.experiment.budget import BudgetTracker, BudgetUse
from placax_agents.experiment.presets import maskplace, training


def _config(**overrides) -> ExperimentConfig:
    base = dict(
        name="test",
        seed=1,
        environment=EnvironmentSpec(
            benchmark=BenchmarkSpec("benchmarks/adaptec1", grid=32, macro_budget=8),
            reward=Spec("hpwl", {"dense": True}),
            state=Spec("canvas"),
            budget=Budget(iterations=3),
        ),
        agent=AgentSpec(
            policy=Spec("cnn"), algorithm=Spec("ppo"), optimizer=Spec("adam"),
            loop=Spec("sequential"),
        ),
    )
    base.update(overrides)
    return ExperimentConfig(**base)


# --------------------------------------------------------------------------- serialization


def test_config_round_trips_through_json_losslessly() -> None:
    # A config that can't survive a results file isn't a record of anything.
    config = _config()
    assert ExperimentConfig.from_json(config.to_json()) == config


def test_written_config_carries_its_hashes_for_readers_that_dont_import_placax() -> None:
    config = _config()
    data = json.loads(config.to_json())
    assert data["environment_hash"] == config.environment_hash()
    assert data["full_hash"] == config.full_hash()


def test_config_write_and_read_round_trip(tmp_path: pathlib.Path) -> None:
    config = _config()
    path = tmp_path / "nested" / "config.json"
    config.write(path)
    assert ExperimentConfig.read(path) == config


def test_spec_accepts_a_bare_string_as_shorthand() -> None:
    assert Spec.from_dict("cnn") == Spec("cnn", {})
    assert Spec.from_dict(None) is None


# --------------------------------------------------------------------------- hashing


def test_environment_hash_ignores_the_agent_and_the_seed() -> None:
    # The whole point: swapping the agent must leave the environment identical, so two agents
    # can be compared. If this ever stops holding, no comparison in this project is valid.
    base = _config()
    other_agent = _config(agent=AgentSpec(
        policy=Spec("resnet_coarse_fine"), algorithm=Spec("ppo", {"gamma": 0.5}),
        optimizer=Spec("adam", {"learning_rate": 1.0}), loop=Spec("buffered"),
    ))
    other_seed = _config(seed=999)
    assert base.environment_hash() == other_agent.environment_hash() == other_seed.environment_hash()
    assert base.full_hash() != other_agent.full_hash()
    assert base.full_hash() != other_seed.full_hash()


@pytest.mark.parametrize("field, value", [
    ("grid", 64),
    ("macro_budget", 16),
])
def test_environment_hash_changes_when_the_benchmark_changes(field: str, value) -> None:
    benchmark_kwargs = {"grid": 32, "macro_budget": 8, field: value}
    changed = _config(environment=EnvironmentSpec(
        benchmark=BenchmarkSpec("benchmarks/adaptec1", **benchmark_kwargs),
        reward=Spec("hpwl", {"dense": True}), state=Spec("canvas"), budget=Budget(iterations=3),
    ))
    assert changed.environment_hash() != _config().environment_hash()


def test_environment_hash_changes_when_the_budget_changes() -> None:
    # Compute budget is part of the comparability contract, not a run-control detail: two agents
    # given different budgets did not have a fair fight.
    changed = _config(environment=EnvironmentSpec(
        benchmark=BenchmarkSpec("benchmarks/adaptec1", grid=32, macro_budget=8),
        reward=Spec("hpwl", {"dense": True}), state=Spec("canvas"),
        budget=Budget(iterations=4),
    ))
    assert changed.environment_hash() != _config().environment_hash()


def test_hash_is_stable_across_key_ordering() -> None:
    a = _config(environment=EnvironmentSpec(
        benchmark=BenchmarkSpec("benchmarks/adaptec1", grid=32, macro_budget=8),
        reward=Spec("hpwl", {"dense": True, "reward_scale": 2.0}), state=Spec("canvas"),
        budget=Budget(iterations=3),
    ))
    b = _config(environment=EnvironmentSpec(
        benchmark=BenchmarkSpec("benchmarks/adaptec1", grid=32, macro_budget=8),
        reward=Spec("hpwl", {"reward_scale": 2.0, "dense": True}), state=Spec("canvas"),
        budget=Budget(iterations=3),
    ))
    assert a.environment_hash() == b.environment_hash()


# --------------------------------------------------------------------------- comparability


def test_assert_comparable_accepts_two_agents_in_one_environment() -> None:
    ppo = _config()
    other = _config(agent=AgentSpec(
        policy=Spec("resnet_coarse_fine"), algorithm=Spec("ppo"),
        optimizer=Spec("maskplace_split"), loop=Spec("buffered"),
    ))
    assert_comparable(ppo, other)  # must not raise


def test_assert_comparable_rejects_differing_environments_and_names_the_differences() -> None:
    with pytest.raises(ValueError) as excinfo:
        assert_comparable(maskplace("benchmarks/adaptec1"), training("benchmarks/adaptec1"))
    message = str(excinfo.value)
    # The error has to be actionable - "not comparable" alone sends someone diffing two scripts,
    # which is the situation this whole mechanism exists to remove.
    assert "benchmark.grid: 224 != 64" in message
    assert "benchmark.order.name" in message
    assert "action_mask" in message


def test_assert_comparable_is_a_no_op_below_two_configs() -> None:
    assert_comparable()
    assert_comparable(_config())


# --------------------------------------------------------------------------- budget


def test_budget_requires_at_least_one_cap() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Budget()


@pytest.mark.parametrize("kwargs", [{"iterations": 0}, {"env_steps": -1}, {"wall_clock_s": 0.0}])
def test_budget_rejects_non_positive_caps(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Budget(**kwargs)


def test_tracker_charges_env_steps_as_episodes_times_steps_per_episode() -> None:
    # env_steps is the cross-agent currency; if this accounting is wrong, every budgeted
    # comparison is wrong with it.
    tracker = BudgetTracker(Budget(env_steps=100))
    use = tracker.record_iteration(episodes=10, steps_per_episode=4)
    assert (use.iterations, use.episodes, use.env_steps) == (1, 10, 40)
    tracker.record_iteration(episodes=10, steps_per_episode=4)
    assert tracker.exhausted() is None
    assert tracker.record_iteration(episodes=10, steps_per_episode=4).env_steps == 120
    assert tracker.exhausted() == "env_steps"


def test_tracker_reports_iteration_cap_when_that_is_what_binds() -> None:
    tracker = BudgetTracker(Budget(iterations=2))
    tracker.record_iteration(episodes=1, steps_per_episode=1)
    assert tracker.exhausted() is None
    tracker.record_iteration(episodes=1, steps_per_episode=1)
    assert tracker.exhausted() == "iterations"


def test_tracker_resumes_a_budget_rather_than_starting_a_fresh_one() -> None:
    # A resumed run continuing on a fresh budget would let anyone exceed a cap just by
    # restarting, which would make budgeted comparison meaningless.
    prior = BudgetUse(iterations=2, episodes=20, env_steps=80, wall_clock_s=12.0)
    tracker = BudgetTracker(Budget(env_steps=100), prior=prior)
    assert tracker.use.env_steps == 80
    assert tracker.use.wall_clock_s >= 12.0
    tracker.record_iteration(episodes=10, steps_per_episode=4)
    assert tracker.exhausted() == "env_steps"


def test_tracker_wall_clock_exhausts_immediately_past_a_prior_spend() -> None:
    tracker = BudgetTracker(Budget(wall_clock_s=1.0), prior=BudgetUse(wall_clock_s=5.0))
    assert tracker.exhausted() == "wall_clock_s"


def test_budget_use_round_trips_through_dict() -> None:
    use = BudgetUse(iterations=3, episodes=30, env_steps=120, wall_clock_s=1.5)
    assert BudgetUse.from_dict(use.to_dict()) == use


def test_read_accepts_a_runs_manifest_not_just_a_bare_config(tmp_path: pathlib.Path) -> None:
    """The manifest is the only file a run writes that holds a config, so it has to be readable.

    `write_manifest` nests the config under "config" alongside the metrics and the fingerprint,
    while `from_dict` read "name" off the top level - so every documented `--config` path, the
    attributable-PPA one in scripts/validate_design.py included, died on `KeyError: 'name'` when
    pointed at real run output.
    """
    from placax_agents.experiment.run import write_manifest

    config = _config()
    manifest_path = write_manifest(tmp_path / "run", config)
    assert json.loads(manifest_path.read_text())["config"]["name"] == config.name
    assert ExperimentConfig.read(manifest_path) == config


def test_read_still_accepts_a_bare_config_written_by_write(tmp_path: pathlib.Path) -> None:
    # Detected by shape, so both files work through one entry point and no caller has to say
    # which kind it holds.
    config = _config()
    path = tmp_path / "config.json"
    config.write(path)
    assert ExperimentConfig.read(path) == config
