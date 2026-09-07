"""run_experiment: the manifest it leaves behind, the budget it honors, and resuming."""
import json
import pathlib

from placax.core import reset  # noqa: F401  must precede jax imports
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build
from placax_agents.experiment.config import ExperimentConfig
from placax_agents.experiment.presets import training
from placax_agents.experiment.run import LOG_NAME, MANIFEST_NAME, STATE_NAME, run_experiment

import pytest


def _write_tiny_bookshelf(tmp_path: pathlib.Path) -> pathlib.Path:
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    (benchmark_dir / "sample.aux").write_text(
        "RowBasedPlacement : sample.nodes sample.nets sample.wts sample.pl sample.scl\n"
    )
    (benchmark_dir / "sample.nodes").write_text(
        "UCLA nodes 1.0\nNumNodes : 3\nNumTerminals : 3\n"
        "a 4 4 terminal\nb 2 2 terminal\nc 2 4 terminal\n"
    )
    (benchmark_dir / "sample.nets").write_text(
        "UCLA nets 1.0\nNumNets : 2\nNumPins : 4\n"
        "NetDegree : 2 n0\n\ta I : 0.0 0.0\n\tb O : 0.0 0.0\n"
        "NetDegree : 2 n1\n\tb I : 0.0 0.0\n\tc O : 0.0 0.0\n"
    )
    return benchmark_dir


@pytest.fixture
def toy(tmp_path: pathlib.Path):
    """A tiny real benchmark plus a config over it, built once and reused across a test."""
    benchmark_dir = _write_tiny_bookshelf(tmp_path)

    def make(budget: Budget, **overrides):
        config = training(benchmark_dir, budget=budget, **overrides)
        # Small grid: the preset's 64 costs compile time this test doesn't need.
        small = type(config.environment.benchmark)(
            benchmark_dir=str(benchmark_dir), grid=8, macro_budget=None,
            order=config.environment.benchmark.order,
        )
        config = type(config)(
            name=config.name, seed=config.seed, agent=config.agent,
            environment=type(config.environment)(
                benchmark=small, reward=config.environment.reward, state=config.environment.state,
                action_mask=config.environment.action_mask, budget=budget,
            ),
        )
        return config, build(config)

    return make


# --------------------------------------------------------------------------- provenance


def test_manifest_records_the_config_and_the_machine(toy, tmp_path: pathlib.Path) -> None:
    # The gap this closes: a training log used to hold {iteration, loss, real_hpwl} and nothing
    # else, so a result could not be attributed back to the configuration that produced it.
    config, built = toy(Budget(iterations=1))
    output_dir = tmp_path / "run"
    run_experiment(config, output_dir, built=built, eval_every=0)

    manifest = json.loads((output_dir / MANIFEST_NAME).read_text())
    assert ExperimentConfig.from_dict(manifest["config"]) == config
    assert manifest["config"]["environment_hash"] == config.environment_hash()
    fingerprint = manifest["fingerprint"]
    assert fingerprint["packages"]["jax"] is not None
    assert fingerprint["device"]["backend"] in ("cpu", "gpu", "tpu")
    assert isinstance(fingerprint["deterministic_backend"], bool)
    assert "started_at" in manifest


def test_manifest_is_written_before_training_so_a_crashed_run_is_still_attributable(
    toy, tmp_path: pathlib.Path
) -> None:
    config, built = toy(Budget(iterations=1))
    output_dir = tmp_path / "run"

    def exploding_step(*_args, **_kwargs):
        raise RuntimeError("boom")

    crashing = type(built)(
        config=built.config, benchmark=built.benchmark, policy=built.policy,
        state_fn=built.state_fn, extra_illegal_fn=built.extra_illegal_fn,
        optimizer=built.optimizer, ppo_config=built.ppo_config, step_fn=exploding_step,
        episodes_per_iteration=built.episodes_per_iteration,
    )
    with pytest.raises(RuntimeError, match="boom"):
        run_experiment(config, output_dir, built=crashing, eval_every=0)
    assert (output_dir / MANIFEST_NAME).exists()


def test_every_log_line_carries_the_run_hash_and_the_budget_spend(toy, tmp_path: pathlib.Path) -> None:
    config, built = toy(Budget(iterations=3))
    output_dir = tmp_path / "run"
    _variables, log = run_experiment(config, output_dir, built=built, eval_every=0)

    lines = [json.loads(line) for line in (output_dir / LOG_NAME).read_text().splitlines()]
    assert len(lines) == 3
    assert [entry["iteration"] for entry in lines] == [1, 2, 3]
    for entry in lines:
        assert entry["full_hash"] == config.full_hash()
        assert entry["env_steps"] == entry["iteration"] * built.env_steps_per_iteration
        assert entry["wall_clock_s"] >= 0.0
    assert log == lines


def test_output_dir_none_writes_nothing_but_still_trains(toy, tmp_path: pathlib.Path) -> None:
    config, built = toy(Budget(iterations=2))
    _variables, log = run_experiment(config, None, built=built, eval_every=0)
    assert len(log) == 2
    assert list(tmp_path.glob("**/manifest.json")) == []


# --------------------------------------------------------------------------- budget


def test_run_stops_on_the_iteration_budget(toy, tmp_path: pathlib.Path) -> None:
    config, built = toy(Budget(iterations=4))
    _variables, log = run_experiment(config, tmp_path / "run", built=built, eval_every=0)
    assert len(log) == 4


def test_run_stops_on_the_env_step_budget(toy, tmp_path: pathlib.Path) -> None:
    # env_steps is the unit that makes two different loop shapes comparable, so it has to bind
    # the loop directly rather than being converted into an iteration count somewhere.
    config, built = toy(Budget(env_steps=3 * 4))
    _variables, log = run_experiment(config, tmp_path / "run", built=built, eval_every=0)
    assert len(log) == 4
    assert log[-1]["env_steps"] == 12


def test_run_never_exceeds_an_env_step_budget_that_divides_unevenly(
    toy, tmp_path: pathlib.Path
) -> None:
    # The reason can_afford() checks before the iteration rather than after: an iteration is
    # atomic, so a post-hoc check overshoots by a whole iteration - and by a DIFFERENT amount for
    # each loop shape, which silently un-does the budget's whole purpose.
    config, built = toy(Budget(env_steps=11))   # 3 macros/episode, so 3 per iteration
    _variables, log = run_experiment(config, tmp_path / "run", built=built, eval_every=0)
    assert log[-1]["env_steps"] == 9
    assert log[-1]["env_steps"] <= 11


def test_a_budget_too_small_for_one_iteration_is_an_error_not_an_empty_run(
    toy, tmp_path: pathlib.Path
) -> None:
    # Exiting successfully having trained nothing would look like a finished run in a results
    # directory, which is worse than failing.
    config, built = toy(Budget(env_steps=1))
    with pytest.raises(ValueError, match="cannot afford one iteration"):
        run_experiment(config, tmp_path / "run", built=built, eval_every=0)


def test_two_loop_shapes_get_the_same_env_steps_from_one_budget(toy, tmp_path: pathlib.Path) -> None:
    # The whole reason the budget exists: a sequential loop and a 4-env parallel loop given the
    # same env_step budget must do the same amount of environment work, even though they need
    # very different iteration counts to get there.
    budget = Budget(env_steps=25)   # deliberately not a multiple of either loop's iteration cost
    seq_config, seq_built = toy(budget, n_envs=1)
    par_config, par_built = toy(budget, n_envs=4)

    _v, seq_log = run_experiment(seq_config, tmp_path / "seq", built=seq_built, eval_every=0)
    _v, par_log = run_experiment(par_config, tmp_path / "par", built=par_built, eval_every=0)

    assert seq_log[-1]["env_steps"] == par_log[-1]["env_steps"] == 24
    assert len(seq_log) == 4 * len(par_log)
    assert seq_log[-1]["env_steps"] <= 25


def test_budget_spend_survives_a_resume(toy, tmp_path: pathlib.Path) -> None:
    # Without this, anyone could exceed a budget just by restarting the run, and every budgeted
    # comparison would be unenforceable.
    output_dir = tmp_path / "run"
    config, built = toy(Budget(iterations=2))
    run_experiment(config, output_dir, built=built, eval_every=0)

    resumed_config, resumed_built = toy(Budget(iterations=3))
    _variables, log = run_experiment(resumed_config, output_dir, built=resumed_built, eval_every=0)
    assert [entry["iteration"] for entry in log] == [3]

    state = json.loads((output_dir / STATE_NAME).read_text())
    assert state["budget_use"]["iterations"] == 3
    assert state["environment_hash"] == resumed_config.environment_hash()


def test_resuming_an_already_exhausted_budget_does_nothing(toy, tmp_path: pathlib.Path) -> None:
    output_dir = tmp_path / "run"
    config, built = toy(Budget(iterations=2))
    run_experiment(config, output_dir, built=built, eval_every=0)
    _variables, log = run_experiment(config, output_dir, built=built, eval_every=0)
    assert log == []


def test_resume_recovers_iterations_from_a_checkpoint_written_before_budgets_existed(
    toy, tmp_path: pathlib.Path
) -> None:
    # Old runs have a checkpoint but no state.json. Refusing to resume them would be worse than
    # recovering the one dimension the checkpoint does carry.
    output_dir = tmp_path / "run"
    config, built = toy(Budget(iterations=2))
    run_experiment(config, output_dir, built=built, eval_every=0)
    (output_dir / STATE_NAME).unlink()

    resumed_config, resumed_built = toy(Budget(iterations=3))
    _variables, log = run_experiment(resumed_config, output_dir, built=resumed_built, eval_every=0)
    assert [entry["iteration"] for entry in log] == [3]


# --------------------------------------------------------------------------- eval side effects


def test_resumed_run_continues_the_placement_image_iteration_count(toy, tmp_path: pathlib.Path) -> None:
    # Regression (ported from the old script-local loop): resuming must keep counting from where
    # the checkpoint left off, not restart a local counter at 1 - which would both mislabel
    # snapshot filenames and desync the --eval_every cadence.
    output_dir = tmp_path / "run"
    images_dir = tmp_path / "placements"

    config, built = toy(Budget(iterations=5))
    run_experiment(config, output_dir, built=built, eval_every=5, placement_images_dir=images_dir)
    assert sorted(p.name for p in images_dir.glob("*.png")) == ["5.png"]

    resumed_config, resumed_built = toy(Budget(iterations=15))
    run_experiment(resumed_config, output_dir, built=resumed_built, eval_every=5,
                   placement_images_dir=images_dir)
    assert sorted(p.name for p in images_dir.glob("*.png")) == ["10.png", "15.png", "5.png"]


def test_eval_every_gates_the_real_hpwl_computation(toy, tmp_path: pathlib.Path) -> None:
    config, built = toy(Budget(iterations=4))
    _variables, log = run_experiment(config, tmp_path / "run", built=built, eval_every=2)
    assert [entry["real_hpwl"] is not None for entry in log] == [False, True, False, True]


def test_best_checkpoint_is_written_when_real_hpwl_improves(toy, tmp_path: pathlib.Path) -> None:
    output_dir = tmp_path / "run"
    config, built = toy(Budget(iterations=2))
    run_experiment(config, output_dir, built=built, eval_every=1)
    # The first eval always improves on +inf, so this file must exist after any evaluated run.
    assert (output_dir / "best_checkpoint.bin").exists()


def test_patience_stops_the_run_before_the_budget_is_spent(toy, tmp_path: pathlib.Path) -> None:
    config, built = toy(Budget(iterations=50))
    _variables, log = run_experiment(
        config, tmp_path / "run", built=built, eval_every=1, patience=2
    )
    # A toy 3-macro benchmark plateaus almost immediately, so patience must bite well short of 50.
    assert 0 < len(log) < 50
