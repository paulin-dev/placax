"""What a run must pin down and write down: the design, the warm start, the budget, the metrics.

Each test here corresponds to something that used to be true of a run but was recorded nowhere,
or was recorded in a way that got the answer wrong. Together they are the "every experiment uses
exactly the same benchmark, initial placement, constraints, reward, cell placement and compute
budget" claim, made checkable.

Two items from that list are deliberately absent, and saying so here is the point: there is no
**legalization** component (legality is enforced by masking and MEASURED afterwards - see
extras/legality.py - but nothing repairs a placement) and no **routing** at all (only the RUDY
congestion proxy in the reward). Listing them as covered, which this docstring used to, made the
suite look like it checked two invariants the project does not implement.
"""
import dataclasses
import json
import pathlib

from placax.core import replay, reset, step  # noqa: F401  must precede jax imports
from placax.netlist.digest import netlist_digest
from placax.extras.rewards import hpwl, smoothed_wirelength
from placax_agents.experiment.budget import Budget, BudgetTracker, BudgetUse
from placax_agents.experiment.build import build, build_benchmark
from placax_agents.experiment.config import (
    AgentSpec, ExperimentConfig, PhysicalSpec, Spec, assert_comparable,
)
from placax_agents.experiment.presets import maskplace, training
from placax_agents.experiment.run import LOG_NAME, MANIFEST_NAME, run_experiment

import jax
import jax.numpy as jnp
import pytest
from jax import random

NODES = ("UCLA nodes 1.0\nNumNodes : 4\nNumTerminals : 4\n"
         "a 4 4 terminal\nb 2 2 terminal\nc 2 4 terminal\nd 4 2 terminal\n")
NETS = ("UCLA nets 1.0\nNumNets : 3\nNumPins : 6\n"
        "NetDegree : 2 n0\n\ta I : 0.0 0.0\n\tb O : 0.0 0.0\n"
        "NetDegree : 2 n1\n\tb I : 0.0 0.0\n\tc O : 0.0 0.0\n"
        "NetDegree : 2 n2\n\tc I : 0.0 0.0\n\td O : 0.0 0.0\n")


def _bookshelf(directory: pathlib.Path, nodes: str = NODES, nets: str = NETS) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "s.aux").write_text("RowBasedPlacement : s.nodes s.nets s.wts s.pl s.scl\n")
    (directory / "s.nodes").write_text(nodes)
    (directory / "s.nets").write_text(nets)
    return directory


def _small(config, grid: int = 12, **environment_overrides):
    return dataclasses.replace(config, environment=dataclasses.replace(
        config.environment,
        benchmark=dataclasses.replace(config.environment.benchmark, grid=grid),
        **environment_overrides,
    ))


# ------------------------------------------------------- the design's own identity


def test_the_same_netlist_at_two_paths_is_one_design(tmp_path: pathlib.Path) -> None:
    # Hashing the path made two machines mounting one benchmark look incomparable. What
    # identifies a design is its contents.
    here = _bookshelf(tmp_path / "here")
    there = _bookshelf(tmp_path / "elsewhere" / "here")
    a = build(_small(training(here, budget=Budget(iterations=1))))
    b = build(_small(training(there, budget=Budget(iterations=1))))
    assert a.config.benchmark_hash() == b.config.benchmark_hash()
    assert_comparable(a.config, b.config)


def test_editing_a_netlist_in_place_makes_it_a_different_design(tmp_path: pathlib.Path) -> None:
    # The other direction, and the one that mattered: a path-based hash called these identical.
    directory = _bookshelf(tmp_path / "bench")
    before = build(_small(training(directory, budget=Budget(iterations=1)))).config

    edited = NODES.replace("a 4 4 terminal", "a 6 4 terminal")   # one macro grew
    (directory / "s.nodes").write_text(edited)
    after = build(_small(training(directory, budget=Budget(iterations=1)))).config

    assert before.benchmark_hash() != after.benchmark_hash()
    with pytest.raises(ValueError, match="netlist_digest"):
        assert_comparable(before, after)


def test_the_digest_ignores_incidental_ordering(tmp_path: pathlib.Path) -> None:
    # Reformatting a file or emitting its nets in another order does not change the design, and
    # must not invalidate a comparison - only real geometry and connectivity may.
    sizes = {"a": (4.0, 4.0), "b": (2.0, 2.0)}
    nets = [[("a", 0.0, 0.0), ("b", 0.0, 0.0)]]
    reordered_sizes = {"b": (2.0, 2.0), "a": (4.0, 4.0)}
    reordered_nets = [[("b", 0.0, 0.0), ("a", 0.0, 0.0)]]
    assert netlist_digest(sizes, nets) == netlist_digest(reordered_sizes, reordered_nets)
    assert netlist_digest(sizes, nets) != netlist_digest({"a": (5.0, 4.0), "b": (2.0, 2.0)}, nets)


def test_an_unresolved_config_does_not_false_alarm_against_a_resolved_one(tmp_path) -> None:
    # A config straight from a preset has never looked at the design, so it makes no claim about
    # its contents. Treating that as "different design" would be a false alarm in the one
    # mechanism that has to be trustworthy.
    directory = _bookshelf(tmp_path / "bench")
    unresolved = _small(training(directory, budget=Budget(iterations=1)))
    resolved = build(unresolved).config
    assert resolved.environment.benchmark.netlist_digest is not None
    assert unresolved.environment.benchmark.netlist_digest is None
    assert_comparable(unresolved, resolved)


# ---------------------------------------------------------------- graded comparison levels


def test_a_state_representation_study_is_comparable_at_the_task_level(tmp_path) -> None:
    """Two runs differing ONLY in observation are the experiment docs §12 asks for.

    `environment_hash` rejects them by construction - the observation is part of the environment -
    which would leave that whole study with no mechanism at all. `task_hash` is the invariant it
    actually needs: same design, reward, constraints, warm start and budget.
    """
    directory = _bookshelf(tmp_path / "bench")
    canvas = _small(maskplace(directory, budget=Budget(iterations=1)),
                    state=Spec("canvas", {"lookahead": 1}))
    wiremask = _small(maskplace(directory, budget=Budget(iterations=1)))

    assert canvas.task_hash() == wiremask.task_hash()
    assert canvas.environment_hash() != wiremask.environment_hash()
    assert_comparable(canvas, wiremask, level="task")
    with pytest.raises(ValueError, match="do not share a environment"):
        assert_comparable(canvas, wiremask, level="environment")


def test_a_reward_change_breaks_comparability_at_every_level_above_benchmark(tmp_path) -> None:
    directory = _bookshelf(tmp_path / "bench")
    base = _small(training(directory, budget=Budget(iterations=1)))
    other = _small(training(directory, budget=Budget(iterations=1)),
                   reward=Spec("smoothed", {"dense": False}))
    assert base.benchmark_hash() == other.benchmark_hash()
    assert base.task_hash() != other.task_hash()
    with pytest.raises(ValueError, match="reward"):
        assert_comparable(base, other, level="task")


def test_an_unknown_comparison_level_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown comparison level"):
        assert_comparable(level="vibes")


# ------------------------------------------------------------------- initial placement


def test_the_default_initial_placement_is_recorded_rather_than_implied(tmp_path) -> None:
    # Every run used to start from an empty canvas with nothing saying so. `empty` is that same
    # behavior, named - so a later warm-start experiment is distinguishable from today's runs.
    directory = _bookshelf(tmp_path / "bench")
    config = _small(training(directory, budget=Budget(iterations=1)))
    assert config.environment.initial_placement == Spec("empty")
    built = build(config)
    assert built.initial_positions is None
    assert built.n_placed == 0
    assert built.steps_per_episode == built.n_macros


def test_a_warm_start_shortens_the_episode_and_is_hashed(tmp_path: pathlib.Path) -> None:
    directory = _bookshelf(tmp_path / "bench")
    cold = _small(training(directory, budget=Budget(iterations=1)))
    warm = _small(training(directory, budget=Budget(iterations=1)),
                  initial_placement=Spec("greedy_wiremask_prefix", {"n_macros": 2}))

    # It changes results, so it must change the hash - otherwise two runs that started from
    # different placements would be reported as comparable.
    assert cold.environment_hash() != warm.environment_hash()

    built = build(warm)
    assert built.n_placed == 2
    # The agent places only what is left, rather than scanning past the end of the array.
    assert built.steps_per_episode == built.n_macros - 2
    assert int((built.initial_positions[:, 0] >= 0).sum()) == 2


def test_a_warm_started_run_places_every_macro_exactly_once(tmp_path: pathlib.Path) -> None:
    directory = _bookshelf(tmp_path / "bench")
    warm = _small(training(directory, budget=Budget(iterations=1)),
                  initial_placement=Spec("greedy_wiremask_prefix", {"n_macros": 2}))
    built = build(warm)
    state, _log = run_experiment(warm, None, built=built, eval_every=1)
    positions = built.agent.best_positions(state)
    assert int((positions[:, 0] >= 0).sum()) == built.n_macros
    # The pre-placed prefix is kept exactly as the warm start left it.
    assert jnp.array_equal(positions[:2], built.initial_positions[:2])


@pytest.mark.parametrize("loop", [
    Spec("sequential"),
    Spec("parallel", {"n_envs": 2}),
    Spec("buffered", {"n_episodes": 2, "ppo_epochs": 1, "batch_size": 2}),
])
def test_every_loop_shape_trains_warm_started(tmp_path: pathlib.Path, loop: Spec) -> None:
    """The riskiest plumbing in the warm start, so it is checked for all three loops.

    `n_placed` fixes a static shape (the scan length), which means it has to reach each jitted
    step function as a static argument and each vmapped rollout as a closure, not as a traced
    leaf. Getting that wrong fails at trace time in one loop shape and not the others, so testing
    only the sequential loop would leave two thirds of it uncovered.
    """
    directory = _bookshelf(tmp_path / "bench")
    config = _small(
        training(directory, budget=Budget(iterations=1)),
        initial_placement=Spec("greedy_wiremask_prefix", {"n_macros": 1}),
    )
    config = dataclasses.replace(config, agent=dataclasses.replace(config.agent, loop=loop))
    built = build(config)
    assert built.n_placed == 1
    assert built.steps_per_episode == built.n_macros - 1

    _state, log = run_experiment(config, None, built=built, eval_every=1, log_every=10 ** 9)
    # Budget is charged for what the agent actually placed, not for the macros handed to it.
    episodes = log[-1]["episodes"] - 1   # minus the evaluation rollout
    assert log[-1]["env_steps"] == (episodes + 1) * built.steps_per_episode
    assert log[-1]["real_hpwl"] is not None


def test_a_warm_start_that_places_everything_is_refused(tmp_path: pathlib.Path) -> None:
    directory = _bookshelf(tmp_path / "bench")
    config = _small(training(directory, budget=Budget(iterations=1)),
                    initial_placement=Spec("greedy_wiremask_prefix", {"n_macros": 4}))
    with pytest.raises(ValueError, match="nothing to do"):
        build(config)


# ------------------------------------------------------------------------ the budget


def test_evaluation_rollouts_are_charged_to_the_budget(tmp_path: pathlib.Path) -> None:
    """An eval places every remaining macro - exactly as much environment work as an episode.

    Leaving it free meant two runs given one --env_steps budget did different amounts of work
    when their --eval_every differed, which it does between the two shipped training scripts.
    """
    directory = _bookshelf(tmp_path / "bench")
    config = _small(training(directory, budget=Budget(iterations=3)))
    built = build(config)

    _state, evaluated = run_experiment(config, None, built=built, eval_every=1, log_every=10 ** 9)
    _state, unevaluated = run_experiment(config, None, built=built, eval_every=0, log_every=10 ** 9)

    assert evaluated[-1]["eval_env_steps"] == 3 * built.steps_per_episode
    assert unevaluated[-1]["eval_env_steps"] == 0
    assert evaluated[-1]["env_steps"] > unevaluated[-1]["env_steps"]


def test_gradient_steps_are_reported_so_compute_is_not_just_assumed(tmp_path) -> None:
    # env_steps prices environment interaction and deliberately not arithmetic. A run that wants
    # to claim "matched compute" needs this number, and a non-learning agent's zero is the point.
    directory = _bookshelf(tmp_path / "bench")
    config = _small(training(directory, budget=Budget(iterations=2)))
    _state, learner = run_experiment(config, None, built=build(config), log_every=10 ** 9)
    assert learner[-1]["gradient_steps"] == 2   # sequential loop: one update per iteration

    baseline = dataclasses.replace(
        config, agent=AgentSpec(algorithm=Spec("random_search", {"population": 2}))
    )
    _state, floor = run_experiment(baseline, None, built=build(baseline), log_every=10 ** 9)
    assert floor[-1]["gradient_steps"] == 0


def test_budget_use_round_trips_the_new_fields() -> None:
    use = BudgetUse(iterations=1, episodes=2, env_steps=3, eval_env_steps=1, gradient_steps=7)
    assert BudgetUse.from_dict(use.to_dict()) == use


def test_a_tracker_charges_an_evaluation_like_an_episode() -> None:
    tracker = BudgetTracker(Budget(env_steps=100))
    tracker.record_iteration(episodes=2, steps_per_episode=10, gradient_steps=5)
    use = tracker.record_evaluation(10)
    assert (use.env_steps, use.eval_env_steps, use.gradient_steps) == (30, 10, 5)


# ------------------------------------------------------------------ what a run writes down


def test_the_manifest_defines_every_metric_it_logs(tmp_path: pathlib.Path) -> None:
    """A number is only comparable if its definition travels with it.

    `real_hpwl` in particular is macro-to-macro only - the parser drops nets with fewer than two
    macros - so a reader comparing it against a paper's full-netlist HPWL would be wrong without
    ever being told.
    """
    directory = _bookshelf(tmp_path / "bench")
    config = _small(training(directory, budget=Budget(iterations=1)))
    built = build(config)
    output_dir = tmp_path / "run"
    _state, log = run_experiment(config, output_dir, built=built, eval_every=1)

    manifest = json.loads((output_dir / MANIFEST_NAME).read_text())
    assert "MACRO-TO-MACRO" in manifest["metrics"]["real_hpwl"]
    for key in ("real_hpwl", "reward_return", "overlap_ratio", "gradient_steps"):
        assert key in manifest["metrics"], f"{key} is logged but never defined"
        assert key in log[-1], f"{key} is defined in the manifest but never logged"


def test_an_evaluated_log_line_carries_legality_beside_the_score(tmp_path) -> None:
    directory = _bookshelf(tmp_path / "bench")
    config = _small(training(directory, budget=Budget(iterations=1)))
    output_dir = tmp_path / "run"
    _state, _log = run_experiment(config, output_dir, built=build(config), eval_every=1)
    entry = json.loads((output_dir / LOG_NAME).read_text().splitlines()[-1])
    for key in ("real_hpwl", "reward_return", "overlap_ratio", "out_of_bounds_ratio",
                "n_unplaced", "is_legal"):
        assert entry[key] is not None


def test_the_manifest_records_the_resolved_design_not_just_its_path(tmp_path) -> None:
    directory = _bookshelf(tmp_path / "bench")
    config = _small(training(directory, budget=Budget(iterations=1)))
    built = build(config)
    output_dir = tmp_path / "run"
    run_experiment(config, output_dir, built=built, eval_every=0)
    manifest = json.loads((output_dir / MANIFEST_NAME).read_text())
    assert manifest["config"]["environment"]["benchmark"]["netlist_digest"] is not None
    assert manifest["config"]["full_hash"] == built.config.full_hash()


# --------------------------------------------------------------------- the physical stack


def test_the_physical_tools_are_part_of_the_environment_hash(tmp_path) -> None:
    # A PPA number is only attributable if the tools that produced it are recorded with it, so
    # swapping the validator has to be a different environment rather than an invisible change.
    directory = _bookshelf(tmp_path / "bench")
    proxy_only = _small(training(directory, budget=Budget(iterations=1)))
    physical = _small(training(directory, budget=Budget(iterations=1)),
                      physical=PhysicalSpec(Spec("dreamplace"), Spec("openroad")))
    assert proxy_only.environment_hash() != physical.environment_hash()
    assert proxy_only.environment.physical.validator is None   # honest about running no flow
    assert ExperimentConfig.from_json(physical.to_json()) == physical


def test_a_proxy_only_run_builds_no_physical_tools(tmp_path: pathlib.Path) -> None:
    # Neither DREAMPlace nor OpenROAD may be needed to run a wirelength-proxy experiment.
    directory = _bookshelf(tmp_path / "bench")
    built = build(_small(training(directory, budget=Budget(iterations=1))))
    assert built.cell_placer is None and built.validator is None


# ------------------------------------------------------------------------- the reward axis


def test_the_smoothed_surrogate_gives_gradient_where_raw_hpwl_gives_none(tmp_path) -> None:
    """The measurement docs/Action_Space_Decision.md says should decide the action-space question.

    HPWL is a sum of per-net (max - min), so only pins ON a net's bounding box get gradient; a
    pin strictly inside contributes nothing. Log-sum-exp gives every pin a share, which is why
    DREAMPlace optimizes it instead. Three collinear macros make that concrete: the middle one is
    interior to the net's bounding box.

    The interior macro is deliberately OFF-CENTRE. At the exact midpoint the surrogate's gradient
    vanishes too, and for a real reason rather than a bug: the span is soft_max(x) + soft_max(-x),
    whose two softmax weights for a perfectly symmetric pin are equal and cancel. So a symmetric
    test case would pass this assertion for neither function and prove nothing about either.
    """
    positions = jnp.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    pin_idx = jnp.array([[0, 1, 2]])
    pin_offset = jnp.zeros((1, 3, 2))
    valid = jnp.ones((1, 3), dtype=bool)

    raw_grad = jax.grad(lambda p: hpwl(p, pin_idx, pin_offset, valid))(positions)
    smooth_grad = jax.grad(
        lambda p: smoothed_wirelength(p, pin_idx, pin_offset, valid, gamma=1.0)
    )(positions)

    assert float(raw_grad[1, 0]) == 0.0            # the interior macro moves nothing
    assert abs(float(smooth_grad[1, 0])) > 0.0     # under the surrogate, it does


def test_the_smoothed_surrogate_approaches_hpwl_as_gamma_shrinks() -> None:
    # It has to be a surrogate FOR hpwl, not a different objective wearing its name.
    positions = jnp.array([[0.0, 0.0], [5.0, 3.0], [10.0, 1.0]])
    pin_idx = jnp.array([[0, 1, 2]])
    pin_offset = jnp.zeros((1, 3, 2))
    valid = jnp.ones((1, 3), dtype=bool)

    exact = float(hpwl(positions, pin_idx, pin_offset, valid))
    coarse = float(smoothed_wirelength(positions, pin_idx, pin_offset, valid, gamma=2.0))
    fine = float(smoothed_wirelength(positions, pin_idx, pin_offset, valid, gamma=0.05))
    assert abs(fine - exact) < abs(coarse - exact)
    assert fine == pytest.approx(exact, abs=1e-2)


def test_the_smoothed_reward_is_selectable_from_a_config(tmp_path: pathlib.Path) -> None:
    directory = _bookshelf(tmp_path / "bench")
    config = _small(training(directory, budget=Budget(iterations=1)),
                    reward=Spec("smoothed", {"dense": True, "gamma": 1.0}))
    built = build(config)
    positions = built.agent.best_positions(built.agent.init(random.PRNGKey(0)))
    assert float(replay(positions, built.benchmark.reward_fn, built.benchmark.params)) != 0.0


# --------------------------------------------------------------------------- the kernel


def test_replay_drives_the_same_kernel_from_a_pre_committed_sequence(tmp_path) -> None:
    """The population-method claim the design has always made, as running code.

    `step()` cannot tell whether an action came from a policy or a genome, so replaying a
    finished placement has to give exactly the reward the episode that produced it did.
    """
    directory = _bookshelf(tmp_path / "bench")
    built = build(_small(training(directory, budget=Budget(iterations=1)),
                         reward=Spec("hpwl", {"dense": True})))
    benchmark = built.benchmark
    positions = built.agent.best_positions(built.agent.init(random.PRNGKey(0)))

    # Step through by hand, the way a training rollout does.
    state = reset(benchmark.params)
    total = 0.0
    for i in range(benchmark.params.n_macros):
        state, reward, _done = step(state, positions[i], benchmark.reward_fn, benchmark.params)
        total += float(reward)

    assert float(replay(positions, benchmark.reward_fn, benchmark.params)) == pytest.approx(
        total, rel=1e-5
    )


def test_replay_vmaps_over_a_population(tmp_path: pathlib.Path) -> None:
    # "vmap the same episode-replay function across a population of pre-committed action
    # sequences" - docs §1. No separate batch-evaluation entry point, as promised.
    directory = _bookshelf(tmp_path / "bench")
    built = build(_small(training(directory, budget=Budget(iterations=1))))
    benchmark = built.benchmark
    population = jnp.stack([
        jnp.zeros((benchmark.params.n_macros, 2), dtype=jnp.int32),
        jnp.ones((benchmark.params.n_macros, 2), dtype=jnp.int32),
    ])
    scores = jax.vmap(lambda p: replay(p, benchmark.reward_fn, benchmark.params))(population)
    assert scores.shape == (2,)


# ------------------------------------------------- the comparison mechanism's own blind spots


@pytest.mark.parametrize("level", ["benchmark", "task", "environment", "full"])
def test_an_unresolved_config_never_false_alarms_at_any_level(tmp_path: pathlib.Path, level: str) -> None:
    """The digest fallback has to hold at every level, not just the default one.

    `_digest_of` walked the identity by guessing at key names, and at the "full" level - where the
    benchmark sits two wrappers down, under "environment" - it found nothing. So the fallback that
    exists to stop a config-without-a-digest reading as a DIFFERENT DESIGN never fired there, and
    the one mechanism that has to be trustworthy raised on two configs naming the same netlist.
    Parameterized over every level so the next wrapper added cannot reintroduce it quietly.
    """
    directory = _bookshelf(tmp_path / "bench")
    unresolved = _small(training(directory, budget=Budget(iterations=1)))
    resolved = build(unresolved).config
    assert resolved.environment.benchmark.netlist_digest is not None
    assert unresolved.environment.benchmark.netlist_digest is None
    assert_comparable(unresolved, resolved, level=level)


@pytest.mark.parametrize("level", ["benchmark", "task", "environment", "full"])
def test_two_designs_are_still_caught_at_every_level(tmp_path: pathlib.Path, level: str) -> None:
    # The other direction of the same fix: tolerating a MISSING digest must not tolerate a
    # different one. Both configs are resolved here, so the digest is live and has to bite.
    edited = NODES.replace("a 4 4 terminal", "a 8 8 terminal")
    one = build(_small(training(_bookshelf(tmp_path / "one"), budget=Budget(iterations=1)))).config
    other = build(_small(training(
        _bookshelf(tmp_path / "other", nodes=edited), budget=Budget(iterations=1)
    ))).config
    with pytest.raises(ValueError, match="netlist_digest"):
        assert_comparable(one, other, level=level)


def test_a_default_written_out_hashes_like_a_default_left_unsaid(tmp_path: pathlib.Path) -> None:
    """Two spellings of one component are one component, and must hash alike.

    Before this, `Spec("hpwl")` and `Spec("hpwl", {...the defaults...})` produced different
    hashes, so `assert_comparable` rejected two identical experiments and a config read back from
    JSON could never match one built from a preset - a hash over the spelling of a config rather
    than over the config.
    """
    directory = _bookshelf(tmp_path / "bench")
    spelled = _small(training(directory, budget=Budget(iterations=1)))
    bare = _small(training(directory, budget=Budget(iterations=1)), reward=Spec("hpwl"))

    assert spelled.environment.reward.kwargs  # the preset really does spell its defaults out
    assert not bare.environment.reward.kwargs
    assert spelled.environment_hash() == bare.environment_hash()
    assert_comparable(spelled, bare)


def test_a_non_default_value_still_changes_the_hash(tmp_path: pathlib.Path) -> None:
    # Completing defaults must not flatten real differences - the failure mode that would make
    # the whole mechanism useless in the opposite direction.
    directory = _bookshelf(tmp_path / "bench")
    base = _small(training(directory, budget=Budget(iterations=1)))
    dense = _small(training(directory, budget=Budget(iterations=1)),
                   reward=Spec("hpwl", {"dense": True}))
    assert base.environment_hash() != dense.environment_hash()


def test_completing_defaults_leaves_the_written_config_as_the_author_wrote_it(tmp_path) -> None:
    # Hashing completes a Spec; serialization does not. A written config is a record of what
    # someone wrote, and has to round-trip unchanged.
    config = _small(training(_bookshelf(tmp_path / "bench"), budget=Budget(iterations=1)),
                    reward=Spec("hpwl"))
    assert json.loads(config.to_json())["environment"]["reward"] == {"name": "hpwl", "kwargs": {}}
    assert ExperimentConfig.from_json(config.to_json()) == config


def test_the_physical_stack_is_not_completed_with_its_builders_defaults() -> None:
    """Machine paths must never reach the environment hash, which completing them would do.

    `_cell_placer_dreamplace` and `OpenROADValidator` both carry defaults for things that belong
    to a host, not an experiment - `dreamplace_root`, `openroad_binary`. Completing those would
    fold an install path into the hash and stop two labs running one experiment from comparing.
    """
    from placax_agents.experiment.config import EnvironmentSpec

    physical = PhysicalSpec(cell_placer=Spec("dreamplace"), validator=Spec("openroad"))
    identity = EnvironmentSpec(
        benchmark=training("b").environment.benchmark, reward=Spec("hpwl"), state=Spec("canvas"),
        budget=Budget(iterations=1), physical=physical,
    ).task_identity()
    written = json.dumps(identity["physical"])
    assert "dreamplace_root" not in written and "openroad_binary" not in written
    assert identity["physical"] == physical.to_dict()


def test_a_derived_kwarg_is_recorded_at_the_value_the_run_uses(tmp_path: pathlib.Path) -> None:
    """The split optimizer's value_coef is derived from PPO's, so the record has to show that.

    It used to be injected inside the agent builder, after the config had been hashed and written
    - so a run with a non-default value_coef recorded the optimizer's own 0.5 while actually
    running PPO's. The manifest described a run that never happened.
    """
    config = _small(maskplace(_bookshelf(tmp_path / "bench"), budget=Budget(iterations=1)))
    config = dataclasses.replace(config, agent=dataclasses.replace(
        config.agent,
        algorithm=Spec("ppo", {**config.agent.algorithm.kwargs, "value_coef": 0.25}),
    ))
    assert "value_coef" not in config.agent.optimizer.kwargs

    resolved = build(config).config
    assert resolved.agent.optimizer.kwargs["value_coef"] == 0.25
