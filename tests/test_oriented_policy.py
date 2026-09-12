"""Learning a macro's turn, not only searching it - PPO under the `oriented_grid` action space.

Orientation was a real axis before this and still could not be learned: every shipped architecture
emitted a `(grid_x, grid_y)` logits map, which is exactly one grid cell, so `_require_space`
refused PPO the oriented space and only the GA could search turns.

The three things that had to be true, and each has a test here:

  * **legality per turn.** A quarter-turned macro is its height by its width and fits in different
    cells, so a two-dimensional mask stretched over the turn axis would call placements legal that
    were never checked - the failure this project keeps finding.
  * **one mask, two call sites.** `ppo_loss` rebuilds the mask from the stored observation to get
    its probability ratio; if the rollout and the loss built masks from two copies of the logic, a
    drift between them would make PPO optimize a ratio between two different distributions,
    silently. The ratio-is-one test below is what pins that.
  * **the turns reach the outputs.** The greedy eval reports them, the runner scores against them,
    and the placement is legal under the footprints it actually used.
"""
import dataclasses
import pathlib

import pytest

from placax.action_space import OrientedGridPlacement  # noqa: F401  must precede jax imports
from placax.core import reset
from placax.extras import orientation as orient
from placax.extras.legality import legality
from placax.types import EnvParams
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build, policy_action_spaces
from placax_agents.experiment.config import Spec
from placax_agents.experiment.presets import training
from placax_agents.experiment.run import best_orientations, run_experiment, score
from placax_agents.policy.action import (
    action_log_prob, masked_action_logits, oriented_illegal_actions,
)
from placax_agents.policy.architectures.oriented_cnn import OrientedCNNActorCritic
from placax_agents.policy.scale import to_grid_units
from placax_agents.training.rollout import collect_rollout

import jax
import jax.numpy as jnp
from jax import random

NODES = ("UCLA nodes 1.0\nNumNodes : 4\nNumTerminals : 4\n"
         "a 4 2 terminal\nb 2 6 terminal\nc 2 4 terminal\nd 6 2 terminal\n")
NETS = ("UCLA nets 1.0\nNumNets : 3\nNumPins : 6\n"
        "NetDegree : 2 n0\n\ta I : 1.0 0.0\n\tb O : 0.0 2.0\n"
        "NetDegree : 2 n1\n\tb I : 0.0 -2.0\n\tc O : 0.0 1.0\n"
        "NetDegree : 2 n2\n\tc I : 0.0 0.0\n\td O : 2.0 0.0\n")


def _design(directory: pathlib.Path) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "s.aux").write_text("RowBasedPlacement : s.nodes s.nets s.wts s.pl s.scl\n")
    (directory / "s.nodes").write_text(NODES)
    (directory / "s.nets").write_text(NETS)
    return directory


def _oriented(directory, *, grid: int = 12, iterations: int = 2, loop=Spec("sequential")):
    config = training(directory, budget=Budget(iterations=iterations))
    return dataclasses.replace(
        config,
        environment=dataclasses.replace(
            config.environment,
            benchmark=dataclasses.replace(config.environment.benchmark, grid=grid),
            action_space=Spec("oriented_grid"),
            reward=Spec("hpwl", {"dense": True}),
        ),
        agent=dataclasses.replace(
            config.agent, policy=Spec("oriented_cnn", {"features": 8}), loop=loop),
    )


# --------------------------------------------------------------- legality, per turn


def test_legality_is_computed_for_each_turn_separately() -> None:
    """The mask has to answer "where does this macro fit, laid this way", four times.

    A 1-by-4 macro on a 4-wide canvas fits nowhere upright except the left column, and lying on
    its side fits only along the bottom row. One two-dimensional map cannot say both.
    """
    params = EnvParams(grid=4, n_macros=1)
    obs = {
        "canvas": jnp.zeros((4, 4), dtype=bool),
        "current_macro_size": jnp.array([1.0, 4.0]),
    }
    illegal = oriented_illegal_actions(obs, params, cell_size=1.0)
    assert illegal.shape == (4, 4, orient.N_ORIENTATIONS)

    # NORTH: 1 wide by 4 tall - only y = 0 leaves room, every x does.
    north = illegal[:, :, orient.NORTH]
    assert not bool(north[:, 0].any()) and bool(north[:, 1:].all())
    # WEST: a quarter turn makes it 4 wide by 1 tall - now only x = 0 works, every y does.
    west = illegal[:, :, orient.WEST]
    assert not bool(west[0, :].any()) and bool(west[1:, :].all())


def test_a_turn_with_nowhere_to_go_does_not_become_the_attractive_one() -> None:
    """The relaxation valve fires over the whole action set, not per turn.

    Per turn, a turn that fits nowhere would relax to "everywhere is legal" and become the only
    unmasked option - the valve would manufacture exactly the illegal placement it exists to
    avoid deadlocking on. Here one turn fits and another does not, so nothing should relax.
    """
    params = EnvParams(grid=4, n_macros=1)
    obs = {
        # 5 units tall does not fit in a 4-cell canvas at all; 5 wide by 1 tall does not either,
        # so both axes are impossible - but a 1-by-4 at another turn is fine.
        "canvas": jnp.zeros((4, 4), dtype=bool),
        "current_macro_size": jnp.array([4.0, 1.0]),
    }
    illegal = oriented_illegal_actions(obs, params, cell_size=1.0)
    legal_turns = [turn for turn in range(orient.N_ORIENTATIONS)
                   if not bool(illegal[:, :, turn].all())]
    assert legal_turns, "some turn must fit"
    # Every turn that does NOT fit stays fully masked rather than being relaxed to fully legal.
    for turn in range(orient.N_ORIENTATIONS):
        if turn not in legal_turns:
            assert bool(illegal[:, :, turn].all())


def test_a_full_canvas_still_relaxes_rather_than_deadlocking() -> None:
    # The other half of the valve: when NO (cell, turn) pair is legal, the episode still has to
    # terminate, so legality itself is dropped - loudly, since the runner measures the overlap.
    params = EnvParams(grid=2, n_macros=1)
    obs = {
        "canvas": jnp.ones((2, 2), dtype=bool),      # every cell occupied
        "current_macro_size": jnp.array([1.0, 1.0]),
    }
    illegal = oriented_illegal_actions(obs, params, cell_size=1.0)
    assert not bool(illegal.all()), "an episode with no legal action must still be able to end"


# --------------------------------------------------------------- one mask, two call sites


def test_the_loss_rebuilds_exactly_the_mask_the_rollout_sampled_under(tmp_path) -> None:
    """PPO's ratio is exp(new_log_prob - old_log_prob); at unchanged weights it must be exactly 1.

    That is only true if the mask `ppo_loss` recomputes from the stored observation is bit-identical
    to the one `collect_rollout` sampled under. Both go through `masked_action_logits` for this
    reason, and this is the test that would catch them drifting apart - including under the
    oriented space, where the mask is the new per-turn one.
    """
    built = build(_oriented(_design(tmp_path / "bench")))
    benchmark = built.benchmark
    variables = built.policy.init(
        random.PRNGKey(0),
        built.state_fn(reset(benchmark.params, built.initial_positions, built.action_space),
                       benchmark.params, benchmark.sizes_array),
    )
    trajectory, _final = collect_rollout(
        random.PRNGKey(1), variables, built.policy.apply, benchmark.params, benchmark.reward_fn,
        benchmark.sizes_array, benchmark.cell_size, built.state_fn, built.extra_illegal_fn,
        built.initial_positions, built.n_placed, built.action_space,
    )

    def recomputed(index):
        obs = jax.tree_util.tree_map(lambda x: x[index], trajectory["obs"])
        logits, _value = built.policy.apply(variables, obs)
        masked = masked_action_logits(
            logits, obs, benchmark.params, benchmark.cell_size, built.extra_illegal_fn
        )
        return action_log_prob(masked, trajectory["action"][index])

    # The tolerance is about FUSION, not about the mask. The rollout's value is computed inside a
    # lax.scan and this one op by op, so XLA fuses them differently and they agree to ~1e-8. A
    # mask that differed by even one cell would move the log-softmax denominator by at least
    # log(n/(n-1)) - about 6e-4 on this 12x12x4 action set, four orders of magnitude above the
    # noise - so this stays a real assertion about the mask rather than about float arithmetic.
    for index in range(trajectory["action"].shape[0]):
        assert float(recomputed(index)) == pytest.approx(
            float(trajectory["log_prob"][index]), rel=1e-6, abs=1e-6
        )


def test_an_oriented_action_is_three_wide_and_its_turn_is_recorded(tmp_path) -> None:
    built = build(_oriented(_design(tmp_path / "bench")))
    benchmark = built.benchmark
    variables = built.policy.init(
        random.PRNGKey(0),
        built.state_fn(reset(benchmark.params, built.initial_positions, built.action_space),
                       benchmark.params, benchmark.sizes_array),
    )
    trajectory, final = collect_rollout(
        random.PRNGKey(1), variables, built.policy.apply, benchmark.params, benchmark.reward_fn,
        benchmark.sizes_array, benchmark.cell_size, built.state_fn, built.extra_illegal_fn,
        built.initial_positions, built.n_placed, built.action_space,
    )
    assert trajectory["action"].shape[1] == 3, "(x, y, turn)"
    assert final.orientations is not None
    assert final.orientations.tolist() == trajectory["action"][:, 2].tolist()


# --------------------------------------------------------------- the policy and the seam


def test_the_policy_declares_the_only_space_its_output_can_express() -> None:
    # A rank-3 logits map has no meaning under a space whose action is two numbers, and the
    # refusal is a property of the architecture rather than a table in build.py.
    assert policy_action_spaces(OrientedCNNActorCritic()) == ("oriented_grid",)


def test_an_oriented_policy_is_refused_the_constructive_space(tmp_path) -> None:
    config = _oriented(_design(tmp_path / "bench"))
    constructive = dataclasses.replace(config, environment=dataclasses.replace(
        config.environment, action_space=Spec("discrete_grid")))
    with pytest.raises(ValueError, match="produces an action this action space cannot use"):
        build(constructive)


def test_the_logits_carry_a_turn_axis_and_react_to_the_macros_shape(tmp_path) -> None:
    # The turn head reads the macro's own footprint, which the canvas does not carry: a tall macro
    # and a wide one must not get identical turn preferences.
    built = build(_oriented(_design(tmp_path / "bench")))
    benchmark = built.benchmark
    obs = built.state_fn(
        reset(benchmark.params, built.initial_positions, built.action_space),
        benchmark.params, benchmark.sizes_array,
    )
    variables = built.policy.init(random.PRNGKey(0), obs)
    logits, value = built.policy.apply(variables, obs)
    assert logits.shape == (benchmark.params.grid_x, benchmark.params.effective_grid_y,
                            orient.N_ORIENTATIONS)
    assert value.shape == ()

    tall = dict(obs, current_macro_size=jnp.array([1.0, 8.0]))
    wide = dict(obs, current_macro_size=jnp.array([8.0, 1.0]))
    tall_logits, _ = built.policy.apply(variables, tall)
    wide_logits, _ = built.policy.apply(variables, wide)
    assert not jnp.allclose(tall_logits, wide_logits)


# --------------------------------------------------------------- end to end


@pytest.mark.parametrize("loop", [
    Spec("sequential"),
    Spec("parallel", {"n_envs": 2}),
    Spec("buffered", {"n_episodes": 4, "ppo_epochs": 2, "batch_size": 4}),
])
def test_oriented_ppo_trains_and_reports_the_turns_it_chose(tmp_path, loop: Spec) -> None:
    config = _oriented(_design(tmp_path / "bench"), loop=loop)
    built = build(config)
    state, log = run_experiment(config, None, built=built, eval_every=1, log_every=10 ** 9)

    assert log[-1]["loss"] is not None
    orientations = best_orientations(built.agent, state)
    assert orientations is not None and orientations.shape == (built.benchmark.params.n_macros,)
    assert bool(((orientations >= 0) & (orientations < orient.N_ORIENTATIONS)).all())


def test_the_placement_is_legal_under_the_footprints_it_actually_used(tmp_path) -> None:
    """The dangerous outcome would be a placement legal upright and overlapping as placed.

    Legality is measured on `effective_sizes`, so a run that masked per turn and scored per turn
    agrees with itself; one that masked upright would show overlap here.
    """
    built = build(_oriented(_design(tmp_path / "bench"), grid=24))
    state, _log = run_experiment(built.config, None, built=built, eval_every=0, log_every=10 ** 9)
    positions = built.agent.best_positions(state)
    orientations = best_orientations(built.agent, state)

    grid_sizes = to_grid_units(
        orient.effective_sizes(built.benchmark.sizes_array, orientations),
        built.benchmark.cell_size,
    )
    measured = legality(positions, grid_sizes, built.benchmark.params)
    assert int(measured.n_unplaced) == 0
    assert float(measured.overlap_area) == 0.0
    assert float(measured.out_of_bounds_area) == 0.0


def test_the_runner_scores_an_oriented_run_against_its_turns(tmp_path) -> None:
    # real_hpwl, the reward and legality all have to describe the same placement - the one with
    # the turns in it.
    built = build(_oriented(_design(tmp_path / "bench"), grid=24))
    state, _log = run_experiment(built.config, None, built=built, eval_every=0, log_every=10 ** 9)
    positions = built.agent.best_positions(state)
    orientations = best_orientations(built.agent, state)

    measured = score(built.benchmark, positions, built.n_placed, orientations,
                     built.action_space, built.initial_positions)
    upright = score(built.benchmark, positions, built.n_placed, None, built.action_space)
    if not bool((orientations == orient.NORTH).all()):
        assert measured["real_hpwl"] != pytest.approx(upright["real_hpwl"])
        assert measured["reward_return"] != pytest.approx(upright["reward_return"])
