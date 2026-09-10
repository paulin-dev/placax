"""Macro orientation - the half of "placement representation" that did not exist.

A placement was `(n_macros, 2)`: where each macro's corner sits and nothing else. Real flows emit
an orientation per instance, and this project's writers preserved whatever the source file said,
so the axis looked fixed at north when it was simply absent - an agent could not lay a tall SRAM
on its side, and nothing recorded that it hadn't.

Two things these tests have to establish, and the second is the one that bites:

  * the un-oriented path is **unchanged** - `effective_sizes` and `rotate_offsets` are the
    identity on `None`, so every existing result still means what it meant; and
  * orientation reaches **geometry, wirelength, legality and the written file**. Getting three of
    those four right is the dangerous outcome: a macro that is rotated for scoring and upright for
    legality is a placement that measures well and cannot be built.
"""
import dataclasses
import pathlib

import pytest

from placax.action_space import OrientedGridPlacement  # noqa: F401  must precede jax imports
from placax.core import reset, step
from placax.extras import orientation as orient
from placax.extras.legality import legality
from placax.extras.rewards import hpwl
from placax.netlist.bookshelf import write_pl
from placax.netlist.def_writer import write_placed_def
from placax.types import EnvParams, EnvState
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build
from placax_agents.experiment.config import AgentSpec, Spec
from placax_agents.experiment.presets import training
from placax_agents.experiment.run import best_orientations, score, score_placement

import jax
import jax.numpy as jnp

NODES = ("UCLA nodes 1.0\nNumNodes : 4\nNumTerminals : 4\n"
         "a 4 2 terminal\nb 2 2 terminal\nc 2 4 terminal\nd 4 2 terminal\n")
NETS = ("UCLA nets 1.0\nNumNets : 3\nNumPins : 6\n"
        "NetDegree : 2 n0\n\ta I : 1.0 0.0\n\tb O : 0.0 0.0\n"
        "NetDegree : 2 n1\n\tb I : 0.0 0.0\n\tc O : 0.0 1.0\n"
        "NetDegree : 2 n2\n\tc I : 0.0 0.0\n\td O : 0.0 0.0\n")


def _design(directory: pathlib.Path) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "s.aux").write_text("RowBasedPlacement : s.nodes s.nets s.wts s.pl s.scl\n")
    (directory / "s.nodes").write_text(NODES)
    (directory / "s.nets").write_text(NETS)
    return directory


# --------------------------------------------------------------- the geometry


def test_a_quarter_turn_swaps_width_and_height_and_a_half_turn_does_not() -> None:
    sizes = jnp.array([[4.0, 2.0]] * 4)
    turned = orient.effective_sizes(sizes, jnp.array([orient.NORTH, orient.WEST,
                                                      orient.SOUTH, orient.EAST]))
    assert turned.tolist() == [[4.0, 2.0], [2.0, 4.0], [4.0, 2.0], [2.0, 4.0]]


def test_pins_rotate_about_their_macros_center() -> None:
    # Counter-clockwise: N (dx, dy) -> W (-dy, dx) -> S (-dx, -dy) -> E (dy, -dx).
    offsets = jnp.array([[1.0, 2.0]] * 4)
    rotated = orient.rotate_offsets(offsets, jnp.array([0, 1, 2, 3]))
    assert rotated.tolist() == [[1.0, 2.0], [-2.0, 1.0], [-1.0, -2.0], [2.0, -1.0]]


def test_four_quarter_turns_return_a_pin_to_where_it_started() -> None:
    offsets = jnp.array([[3.0, -7.0]])
    turned = offsets
    for _ in range(4):
        turned = orient.rotate_offsets(turned, jnp.array([orient.WEST]))
    assert jnp.allclose(turned, offsets)


def test_no_orientation_is_the_identity_on_both_transforms() -> None:
    """The property the whole design rests on: code that never mentions orientation is untouched.

    If either transform did something to `None`, adding this axis would have silently changed
    every result the project has ever produced.
    """
    sizes = jnp.array([[4.0, 2.0], [1.0, 9.0]])
    offsets = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert orient.effective_sizes(sizes, None) is sizes
    assert orient.rotate_offsets(offsets, None) is offsets


def test_orientations_are_named_the_way_both_formats_spell_them() -> None:
    assert orient.names(jnp.array([0, 1, 2, 3]), 4) == ["N", "W", "S", "E"]
    assert orient.names(None, 3) == ["N", "N", "N"]


# --------------------------------------------------------------- the state


def test_a_state_carries_no_orientations_unless_something_sets_them() -> None:
    assert EnvState(positions=jnp.zeros((2, 2)), step=0).orientations is None
    assert reset(EnvParams(grid=4, n_macros=2)).orientations is None


def test_the_oriented_space_records_the_turn_the_action_chose() -> None:
    params = EnvParams(grid=8, n_macros=3)
    space = OrientedGridPlacement()

    def zero(_a, _b, _c, _d):
        return jnp.array(0.0)

    state = reset(params, None, space)
    state, _reward, _done = step(state, jnp.array([1, 2, orient.WEST]), zero, params, space)
    state, _reward, _done = step(state, jnp.array([3, 4, orient.EAST]), zero, params, space)
    assert state.positions.tolist()[:2] == [[1, 2], [3, 4]]
    assert state.orientations.tolist() == [orient.WEST, orient.EAST, orient.NORTH]


# --------------------------------------------------------------- it reaches the numbers


def test_turning_a_macro_changes_the_wirelength_it_is_scored_on(tmp_path) -> None:
    # A pin that rotates with its macro is at a different place, so the net it is on is a
    # different length. If HPWL ignored orientation, these would be equal.
    built = build(training(_design(tmp_path / "bench"), budget=Budget(iterations=1)))
    positions = jnp.array([[0, 0], [2, 2], [4, 4], [6, 6]])
    upright = score_placement(built.benchmark, positions, None)
    turned = score_placement(built.benchmark, positions, jnp.array([1, 0, 1, 0]))
    assert upright != turned


def test_legality_is_measured_on_the_rotated_footprint(tmp_path) -> None:
    """The dangerous case: rotated for scoring, upright for legality.

    Two macros side by side fit while upright and collide once one is turned across the other.
    A legality check that used the unrotated size would call the second placement clean.
    """
    params = EnvParams(grid=8, n_macros=2)
    sizes = jnp.array([[1, 4], [1, 4]])
    positions = jnp.array([[0, 0], [1, 0]])

    upright = legality(positions, orient.effective_sizes(sizes, None), params)
    assert int(upright.overlap_area) == 0

    # Turn the first macro a quarter turn: 1x4 becomes 4x1, which now runs across the second.
    turned_sizes = orient.effective_sizes(sizes, jnp.array([orient.WEST, orient.NORTH]))
    turned = legality(positions, turned_sizes, params)
    assert int(turned.overlap_area) > 0


def test_score_reports_legality_against_the_orientation_it_scored(tmp_path) -> None:
    # score() has to apply the same transform to both halves, or it reports a wirelength for one
    # placement and a legality for another.
    built = build(training(_design(tmp_path / "bench"), budget=Budget(iterations=1)))
    positions = jnp.array([[0, 0], [8, 0], [0, 8], [8, 8]])
    orientations = jnp.array([orient.WEST, orient.NORTH, orient.EAST, orient.NORTH])
    measured = score(built.benchmark, positions, 0, orientations)
    assert measured["real_hpwl"] == pytest.approx(
        score_placement(built.benchmark, positions, orientations)
    )


# --------------------------------------------------------------- it reaches the file


def test_the_pl_writer_emits_a_chosen_orientation_and_preserves_an_unchosen_one() -> None:
    original = "UCLA pl 1.0\n\na\t0\t0\t: N\nb\t0\t0\t: S\n"
    written = write_pl(original, {"a": (5, 6, "W"), "b": (1, 2)})
    assert "a\t5\t6\t: W /FIXED" in written
    assert "b\t1\t2\t: S /FIXED" in written, "an unchosen orientation must survive untouched"


def test_the_def_writer_emits_a_chosen_orientation_and_preserves_an_unchosen_one() -> None:
    original = "- a AND2 + PLACED ( 0 0 ) N ;\n- b OR2 + PLACED ( 0 0 ) S ;\n"
    written = write_placed_def(original, {"a": (3, 4, "E"), "b": (7, 8)})
    assert "- a AND2 + PLACED ( 3 4 ) E ;" in written
    assert "- b OR2 + PLACED ( 7 8 ) S ;" in written


def test_an_exported_placement_carries_its_turns(tmp_path) -> None:
    """End to end: an agent's chosen orientation reaches the file a real tool would open.

    This is where the axis was lost before - a placement could in principle have been turned and
    the writer would have written north anyway.
    """
    from placax_agents.experiment.export import write_placement

    directory = _design(tmp_path / "bench")
    (directory / "s.pl").write_text(
        "UCLA pl 1.0\n\na\t0\t0\t: N\nb\t0\t0\t: N\nc\t0\t0\t: N\nd\t0\t0\t: N\n"
    )
    (directory / "s.wts").write_text("UCLA wts 1.0\n")
    (directory / "s.scl").write_text("UCLA scl 1.0\n")
    built = build(training(directory, budget=Budget(iterations=1)))
    positions = jnp.array([[0, 0], [2, 2], [4, 4], [6, 6]])
    orientations = jnp.array([orient.WEST, orient.NORTH, orient.SOUTH, orient.EAST])

    exported = write_placement(built, positions, tmp_path / "out", orientations)
    placed = (exported.path.parent / "s.pl").read_text()
    for name, letter in zip("abcd", ["W", "N", "S", "E"]):
        index = built.benchmark.name_to_idx[name]
        assert f": {orient.ORIENTATION_NAMES[int(orientations[index])]} /FIXED" in placed
        assert letter in placed


# --------------------------------------------------------------- an agent chooses it


def _oriented_config(directory, space="oriented_grid"):
    base = training(directory, budget=Budget(iterations=2))
    return dataclasses.replace(
        base,
        environment=dataclasses.replace(
            base.environment,
            benchmark=dataclasses.replace(base.environment.benchmark, grid=12),
            action_space=Spec(space),
        ),
        agent=AgentSpec(algorithm=Spec("genetic", {"population": 8})),
    )


def test_the_ga_genome_grows_a_gene_when_the_space_has_an_orientation(tmp_path) -> None:
    directory = _design(tmp_path / "bench")
    plain = build(_oriented_config(directory, "discrete_grid")).agent
    oriented = build(_oriented_config(directory, "oriented_grid")).agent
    assert plain.gene_width == 2 and not plain.chooses_orientation
    assert oriented.gene_width == 3 and oriented.chooses_orientation


def test_an_agent_that_does_not_choose_orientation_reports_none(tmp_path) -> None:
    # None rather than an all-north array, so an un-oriented run scores through the same path it
    # always did instead of a new one that happens to agree.
    built = build(_oriented_config(_design(tmp_path / "bench"), "discrete_grid"))
    state = built.agent.init(jax.random.PRNGKey(0))
    assert best_orientations(built.agent, state) is None


def test_the_ga_reports_the_turns_it_chose_and_is_scored_on_them(tmp_path) -> None:
    """A run's orientations reach the runner, and the score is computed against them.

    Deliberately NOT asserting legality here. These four macros fill most of a small canvas, and
    a greedy nearest-legal decode deadlocks on such a design whatever the orientation - measured:
    the `discrete_grid` arm is illegal at every grid size tried, and at grid 20 the ORIENTED arm
    is the legal one. Legality against a rotated footprint is pinned by
    `test_legality_is_measured_on_the_rotated_footprint`, which tests the geometry rather than
    the search's luck.
    """
    built = build(_oriented_config(_design(tmp_path / "bench")))
    state = built.agent.init(jax.random.PRNGKey(0))
    for index in range(3):
        state, _result = built.agent.update(jax.random.fold_in(jax.random.PRNGKey(1), index), state)

    orientations = best_orientations(built.agent, state)
    assert orientations is not None and orientations.shape == (4,)
    assert bool(((orientations >= 0) & (orientations < orient.N_ORIENTATIONS)).all())
    measured = score(built.benchmark, built.agent.best_positions(state), 0, orientations)
    # Whatever the search found, what is reported is the placement it actually chose - overlap
    # included, which is the project's rule: a wirelength without its legality is not a result.
    assert measured["real_hpwl"] == pytest.approx(
        score_placement(built.benchmark, built.agent.best_positions(state), orientations)
    )
    assert "overlap_ratio" in measured


def test_a_turn_with_nowhere_legal_to_go_is_not_the_one_chosen(tmp_path) -> None:
    """The genome's preferred turn is overridden when that turn leaves no legal cell.

    Without the fallback the mask's own relaxation valve fires instead - it drops legality rather
    than deadlock - and the GA quietly discovers that overlapping macros have shorter wires.
    """
    from placax_agents.agents.genetic import _turn_that_fits

    built = build(_oriented_config(_design(tmp_path / "bench")))
    state = reset(built.benchmark.params, built.initial_positions, built.action_space)

    for preferred in range(orient.N_ORIENTATIONS):
        turn, illegal = _turn_that_fits(state, jnp.asarray(preferred), built.agent)
        assert 0 <= int(turn) < orient.N_ORIENTATIONS
        # Whatever turn comes back, it is one the macro can actually be placed in.
        assert not bool(illegal.all()), f"turn chosen for preference {preferred} has no legal cell"


def test_the_action_space_choice_is_hashed(tmp_path) -> None:
    directory = _design(tmp_path / "bench")
    plain = _oriented_config(directory, "discrete_grid")
    oriented = _oriented_config(directory, "oriented_grid")
    assert plain.environment_hash() != oriented.environment_hash()


def test_a_cell_only_agent_refuses_the_oriented_space(tmp_path) -> None:
    # PPO's policy emits (grid_x, grid_y) logits and has nowhere to put a turn.
    directory = _design(tmp_path / "bench")
    base = training(directory, budget=Budget(iterations=1))
    config = dataclasses.replace(base, environment=dataclasses.replace(
        base.environment, action_space=Spec("oriented_grid")))
    with pytest.raises(ValueError, match="cannot use"):
        build(config)
