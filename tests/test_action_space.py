"""The ActionSpace seam, and the local-search agent that proves it is real.

`step()` used to hold the whole decision in one line - one macro per step, in array order, at an
integer grid cell, ending after n_macros of them - in a project whose stated purpose is to be the
environment several paradigms are compared in. These tests cover both directions of the fix: the
default behaviour is *unchanged* (every existing result stays meaningful), and a second space
genuinely works end to end rather than merely satisfying an interface.

The load-bearing pair is `test_the_default_space_reproduces_the_old_kernel_exactly` and
`test_local_search_improves_a_placement_the_constructive_kernel_could_not_touch`. The first says
the seam cost nothing; the second says it bought something.
"""
import dataclasses
import pathlib

import pytest

from placax.action_space import (  # noqa: F401  must precede jax imports
    DISCRETE_GRID, ActionSpace, DiscreteGridPlacement, Perturbation,
)
from placax.core import replay, reset, step
from placax.types import EnvParams
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build
from placax_agents.experiment.config import AgentSpec, Spec, assert_comparable
from placax_agents.experiment.presets import training
from placax_agents.experiment.run import run_experiment, score

import jax
import jax.numpy as jnp
from flax import linen as nn

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


def _zero(_old_positions, _new_positions, _old_placed, _new_placed):
    return jnp.array(0.0)


PARAMS = EnvParams(grid=8, n_macros=4)


# ------------------------------------------------------- the default costs nothing


def test_the_default_space_reproduces_the_old_kernel_exactly() -> None:
    """The historical transition, asserted line for line against what the kernel now delegates.

    Every result this project has produced came out of `positions.at[state.step].set(action)`
    ending at `n_macros`. If the seam changed that even slightly, it would silently invalidate
    them - so the old behaviour is pinned here rather than trusted to a refactor.
    """
    state = reset(PARAMS)
    assert state.positions.tolist() == [[-1, -1]] * 4 and int(state.step) == 0

    positions = jnp.full((4, 2), -1)
    for index in range(4):
        action = jnp.array([index, index + 1])
        state, _reward, done = step(state, action, _zero, PARAMS)
        positions = positions.at[index].set(action)
        assert state.positions.tolist() == positions.tolist()
        assert int(state.step) == index + 1
        assert bool(done) == (index == 3)


def test_the_default_is_used_when_no_space_is_named() -> None:
    # Nothing that does not ask for another space should be able to tell the seam exists.
    assert reset(PARAMS).positions.tolist() == DISCRETE_GRID.reset(PARAMS, None).positions.tolist()
    assert training("benchmarks/adaptec1").environment.action_space == Spec("discrete_grid")


def test_a_warm_start_still_resumes_where_it_left_off() -> None:
    warm = jnp.array([[1, 1], [2, 2], [-1, -1], [-1, -1]])
    state = reset(PARAMS, warm)
    assert int(state.step) == 2
    state, _reward, _done = step(state, jnp.array([5, 5]), _zero, PARAMS)
    # The prefix is untouched and the new macro lands at index 2, not at 0.
    assert state.positions.tolist() == [[1, 1], [2, 2], [5, 5], [-1, -1]]


# ------------------------------------------------------- the second space is real


def test_perturbation_moves_a_macro_the_action_names() -> None:
    space = Perturbation(n_moves=3)
    warm = jnp.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    state = reset(PARAMS, warm, space)
    # (macro, x, y): macro 2 moves, and nothing else does - which the constructive kernel, writing
    # only at state.step, has no action that can express.
    state, _reward, done = step(state, jnp.array([2, 7, 7]), _zero, PARAMS, space)
    assert state.positions.tolist() == [[0, 0], [1, 1], [7, 7], [3, 3]]
    assert not bool(done)


def test_a_perturbation_episode_ends_on_its_move_budget_not_the_macro_count() -> None:
    # The other half of what state.step used to conflate: how many actions have been taken is not
    # how many macros there are.
    space = Perturbation(n_moves=2)
    state = reset(PARAMS, jnp.zeros((4, 2), dtype=jnp.int32), space)
    state, _r, first = step(state, jnp.array([0, 1, 1], dtype=jnp.int32), _zero, PARAMS, space)
    state, _r, second = step(state, jnp.array([1, 2, 2], dtype=jnp.int32), _zero, PARAMS, space)
    assert not bool(first) and bool(second)
    assert space.episode_length(PARAMS, 0) == 2


def test_perturbation_refuses_an_empty_canvas() -> None:
    # There is nothing to move. Refusing beats silently perturbing -1 sentinels into a placement.
    with pytest.raises(ValueError, match="already placed"):
        reset(PARAMS, None, Perturbation())


def test_target_is_undefined_for_a_space_that_has_no_current_macro() -> None:
    """The crux of the generalization, pinned.

    `state.step` meant both "how many actions taken" and "which macro is next". A perturbation
    space can answer the first and not the second - the macro is part of an action nobody has
    chosen yet - so it returns a sentinel rather than a plausible index that would silently feed
    a constructive observation the wrong macro's size.
    """
    state = reset(PARAMS, jnp.zeros((4, 2), dtype=jnp.int32), Perturbation())
    assert int(DISCRETE_GRID.target(reset(PARAMS), PARAMS)) == 0
    assert int(Perturbation().target(state, PARAMS)) == -1


def test_every_space_satisfies_the_protocol() -> None:
    assert isinstance(DiscreteGridPlacement(), ActionSpace)
    assert isinstance(Perturbation(), ActionSpace)


def test_a_random_action_has_the_shape_of_its_own_space() -> None:
    # The kernel does not know what an action looks like; the space does.
    assert DISCRETE_GRID.random_action(jax.random.PRNGKey(0), PARAMS).shape == (2,)
    assert Perturbation().random_action(jax.random.PRNGKey(0), PARAMS).shape == (3,)


def test_replay_still_drives_the_default_space() -> None:
    positions = jnp.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    assert float(replay(positions, _zero, PARAMS)) == 0.0


# ------------------------------------------------------- through a config


def _perturbation_config(directory, **agent_kwargs):
    base = training(directory, budget=Budget(iterations=2))
    return dataclasses.replace(
        base,
        environment=dataclasses.replace(
            base.environment,
            benchmark=dataclasses.replace(base.environment.benchmark, grid=12),
            action_space=Spec("perturbation", {"n_moves": 8}),
            initial_placement=Spec("greedy_wiremask_prefix", {"n_macros": 4}),
            # Local search reads the per-step reward as its improvement signal, so it needs a
            # dense one - a terminal-only reward gives it nothing to climb.
            reward=Spec("hpwl", {"dense": True}),
        ),
        agent=AgentSpec(algorithm=Spec("local_search", agent_kwargs)),
    )


def test_the_action_space_is_part_of_the_environment_hash(tmp_path) -> None:
    """Two agents compared on one task must be moving macros under the same rules.

    That is why it sits in the environment half: if it were the agent's, one competitor could
    change what an action means and `assert_comparable` would still call the two runs comparable.
    """
    directory = _design(tmp_path / "bench")
    constructive = training(directory, budget=Budget(iterations=1))
    perturbing = dataclasses.replace(constructive, environment=dataclasses.replace(
        constructive.environment, action_space=Spec("perturbation")))
    assert constructive.environment_hash() != perturbing.environment_hash()
    assert constructive.task_hash() != perturbing.task_hash()
    with pytest.raises(ValueError, match="action_space"):
        assert_comparable(constructive, perturbing)


def test_an_agent_that_chooses_a_cell_refuses_a_space_that_wants_a_macro(tmp_path) -> None:
    # A policy emitting (grid_x, grid_y) logits has no way to name a macro. Feeding its 2-vector
    # to a space expecting (macro, x, y) would place macros at coordinates read off a macro index.
    directory = _design(tmp_path / "bench")
    base = training(directory, budget=Budget(iterations=1))
    config = dataclasses.replace(base, environment=dataclasses.replace(
        base.environment, action_space=Spec("perturbation")))
    with pytest.raises(ValueError, match="produces an action this action space cannot use"):
        build(config)


def test_local_search_refuses_a_constructive_space(tmp_path) -> None:
    # A PARTIAL warm start, so the "nothing left to do" guard passes and the agent's own check is
    # what fires - the two refusals are different and both are wanted.
    directory = _design(tmp_path / "bench")
    base = _perturbation_config(directory)
    config = dataclasses.replace(base, environment=dataclasses.replace(
        base.environment, action_space=Spec("discrete_grid"),
        initial_placement=Spec("greedy_wiremask_prefix", {"n_macros": 2})))
    with pytest.raises(ValueError, match="only the 'perturbation' action space"):
        build(config)


def test_a_constructive_space_still_refuses_a_warm_start_that_placed_everything(tmp_path) -> None:
    # The original guard, kept: under discrete_grid a full placement really does leave nothing to
    # append, and the message now names the space so the two cases are told apart.
    directory = _design(tmp_path / "bench")
    base = _perturbation_config(directory)
    config = dataclasses.replace(base, environment=dataclasses.replace(
        base.environment, action_space=Spec("discrete_grid")))
    with pytest.raises(ValueError, match="nothing to do under the 'discrete_grid'"):
        build(config)


def test_a_full_initial_placement_is_required_rather_than_rejected(tmp_path) -> None:
    """"Everything is placed" is a mistake constructively and a REQUIREMENT here.

    The guard used to be unconditional, so the one initial placement a perturbation space needs
    was the one the builder refused.
    """
    built = build(_perturbation_config(_design(tmp_path / "bench")))
    assert built.n_placed == built.benchmark.params.n_macros
    assert built.steps_per_episode == 8  # the move budget, not the macros left to place


def test_local_search_improves_a_placement_the_constructive_kernel_could_not_touch(tmp_path) -> None:
    """The payoff: a complete placement gets better, by moving macros already committed.

    No constructive agent can do this at all - once a macro is down, `positions.at[state.step]`
    never revisits it.
    """
    built = build(_perturbation_config(_design(tmp_path / "bench")))
    state = built.agent.init(jax.random.PRNGKey(0))
    start = score(built.benchmark, state["best_positions"], built.n_placed)["real_hpwl"]
    for index in range(4):
        state, _result = built.agent.update(jax.random.fold_in(jax.random.PRNGKey(1), index), state)
    end = score(built.benchmark, built.agent.best_positions(state), built.n_placed)
    assert end["real_hpwl"] <= start
    assert end["is_legal"], "a move must land somewhere legal, or the search cheated"


def test_hill_climbing_never_accepts_a_worsening_move(tmp_path) -> None:
    # temperature=0 is strict descent: the episode's total improvement cannot be negative.
    built = build(_perturbation_config(_design(tmp_path / "bench"), temperature=0.0))
    state = built.agent.init(jax.random.PRNGKey(0))
    for index in range(3):
        state, result = built.agent.update(jax.random.fold_in(jax.random.PRNGKey(2), index), state)
        assert result.metrics["episode_improvement"] >= 0.0


def test_a_proposed_move_lifts_the_macro_before_asking_where_it_may_go(tmp_path) -> None:
    """Without lifting, a macro blocks every cell it occupies and can only "move" onto itself.

    Silent and total: the search would report a healthy acceptance rate while never relocating
    anything, because the only legal target is where it already is.
    """
    built = build(_perturbation_config(_design(tmp_path / "bench")))
    state = reset(built.benchmark.params, built.initial_positions, built.action_space)
    macro = jnp.asarray(0)
    lifted = built.agent._illegal_for_macro(state, macro)
    here = state.positions[macro]
    assert not bool(lifted[here[0], here[1]]), "its own cell must be legal once it is lifted"
    assert not bool(lifted.all()), "some cell must be available to move to"


def test_a_local_search_run_leaves_the_same_evidence_as_any_other(tmp_path) -> None:
    directory = _design(tmp_path / "bench")
    output = tmp_path / "run"
    _state, log = run_experiment(_perturbation_config(directory), output, eval_every=1)
    assert (output / "manifest.json").exists()
    assert log and log[-1]["gradient_steps"] == 0
    assert "acceptance_rate" in log[-1]


# ------------------------------------------- the space reaches the LEARNER, not only the agents


class _TinyPolicy(nn.Module):
    """A minimal actor-critic emitting the usual (grid_x, grid_y) logits map."""

    grid: int

    @nn.compact
    def __call__(self, obs):
        x = obs["canvas"].astype(jnp.float32).ravel()
        logits = nn.Dense(self.grid * self.grid)(x).reshape(self.grid, self.grid)
        return logits, nn.Dense(1)(x)[0]


def _zero_reward(_a, _b, _c, _d, _orientations=None):
    return jnp.array(0.0)


def test_the_ppo_rollout_drives_the_configured_action_space() -> None:
    """`collect_rollout` and `evaluate` used to call reset()/step() with no space at all.

    So the whole PPO path was silently `discrete_grid` whatever a config said, and the one axis
    the kernel was generalized for could never be studied with the learner. The space is threaded
    through now; a recording space proves the rollout actually drives it.
    """
    from placax_agents.policy.observation import observation
    from placax_agents.training.rollout import collect_rollout

    @dataclasses.dataclass(frozen=True)
    class PinnedToOrigin(DiscreteGridPlacement):
        """Both halves of what the rollout must take from a space: its transition, and its length.

        Asserted through the trajectory rather than through a Python counter - `lax.scan` traces
        its body once, so a side effect inside `apply` fires once however many steps run.
        """

        def apply(self, state, action, params):
            return DiscreteGridPlacement.apply(self, state, jnp.zeros_like(action), params)

        def episode_length(self, params, n_placed):
            return 2   # deliberately NOT n_macros - n_placed, which is what the rollout assumed

    params = EnvParams(grid=6, n_macros=3)
    sizes = jnp.ones((3, 2))
    policy = _TinyPolicy(grid=6)
    variables = policy.init(
        jax.random.PRNGKey(0), observation(reset(params), params, sizes, cell_size=1.0)
    )
    trajectory, final_state = collect_rollout(
        jax.random.PRNGKey(0), variables, policy.apply, params, _zero_reward, sizes, 1.0,
        action_space=PinnedToOrigin(),
    )
    assert trajectory["action"].shape[0] == 2, "the rollout sized the episode itself"
    assert final_state.positions[:2].tolist() == [[0, 0], [0, 0]], (
        "the rollout applied its own transition instead of the space's"
    )


def test_the_rollout_length_comes_from_the_space_not_from_the_macro_count() -> None:
    # A perturbation episode is a move budget and has nothing to do with how many macros exist,
    # so `n_macros - n_placed` was the wrong scan length the moment a second space existed.
    params = EnvParams(grid=6, n_macros=3)
    assert Perturbation(n_moves=9).episode_length(params, n_placed=3) == 9
    assert DISCRETE_GRID.episode_length(params, n_placed=1) == 2


def test_a_policy_declares_which_action_spaces_its_output_can_express() -> None:
    """Read from the policy rather than hard-coded against the agent's name.

    A `(grid_x, grid_y)` logits map is exactly one grid cell, so every shipped architecture drives
    `discrete_grid` and says so. A policy with a turn axis declares more - and then has to supply
    per-turn legality, which `legal_action_logits` insists on rather than broadcasting a
    position-only mask over an axis it never checked.
    """
    from placax_agents.experiment.build import CELL_ONLY, policy_action_spaces
    from placax_agents.policy.action import legal_action_logits
    from placax_agents.policy.architectures.cnn import CNNActorCritic

    assert policy_action_spaces(CNNActorCritic(features=4, num_conv_layers=1)) == CELL_ONLY

    class Oriented(CNNActorCritic):
        action_spaces = ("discrete_grid", "oriented_grid")

    assert "oriented_grid" in policy_action_spaces(Oriented(features=4, num_conv_layers=1))

    # ...and a three-dimensional logits map is refused by the mask rather than silently stretched.
    with pytest.raises(ValueError, match="legality computed FOR that axis"):
        legal_action_logits(
            jnp.zeros((4, 4, 4)), jnp.zeros((4, 4), dtype=bool), EnvParams(grid=4, n_macros=1),
            (1, 1),
        )


def test_an_action_mask_that_needs_a_current_macro_is_refused_without_one(tmp_path) -> None:
    """`ActionSpace.target()` finally decides something.

    `wiremask_quality` reads `current_macro_size` and a wiremask baseline built from "the first
    `state.step` macros are down". Under a perturbation space `state.step` is a move counter, so
    both describe an arbitrary macro - and JAX clamps the out-of-range index rather than raising,
    so the mask came back looking perfectly well-formed while masking for the wrong macro.
    """
    from placax_agents.experiment.presets import maskplace

    directory = _design(tmp_path / "masked")
    config = maskplace(directory, budget=Budget(iterations=1))
    config = dataclasses.replace(config, environment=dataclasses.replace(
        config.environment,
        benchmark=dataclasses.replace(config.environment.benchmark, grid=24),
        action_space=Spec("perturbation", {"n_moves": 2}),
        initial_placement=Spec("greedy_wiremask_prefix", {"n_macros": None}),
    ), agent=AgentSpec(algorithm=Spec("local_search")))
    assert config.environment.action_mask is not None, "the fixture must actually constrain"

    with pytest.raises(ValueError, match="has no such macro"):
        build(config)

    # Dropping the mask - the environment's own choice - is what makes the pairing runnable.
    without = dataclasses.replace(config, environment=dataclasses.replace(
        config.environment, action_mask=None))
    assert build(without).agent.name == "local_search"
