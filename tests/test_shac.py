"""SHAC end to end - the comparison this project was founded to run, finally runnable.

The spec's §13 frames the whole project on one question: does an analytic policy gradient beat PPO
on placement, or does it not? Answering it needed three environment-level pieces, and these tests
cover the seam where they meet rather than the pieces themselves (`test_density.py` pins the
density term's arithmetic; `test_action_space.py` pins the spaces).

What has to be true for a SHAC result to mean anything:

  * the gradient must reach the parameters at all - a continuous action exists precisely because
    `d(action)/d(theta)` through a categorical does not;
  * it must span the HORIZON, not one step. A "short-horizon" method whose window is silently one
    step is just a myopic policy gradient with extra machinery, and it would look identical from
    outside;
  * the environment must refuse the combinations that cannot work, rather than running them - a
    mask that cannot mask, a reward whose legality has no gradient;
  * and the placement it produces has to be measured for legality like everyone else's, because
    continuous placements are NOT legal by construction the way masked ones are.
"""
import dataclasses
import pathlib

import pytest

from placax.action_space import ContinuousPlacement  # noqa: F401  must precede jax imports
from placax.core import reset
from placax_agents.agents.shac import SHACAgent, _loss, _windowed_return
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build
from placax_agents.experiment.config import AgentSpec, Spec
from placax_agents.experiment.presets import training
from placax_agents.experiment.run import run_experiment
from placax_agents.policy.architectures.continuous import (
    ContinuousActorCritic, mean_action, sample_continuous,
)

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


def _shac(directory, *, grid: int = 24, iterations: int = 3, horizon: int = 2,
          density_weight: float = 1.0, target_density: float = 1.0):
    config = training(directory, budget=Budget(iterations=iterations))
    return dataclasses.replace(
        config,
        environment=dataclasses.replace(
            config.environment,
            benchmark=dataclasses.replace(config.environment.benchmark, grid=grid),
            action_space=Spec("continuous"),
            action_mask=None,
            reward=Spec("differentiable", {"density_weight": density_weight,
                                           "target_density": target_density}),
        ),
        agent=AgentSpec(
            algorithm=Spec("shac", {"horizon": horizon}),
            policy=Spec("continuous", {"features": 32, "num_layers": 2}),
            optimizer=Spec("adam", {"learning_rate": 3e-3}),
        ),
    )


# --------------------------------------------------------------- the gradient exists


def test_the_gradient_reaches_the_policy_through_the_placement(tmp_path) -> None:
    """The property every other agent here cannot have.

    PPO's gradient comes from a score-function estimator - the environment is a black box and
    `d(reward)/d(action)` is never formed. This asserts the other thing: differentiating the
    episode's return with respect to the policy's parameters produces something nonzero, which is
    only possible because the action is continuous, the wirelength is smoothed and legality is a
    cost rather than a mask.
    """
    built = build(_shac(_design(tmp_path / "bench")))
    agent = built.agent
    state = agent.init(random.PRNGKey(0))
    grads = jax.grad(_loss, has_aux=True)(
        state["variables"], agent, agent.policy.apply, random.PRNGKey(1)
    )[0]
    total = sum(
        float(jnp.abs(leaf).sum()) for leaf in jax.tree_util.tree_leaves(grads["params"])
    )
    assert total > 0.0


def test_the_gradient_spans_the_horizon_rather_than_one_step(tmp_path) -> None:
    """"Short-horizon" has to mean something, and this is the only place it is visible.

    The window is implemented by cutting the gradient path through the carried placement every H
    steps. If that cut happened every step - or never - the method would still run, still train
    and still look exactly like this from outside. So: the same episode, same seed, same weights,
    differentiated at two horizons must produce two different gradients, and a longer horizon must
    see more of the episode than a shorter one.
    """
    config = _shac(_design(tmp_path / "bench"))
    built = build(config)
    variables = built.agent.init(random.PRNGKey(0))["variables"]

    def gradient_at(horizon: int):
        agent = dataclasses.replace(config, agent=dataclasses.replace(
            config.agent, algorithm=Spec("shac", {"horizon": horizon})))
        one = build(agent, benchmark=built.benchmark).agent
        grads = jax.grad(_loss, has_aux=True)(
            variables, one, one.policy.apply, random.PRNGKey(1)
        )[0]
        return jnp.concatenate([
            jnp.ravel(leaf) for leaf in jax.tree_util.tree_leaves(grads["params"])
        ])

    short, long = gradient_at(1), gradient_at(4)
    assert not jnp.allclose(short, long), (
        "the horizon changed nothing, so backpropagation through time is not happening"
    )


def test_a_window_return_bootstraps_at_its_boundary_and_not_before() -> None:
    """The other half of the truncation: the RETURN must restart where the gradient was cut.

    Inside a window the return accumulates the steps that follow; at a boundary it takes the
    critic's estimate instead, so a window never sees the next window's rewards. Worked by hand on
    four steps with a horizon of two.
    """
    rewards = jnp.array([1.0, 2.0, 4.0, 8.0])
    values = jnp.array([10.0, 20.0, 40.0, 80.0])
    boundaries = jnp.array([False, True, False, True])
    returns = _windowed_return(rewards, values, boundaries, gamma=0.5)

    # Step 3 is a boundary: r + gamma * V(3) = 8 + 0.5 * 80.
    assert float(returns[3]) == pytest.approx(48.0)
    # Step 2 is inside the window, so it accumulates step 3's return: 4 + 0.5 * 48.
    assert float(returns[2]) == pytest.approx(28.0)
    # Step 1 is a boundary again - it must NOT see step 2. 2 + 0.5 * V(1) = 2 + 10.
    assert float(returns[1]) == pytest.approx(12.0)
    assert float(returns[0]) == pytest.approx(1.0 + 0.5 * 12.0)


# --------------------------------------------------------------- the policy


def test_sampling_is_reparameterized_so_the_action_carries_a_gradient() -> None:
    # `mean + sigma * eps`: the sample is a differentiable function of the parameters, which a
    # categorical draw over grid cells is not. Without this there is no SHAC.
    params = {"mean": jnp.array([3.0, 4.0]), "log_std": jnp.array([0.0, 0.0])}
    gradient = jax.grad(
        lambda p: sample_continuous(random.PRNGKey(0), p).sum()
    )(params)
    assert float(jnp.abs(gradient["mean"]).sum()) > 0.0
    assert float(jnp.abs(gradient["log_std"]).sum()) > 0.0


def test_the_policys_mean_keeps_the_whole_macro_on_the_canvas(tmp_path) -> None:
    """The continuous counterpart of `boundary_mask`, and a bug found by running it.

    Bounding the CORNER to the canvas lets a policy saturate its sigmoid and put a macro's whole
    body past the far edge - measured, that is what a continuous policy on a crowded canvas learns
    to do, because leaving was cheaper than overlapping. The head bounds the corner to
    `canvas - footprint` instead, so the mean cannot ask for a placement that does not fit.
    """
    policy = ContinuousActorCritic(grid_x=10, grid_y=10, cell_size=1.0, size_scale=4.0,
                                   features=16, num_layers=1)
    obs = {
        "positions": jnp.full((2, 2), -1.0),
        "placed_mask": jnp.zeros((2,), dtype=bool),
        "step": jnp.array(0),
        "current_macro_size": jnp.array([4.0, 3.0]),
        "lookahead_sizes": jnp.ones((1, 2)),
    }
    variables = policy.init(random.PRNGKey(0), obs)
    # Push the head to both extremes by scaling the trunk's output, and check the macro still fits.
    for scale in (-1e4, 1e4):
        pushed = jax.tree_util.tree_map(lambda leaf: leaf * scale, variables)
        action_params, _value = policy.apply(pushed, obs)
        corner = mean_action(action_params)
        assert float(corner[0]) >= 0.0 and float(corner[0] + 4.0) <= 10.0 + 1e-4
        assert float(corner[1]) >= 0.0 and float(corner[1] + 3.0) <= 10.0 + 1e-4


# --------------------------------------------------------------- the environment refuses nonsense


def test_a_mask_is_refused_where_there_are_no_cells_to_mask(tmp_path) -> None:
    """Masking rules CELLS out of a categorical. A continuous action has none.

    Left unchecked, the mask would be built, hashed, written into the manifest and then ignored -
    the environment claiming a constraint it was not applying, which is the exact failure the
    comparison machinery exists to prevent.
    """
    config = _shac(_design(tmp_path / "bench"))
    masked = dataclasses.replace(config, environment=dataclasses.replace(
        config.environment, state=Spec("wiremask", {"lookahead": 2}),
        action_mask=Spec("wiremask_quality", {"margin": 1.0})))
    with pytest.raises(ValueError, match="no cells to rule out"):
        build(masked)


def test_shac_is_refused_a_discrete_space(tmp_path) -> None:
    config = _shac(_design(tmp_path / "bench"))
    discrete = dataclasses.replace(config, environment=dataclasses.replace(
        config.environment, action_space=Spec("discrete_grid")))
    with pytest.raises(ValueError, match="produces an action this action space cannot use"):
        build(discrete)


def test_a_continuous_space_is_refused_to_a_discrete_policy(tmp_path) -> None:
    # The other direction: a (grid_x, grid_y) logits map has nothing to say about a real-valued
    # coordinate, so PPO cannot be dropped into this environment either.
    config = _shac(_design(tmp_path / "bench"))
    ppo = dataclasses.replace(config, agent=AgentSpec(
        algorithm=Spec("ppo"), policy=Spec("cnn"), optimizer=Spec("adam"),
        loop=Spec("sequential")))
    with pytest.raises(ValueError, match="produces an action this action space cannot use"):
        build(ppo)


# --------------------------------------------------------------- end to end


def test_shac_trains_and_reports_what_every_other_agent_reports(tmp_path) -> None:
    config = _shac(_design(tmp_path / "bench"), iterations=4)
    built = build(config)
    state, log = run_experiment(config, None, built=built, eval_every=2, log_every=10 ** 9)

    assert len(log) == 4
    assert log[-1]["loss"] is not None
    assert log[-1]["gradient_steps"] == 4, "one differentiable pass, one optimizer step"
    assert log[-1]["env_steps"] > 0
    evaluated = [entry for entry in log if entry["real_hpwl"] is not None]
    assert evaluated, "the runner must score a SHAC placement like any other"
    # Legality is measured, not assumed - a continuous placement is not legal by construction.
    assert "overlap_ratio" in evaluated[-1] and "is_legal" in evaluated[-1]


def test_a_continuous_placement_is_real_valued_and_still_scores(tmp_path) -> None:
    # The placement is floats, which is the whole point; everything downstream still measures it.
    built = build(_shac(_design(tmp_path / "bench")))
    state = built.agent.init(random.PRNGKey(0))
    positions = built.agent.best_positions(state)
    assert positions.dtype == jnp.float32
    assert not jnp.allclose(positions, jnp.round(positions)), "a continuous action should not land on integers"

    from placax_agents.experiment.run import score

    measured = score(built.benchmark, positions, built.n_placed, None, built.action_space,
                     built.initial_positions)
    assert measured["real_hpwl"] > 0.0
    assert "is_legal" in measured


def test_shac_reduces_its_own_objective(tmp_path) -> None:
    """The honest end-to-end claim: it optimizes what it was given.

    NOT that it produces a legal placement in four iterations on a toy - that is a tuning
    question, and `density_weight` is the knob. What must be true is that the objective it
    descends actually goes down.
    """
    config = _shac(_design(tmp_path / "bench"), iterations=12, grid=24)
    built = build(config)
    _state, log = run_experiment(config, None, built=built, eval_every=0, log_every=10 ** 9)
    first, last = log[0]["loss"], log[-1]["loss"]
    assert last < first, f"loss did not fall: {first} -> {last}"
