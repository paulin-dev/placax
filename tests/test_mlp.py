"""The coordinate arm of the state-representation study, which shipped with no tests at all.

`POLICIES["mlp"]` is docs §12's "raw coordinates vs. image (CNN) vs. graph (GNN)" comparison's
non-CNN arm, and nothing exercised it: no test file mentioned it, so the one architecture whose
input is hand-assembled from five separate observation keys was also the one nothing would catch
breaking. It is also the architecture most exposed to a change in the observation, since it reads
`positions`, `placed_mask`, `step`, `current_macro_size` and `lookahead_sizes` directly.

What these pin down: the shapes and the mask-compatible output, that it actually reads the
coordinates it claims to, that the unplaced sentinel never reaches the network, and that it
trains under all three loop shapes.
"""
import dataclasses
import pathlib

from placax.core import reset  # noqa: F401  must precede jax imports
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build
from placax_agents.experiment.config import Spec
from placax_agents.experiment.presets import training
from placax_agents.experiment.registry import POLICIES
from placax_agents.experiment.run import run_experiment
from placax_agents.policy.architectures.mlp import MLPActorCritic

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


@pytest.fixture
def toy(tmp_path: pathlib.Path):
    """A 4-macro design on a 12-cell canvas, built with the MLP policy."""
    directory = tmp_path / "bench"
    directory.mkdir()
    (directory / "s.aux").write_text("RowBasedPlacement : s.nodes s.nets s.wts s.pl s.scl\n")
    (directory / "s.nodes").write_text(NODES)
    (directory / "s.nets").write_text(NETS)

    def make(loop: Spec = Spec("sequential"), iterations: int = 2):
        config = training(directory, budget=Budget(iterations=iterations))
        config = dataclasses.replace(
            config,
            environment=dataclasses.replace(config.environment, benchmark=dataclasses.replace(
                config.environment.benchmark, grid=12)),
            agent=dataclasses.replace(
                config.agent, policy=Spec("mlp", {"features": 16, "num_layers": 1}), loop=loop),
        )
        return config, build(config)

    return make


def _observation(built):
    return built.state_fn(
        reset(built.benchmark.params, built.initial_positions),
        built.benchmark.params, built.benchmark.sizes_array,
    )


def test_the_mlp_emits_the_same_logits_map_every_other_architecture_does(toy) -> None:
    # It has to drop into the identical masking, sampling and loss path, so its output shape is
    # the canvas's - that is what makes this a study of the REPRESENTATION rather than of two
    # different action spaces wearing one name.
    _config, built = toy()
    obs = _observation(built)
    variables = built.policy.init(random.PRNGKey(0), obs)
    logits, value = built.policy.apply(variables, obs)
    params = built.benchmark.params
    assert logits.shape == (params.grid_x, params.effective_grid_y)
    assert value.shape == ()


def test_the_mlp_reads_the_coordinates_it_claims_to(toy) -> None:
    # The point of the arm: this architecture consumes `positions`, not the canvas image. Moving
    # a placed macro must move its logits, or it is not reading what the comparison says it does.
    _config, built = toy()
    obs = _observation(built)
    placed = dict(obs)
    placed["positions"] = obs["positions"].at[0].set(jnp.array([1, 1]))
    placed["placed_mask"] = obs["placed_mask"].at[0].set(True)
    moved = dict(placed)
    moved["positions"] = placed["positions"].at[0].set(jnp.array([9, 9]))

    variables = built.policy.init(random.PRNGKey(0), obs)
    here, _ = built.policy.apply(variables, placed)
    there, _ = built.policy.apply(variables, moved)
    assert not jnp.allclose(here, there)


def test_the_unplaced_sentinel_never_reaches_the_network(toy) -> None:
    """-1 would be the largest-magnitude input in the vector; the placed mask carries the meaning.

    An unplaced macro's row is zeroed, so changing the coordinates stored against an UNPLACED
    macro cannot change the policy's output at all.
    """
    _config, built = toy()
    obs = _observation(built)
    variables = built.policy.init(random.PRNGKey(0), obs)

    unplaced = dict(obs)
    unplaced["positions"] = obs["positions"].at[0].set(jnp.array([7, 7]))
    assert not bool(obs["placed_mask"][0]), "fixture must start with macro 0 unplaced"

    baseline, _ = built.policy.apply(variables, obs)
    with_garbage, _ = built.policy.apply(variables, unplaced)
    assert jnp.allclose(baseline, with_garbage)


def test_the_mlp_is_selectable_from_a_config_and_hashes_as_itself(toy) -> None:
    config, built = toy()
    assert isinstance(built.policy, MLPActorCritic)
    assert "mlp" in POLICIES
    # The study runs on the POLICY axis, so two arms share an environment exactly - which is a
    # stronger claim than the spec anticipated, and worth asserting rather than describing.
    cnn = dataclasses.replace(config, agent=dataclasses.replace(
        config.agent, policy=Spec("cnn", {"features": 16, "num_conv_layers": 2})))
    assert cnn.environment_hash() == config.environment_hash()
    assert cnn.full_hash() != config.full_hash()


@pytest.mark.parametrize("loop", [
    Spec("sequential"),
    Spec("parallel", {"n_envs": 2}),
    Spec("buffered", {"n_episodes": 4, "ppo_epochs": 2, "batch_size": 4}),
])
def test_the_mlp_trains_under_every_loop_shape(toy, loop: Spec) -> None:
    # The riskiest thing about a hand-assembled input vector is a loop that batches differently,
    # so all three are driven rather than just the default one.
    config, built = toy(loop=loop)
    _state, log = run_experiment(config, None, built=built, eval_every=1, log_every=10 ** 9)
    assert len(log) == 2
    assert log[-1]["loss"] is not None
    assert log[-1]["real_hpwl"] is not None


def test_the_size_scale_normalizes_rather_than_dividing_by_zero() -> None:
    # size_scale comes from the design's largest macro; a degenerate 0.0 must not produce NaNs.
    policy = MLPActorCritic(grid_x=4, grid_y=4, size_scale=0.0, features=8, num_layers=1)
    obs = {
        "positions": jnp.full((2, 2), -1, dtype=jnp.int32),
        "placed_mask": jnp.zeros((2,), dtype=bool),
        "step": jnp.array(0),
        "current_macro_size": jnp.zeros((2,)),
        "lookahead_sizes": jnp.zeros((1, 2)),
    }
    logits, value = policy.apply(policy.init(random.PRNGKey(0), obs), obs)
    assert jnp.all(jnp.isfinite(logits))
    assert jnp.isfinite(value)


def test_the_mlp_is_vmappable_like_every_other_policy() -> None:
    # The parallel and buffered loops vmap the whole rollout, so the module has to tolerate being
    # traced under a batch transform rather than only being called one observation at a time.
    policy = MLPActorCritic(grid_x=4, grid_y=4, size_scale=4.0, features=8, num_layers=1)
    one = {
        "positions": jnp.zeros((2, 2), dtype=jnp.int32),
        "placed_mask": jnp.ones((2,), dtype=bool),
        "step": jnp.array(1),
        "current_macro_size": jnp.ones((2,)),
        "lookahead_sizes": jnp.ones((1, 2)),
    }
    variables = policy.init(random.PRNGKey(0), one)
    batch = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), one)
    logits, values = jax.vmap(lambda obs: policy.apply(variables, obs))(batch)
    assert logits.shape == (2, 4, 4)
    assert values.shape == (2,)
